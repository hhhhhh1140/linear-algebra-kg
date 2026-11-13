"""张量构造、推理流水线（NER -> 实体链接 -> 关系分类 -> 三元组）与评测指标。"""
from __future__ import annotations

from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from common import NA, NER_LABELS, REL_LABELS, RELATIONS, tags_to_spans
from models import BiLSTMRelation, TYPE_LIST

TAG2ID = {t: i for i, t in enumerate(NER_LABELS)}
REL2ID = {r: i for i, r in enumerate(REL_LABELS)}
TYPE2ID = {t: i for i, t in enumerate(TYPE_LIST)}


class Inputs:
    """统一 char / bert 两种输入：把一批句子变成 (x, mask)。"""

    def __init__(self, mode, vocab=None, featurizer=None):
        self.mode, self.vocab, self.feat = mode, vocab, featurizer

    def __call__(self, texts):
        L = max(len(t) for t in texts)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, t in enumerate(texts):
            mask[i, : len(t)] = 1
        if self.mode == "char":
            x = torch.zeros(len(texts), L, dtype=torch.long)
            for i, t in enumerate(texts):
                x[i, : len(t)] = torch.tensor(self.vocab.encode(t))
        else:
            feats = self.feat.featurize(texts)
            x = torch.zeros(len(texts), L, feats[0].size(-1))
            for i, f in enumerate(feats):
                x[i, : f.size(0)] = f
        return x, mask


# ------------------------------------------------------------------ NER
def ner_tensors(inputs, exs, etype_fn):
    x, mask = inputs([e["text"] for e in exs])
    tags = torch.zeros_like(mask)
    for i, e in enumerate(exs):
        for sp in e["entities"]:
            tags[i, sp["start"]] = TAG2ID[f"B-{sp['type']}"]
            for k in range(sp["start"] + 1, sp["end"]):
                tags[i, k] = TAG2ID[f"I-{sp['type']}"]
    return x, mask, tags


@torch.no_grad()
def ner_predict(model, inputs, texts, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        x, mask = inputs(chunk)
        for t, path in zip(chunk, model.predict(x, mask)):
            out.append(tags_to_spans(t, [NER_LABELS[p] for p in path]))
    return out


# ------------------------------------------------------------------ 关系抽取
def re_instances_from_gold(exs):
    inst = []
    for e in exs:
        for r in e["relations"]:
            h, t = e["entities"][r["head"]], e["entities"][r["tail"]]
            inst.append(dict(text=e["text"], h=h, t=t, label=r["label"]))
    return inst


def re_tensors(inputs, inst):
    x, mask = inputs([d["text"] for d in inst])
    B, L = mask.shape
    pos = torch.arange(L).expand(B, L)
    D = BiLSTMRelation.MAXD

    def rel(s, e):
        s = torch.tensor([v for v in s]).unsqueeze(1)
        e = torch.tensor([v for v in e]).unsqueeze(1)
        d = torch.where(pos < s, pos - s, torch.where(pos >= e, pos - e + 1, torch.zeros_like(pos)))
        return d.clamp(-D, D) + D

    ph = rel([d["h"]["start"] for d in inst], [d["h"]["end"] for d in inst])
    pt = rel([d["t"]["start"] for d in inst], [d["t"]["end"] for d in inst])
    mark = torch.zeros(B, L, dtype=torch.long)
    for i, d in enumerate(inst):
        mark[i, d["h"]["start"]:d["h"]["end"]] = 1
        mark[i, d["t"]["start"]:d["t"]["end"]] = 2
    ht = torch.tensor([TYPE2ID[d["h"]["type"]] for d in inst])
    tt = torch.tensor([TYPE2ID[d["t"]["type"]] for d in inst])
    y = torch.tensor([REL2ID.get(d.get("label", NA), 0) for d in inst])
    return (x, mask, ph, pt, mark, ht, tt), y


def type_ok(rel, htype, ttype):
    _, dom, rng = RELATIONS[rel]
    return htype in dom and ttype in rng


@torch.no_grad()
def re_predict(model, inputs, inst, bs=128):
    """返回每个实例的 (标签, 概率)；不满足实体类型约束的关系被屏蔽。"""
    model.eval()
    res = []
    for i in range(0, len(inst), bs):
        chunk = inst[i:i + bs]
        args, _ = re_tensors(inputs, chunk)
        prob = F.softmax(model(*args), -1)
        for d, p in zip(chunk, prob):
            for j, r in enumerate(REL_LABELS):
                if r != NA and not type_ok(r, d["h"]["type"], d["t"]["type"]):
                    p[j] = 0
            p = p / p.sum()
            j = int(p.argmax())
            res.append((REL_LABELS[j], float(p[j])))
    return res


# ------------------------------------------------------------------ 端到端
def link(span, surface):
    """实体链接：表面形式在词典中则映射到规范名，否则作为新实体。"""
    return surface.get(span["text"], span["text"]), span["text"] not in surface


def extract_triples(ner_model, re_model, inputs, texts, surface, threshold=0.5):
    """生文本 -> [(句子, 实体列表, 三元组列表)]"""
    all_spans = ner_predict(ner_model, inputs, texts)
    inst, owner = [], []
    for si, (text, spans) in enumerate(zip(texts, all_spans)):
        for sp in spans:
            sp["name"], sp["new"] = link(sp, surface)
        for a in spans:
            for b in spans:
                if a is not b and a["name"] != b["name"]:
                    inst.append(dict(text=text, h=a, t=b))
                    owner.append(si)
    preds = re_predict(re_model, inputs, inst) if inst else []
    triples = defaultdict(list)
    for d, si, (lab, p) in zip(inst, owner, preds):
        if lab != NA and p >= threshold:
            triples[si].append(dict(head=d["h"]["name"], relation=lab, tail=d["t"]["name"], score=round(p, 4),
                                    head_type=d["h"]["type"], tail_type=d["t"]["type"],
                                    new_entity=d["h"]["new"] or d["t"]["new"]))
    return [(t, all_spans[i], triples[i]) for i, t in enumerate(texts)]


# ------------------------------------------------------------------ 指标
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), support=tp + fn)


def ner_metrics(golds, preds, held_names=frozenset()):
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])
    held_tp = held_n = 0
    for g, p in zip(golds, preds):
        gs = {(s["start"], s["end"], s["type"]) for s in g}
        ps = {(s["start"], s["end"], s["type"]) for s in p}
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
        for s in gs & ps: per[s[2]][0] += 1
        for s in ps - gs: per[s[2]][1] += 1
        for s in gs - ps: per[s[2]][2] += 1
        for s in g:
            if s.get("name") in held_names:
                held_n += 1
                held_tp += (s["start"], s["end"], s["type"]) in ps
    out = dict(overall=prf(tp, fp, fn), per_type={k: prf(*v) for k, v in sorted(per.items())})
    if held_n:
        out["unseen_entity_recall"] = round(held_tp / held_n, 4)
        out["unseen_entity_mentions"] = held_n
    return out


def re_metrics(golds, preds):
    tp = fp = fn = 0
    per = defaultdict(lambda: [0, 0, 0])
    for g, (p, _) in zip(golds, preds):
        if p == g and g != NA:
            tp += 1; per[g][0] += 1
        else:
            if p != NA:
                fp += 1; per[p][1] += 1
            if g != NA:
                fn += 1; per[g][2] += 1
    return dict(micro=prf(tp, fp, fn), per_relation={k: prf(*v) for k, v in sorted(per.items())},
                confusion=Counter(f"{g}->{p}" for g, (p, _) in zip(golds, preds) if g != p).most_common(8))
