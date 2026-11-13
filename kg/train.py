"""训练并评测 NER（BiLSTM-CRF）与关系抽取（BiLSTM）模型。

用法：
  python kg/train.py --encoder char   # 随机初始化字向量 + BiLSTM
  python kg/train.py --encoder bert   # 冻结 bert-base-chinese 特征 + BiLSTM
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from common import CORPUS, NA, OUT, load_entities, load_triples, read_jsonl
from models import BertFeaturizer, BiLSTMCRF, BiLSTMRelation, CharVocab
from pipeline import (Inputs, extract_triples, ner_metrics, ner_predict, ner_tensors, re_instances_from_gold,
                      re_metrics, re_predict, re_tensors, prf, type_ok)
from build_corpus import HELD_ENTITIES


def set_seed(s=42):
    random.seed(s); torch.manual_seed(s)


def batches(items, bs, shuffle):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


def train_ner(model, inputs, train, dev, lr, epochs, patience=6):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0
        for b in batches(train, 32, True):
            x, mask, tags = ner_tensors(inputs, b, None)
            loss = model.loss(x, tags, mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item() * len(b)
        preds = ner_predict(model, inputs, [e["text"] for e in dev])
        f1 = ner_metrics([e["entities"] for e in dev], preds)["overall"]["f1"]
        print(f"  [NER] epoch {ep:2d} loss {tot / len(train):.3f} dev F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best


def train_re(model, inputs, train_inst, dev_inst, lr, epochs, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0
        for b in batches(train_inst, 64, True):
            args, y = re_tensors(inputs, b)
            loss = torch.nn.functional.cross_entropy(model(*args), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item() * len(b)
        preds = re_predict(model, inputs, dev_inst)
        f1 = re_metrics([d["label"] for d in dev_inst], preds)["micro"]["f1"]
        print(f"  [RE ] epoch {ep:2d} loss {tot / len(train_inst):.3f} dev micro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["char", "bert"], default="char")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    set_seed()
    torch.set_num_threads(4)

    etype, surface = load_entities()
    seed = load_triples(etype)
    for h, r, t in seed:
        assert type_ok(r, etype[h], etype[t]), f"种子三元组不满足类型约束: {h} {r} {t}"

    train, dev, test = (read_jsonl(CORPUS / f"{k}.jsonl") for k in ("train", "dev", "test"))
    held_triples = {tuple(l.split("\t")) for l in (CORPUS / "held_triples.tsv").read_text(encoding="utf-8").split("\n") if l}

    vocab = CharVocab([e["text"] for e in train])
    feat = None
    if args.encoder == "bert":
        t0 = time.time()
        feat = BertFeaturizer()
        feat.featurize([e["text"] for e in train + dev + test])
        print(f"BERT 特征提取完成，用时 {time.time() - t0:.1f}s，设备 {feat.device}")
    inputs = Inputs(args.encoder, vocab, feat)
    lr = 2e-3 if args.encoder == "char" else 1e-3

    out = OUT / args.encoder
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"encoder": args.encoder}

    # ---------------- NER
    print("== 训练 BiLSTM-CRF 实体识别")
    ner = BiLSTMCRF(args.encoder, len(vocab.itos))
    metrics["ner_dev_f1"] = train_ner(ner, inputs, train, dev, lr, args.epochs)
    preds = ner_predict(ner, inputs, [e["text"] for e in test])
    metrics["ner_test"] = ner_metrics([e["entities"] for e in test], preds, HELD_ENTITIES)
    nat = [i for i, e in enumerate(test) if e["source"] == "natural"]
    metrics["ner_test_natural"] = ner_metrics([test[i]["entities"] for i in nat], [preds[i] for i in nat], HELD_ENTITIES)["overall"]

    # ---------------- RE
    print("== 训练 BiLSTM 关系抽取")
    rel = BiLSTMRelation(args.encoder, len(vocab.itos))
    tr_inst, dv_inst, te_inst = (re_instances_from_gold(s) for s in (train, dev, test))
    metrics["re_dev_f1"] = train_re(rel, inputs, tr_inst, dv_inst, lr, min(args.epochs, 30))
    p = re_predict(rel, inputs, te_inst)
    metrics["re_test_gold_entities"] = re_metrics([d["label"] for d in te_inst], p)
    nat_inst = [i for i, d in enumerate(te_inst) if any(e["text"] == d["text"] and e["source"] == "natural" for e in test)]
    metrics["re_test_natural"] = re_metrics([te_inst[i]["label"] for i in nat_inst], [p[i] for i in nat_inst])["micro"]

    # ---------------- 端到端：生文本 -> 三元组
    res = extract_triples(ner, rel, inputs, [e["text"] for e in test], surface)
    tp = fp = fn = 0
    held_hit, held_total = set(), set()
    for e, (_, _, trs) in zip(test, res):
        gold = {(e["entities"][r["head"]]["name"], r["label"], e["entities"][r["tail"]]["name"])
                for r in e["relations"] if r["label"] != NA}
        pred = {(t["head"], t["relation"], t["tail"]) for t in trs}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
        held_total |= gold & held_triples
        held_hit |= pred & held_triples
    metrics["end2end_test"] = prf(tp, fp, fn)
    metrics["held_out_triples_recovered"] = f"{len(held_hit)}/{len(held_total)}"

    torch.save({"ner": ner.state_dict(), "re": rel.state_dict(), "vocab": vocab.itos}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("ner_test_natural", "re_test_natural", "end2end_test",
                                              "held_out_triples_recovered")}, ensure_ascii=False, indent=1))
    print("NER test:", metrics["ner_test"]["overall"], "unseen-entity recall:", metrics["ner_test"].get("unseen_entity_recall"))
    print("RE  test:", metrics["re_test_gold_entities"]["micro"])


if __name__ == "__main__":
    main()
