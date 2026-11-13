"""用训练好的 NER + 关系模型从文本中抽取三元组，与种子三元组合并，做规则推理，并导出 Neo4j / 可视化。

用法：python kg/build_kg.py [--encoder char|bert] [--threshold 0.8]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import torch

from common import (DATA, ENTITY_TYPES, OUT, RELATIONS, load_entities, load_lines, load_triples)
from models import BertFeaturizer, BiLSTMCRF, BiLSTMRelation, CharVocab
from pipeline import Inputs, extract_triples
from export_neo4j import export_cypher

def load_models(encoder):
    ck = torch.load(OUT / encoder / "model.pt")
    vocab = CharVocab([])
    vocab.itos = ck["vocab"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    feat = BertFeaturizer() if encoder == "bert" else None
    ner = BiLSTMCRF(encoder, len(vocab.itos)); ner.load_state_dict(ck["ner"])
    rel = BiLSTMRelation(encoder, len(vocab.itos)); rel.load_state_dict(ck["re"])
    return ner, rel, Inputs(encoder, vocab, feat)


# ------------------------------------------------------------------ 规则推理
RULES = {
    "R1": "传递性：A is_a B 且 B is_a C ⇒ A is_a C",
    "R2": "性质继承：A is_a B 且 B has_property P ⇒ A has_property P",
    "R3": "等价对称：P equivalent_to Q ⇒ Q equivalent_to P",
    "R4": "等价替换（仅矩阵类实体）：X is_a* 矩阵 且 X has_property P 且 P equivalent_to Q ⇒ X has_property Q",
}


def infer(facts: dict):
    """facts: (h,r,t) -> 来源信息。迭代应用规则直到不再产生新事实。"""
    pairs = {(h, t) for h, _, t in facts}
    new = {}
    changed = True
    while changed:
        changed = False
        cur = set(facts) | set(new)
        by_rel = defaultdict(lambda: defaultdict(set))
        for h, r, t in cur:
            by_rel[r][h].add(t)
        cand = []
        for a, bs in by_rel["is_a"].items():
            for b in bs:
                cand += [((a, "is_a", c), "R1", b) for c in by_rel["is_a"].get(b, ())]
                cand += [((a, "has_property", p), "R2", b) for p in by_rel["has_property"].get(b, ())]
        for p, qs in by_rel["equivalent_to"].items():
            cand += [((q, "equivalent_to", p), "R3", None) for q in qs]
        def is_matrix(x, seen=()):
            return x == "矩阵" or any(is_matrix(b, seen + (x,)) for b in by_rel["is_a"].get(x, ()) if b not in seen)
        for x, ps in by_rel["has_property"].items():
            if not is_matrix(x):
                continue
            for p in ps:
                cand += [((x, "has_property", q), "R4", p) for q in by_rel["equivalent_to"].get(p, ())]
        for tr, rule, via in cand:
            h, _, t = tr
            if h == t or tr in cur or (h, t) in pairs:
                continue
            new[tr] = {"source": "inferred", "rule": rule, "via": via}
            pairs.add((h, t))
            changed = True
    return new


# ------------------------------------------------------------------ 导出
def export_html(path, nodes, edges):
    tpl = (DATA.parent / "kg" / "viewer_template.html").read_text(encoding="utf-8")
    data = {"nodes": nodes, "edges": edges,
            "types": ENTITY_TYPES, "relations": {k: v[0] for k, v in RELATIONS.items()}}
    path.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


def plausible_new_entity(name, known):
    """新实体的质量过滤：至少 3 个字、不含"的"和标点、不是已知实体名的片段。"""
    return (len(name) >= 3 and "的" not in name and not any(c in name for c in "，。、；：")
            and not any(name in k for k in known))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["char", "bert"])
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()
    if not args.encoder:   # 默认选端到端 F1 更高的模型
        scores = {}
        for enc in ("char", "bert"):
            p = OUT / enc / "metrics.json"
            if p.exists():
                scores[enc] = json.loads(p.read_text())["end2end_test"]["f1"]
        args.encoder = max(scores, key=scores.get)
    print(f"使用模型: {args.encoder}")

    etype, surface = load_entities()
    seed = load_triples(etype)
    ner, rel, inputs = load_models(args.encoder)

    known = set(etype)
    natural = load_lines("natural_sentences.txt")
    new_text = load_lines("new_text.txt")
    # 教材句子：只补全已知实体之间的关系；新文本：允许发现新实体（需通过质量过滤）
    results = [(r, False) for r in extract_triples(ner, rel, inputs, natural, surface, threshold=0.5)] + \
              [(r, True) for r in extract_triples(ner, rel, inputs, new_text, surface, threshold=0.5)]

    seed_set = set(seed)
    cand = {}
    new_entities = {}
    for (text, spans, trs), allow_new in results:
        ok_names = set(known)
        if allow_new:
            for sp in spans:
                if sp["new"] and plausible_new_entity(sp["name"], known):
                    new_entities.setdefault(sp["name"], {"type": sp["type"], "sentence": text})
                    ok_names.add(sp["name"])
        for t in trs:
            if t["head"] not in ok_names or t["tail"] not in ok_names:
                continue
            key = (t["head"], t["relation"], t["tail"])
            c = cand.setdefault(key, dict(t, count=0, sentences=[]))
            c["score"] = max(c["score"], t["score"])
            c["count"] += 1
            c["sentences"].append(text)

    kg_dir = OUT / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    with open(kg_dir / "extracted_candidates.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["head", "relation", "relation_zh", "tail", "score", "in_seed", "accepted", "new_entity", "sentence"])
        for key, c in sorted(cand.items(), key=lambda kv: -kv[1]["score"]):
            w.writerow([*key[:2], RELATIONS[key[1]][0], key[2], c["score"], key in seed_set,
                        key not in seed_set and c["score"] >= args.threshold, c["new_entity"], c["sentences"][0]])

    # 合并：种子 + 高置信度抽取结果
    facts = {tr: {"source": "seed"} for tr in seed}
    seed_pairs = {(h, t) for h, _, t in seed}
    for key, c in cand.items():
        if key in facts or c["score"] < args.threshold or (key[0], key[2]) in seed_pairs:
            continue
        facts[key] = {"source": "extracted", "score": c["score"], "sentence": c["sentences"][0]}
        for name, typ in ((key[0], c["head_type"]), (key[2], c["tail_type"])):
            if name not in etype:
                etype[name] = typ
                new_entities.setdefault(name, {"type": typ})

    inferred = infer(facts)
    facts.update(inferred)

    used = {h for h, _, _ in facts} | {t for _, _, t in facts}
    nodes = [{"name": n, "type": etype[n], "source": "extracted" if n in new_entities else "seed"}
             for n in etype if n in used]
    new_entities = {k: v for k, v in new_entities.items() if k in used}
    edges = [{"head": h, "relation": r, "tail": t, **info} for (h, r, t), info in facts.items()]

    (kg_dir / "kg.json").write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(kg_dir / "nodes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "type", "type_zh", "source"])
        for n in nodes:
            w.writerow([n["name"], n["type"], ENTITY_TYPES[n["type"]], n["source"]])
    with open(kg_dir / "edges.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["head", "relation", "relation_zh", "tail", "source", "score", "rule"])
        for e in edges:
            w.writerow([e["head"], e["relation"], RELATIONS[e["relation"]][0], e["tail"], e["source"],
                        e.get("score", ""), e.get("rule", "")])
    with open(kg_dir / "inferred_triples.tsv", "w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\trule\tvia\n")
        for (h, r, t), info in sorted(inferred.items(), key=lambda kv: kv[1]["rule"]):
            f.write(f"{h}\t{r}\t{t}\t{info['rule']}\t{info['via'] or ''}\n")
    export_cypher(kg_dir / "neo4j_import.cypher", nodes, edges)
    export_html(kg_dir / "kg_view.html", nodes, edges)

    src = defaultdict(int)
    for e in edges:
        src[e["source"]] += 1
    new_extracted = [e for e in edges if e["source"] == "extracted"]
    print(f"节点 {len(nodes)}（新发现实体 {sum(n['source'] == 'extracted' for n in nodes)}），边 {len(edges)}：{dict(src)}")
    print("规则推理:", {r: sum(1 for i in inferred.values() if i['rule'] == r) for r in RULES})
    print("\n新抽取的三元组（模型从文本中得到、不在种子里）:")
    for e in sorted(new_extracted, key=lambda e: -e["score"]):
        print(f"  ({e['head']}, {RELATIONS[e['relation']][0]}, {e['tail']})  score={e['score']:.3f}")
    print("\n新发现的实体:", {k: v["type"] for k, v in new_entities.items()})


if __name__ == "__main__":
    main()
