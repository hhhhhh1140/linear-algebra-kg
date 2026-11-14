"""对比四种方案在评测集上的准确率：不检索 / 向量 RAG / GraphRAG / 混合。

分两步执行，第二步不加载任何本地模型，占用资源很少：
  1) 检索：为每道题、每种方案做检索，结果存到 outputs/qa/retrieval.json（已存在则跳过）
  2) 作答：调用云端大模型，逐题把结果追加到 outputs/qa/details.jsonl（中断后重跑会接着做）

用法：python qa/evaluate.py [--modes none vector graph hybrid] [--limit N] [--fresh]
      --fresh 清掉已有的检索和作答记录，从头开始
输出：outputs/qa/results.json（汇总）、results.md（表格）、details.jsonl（逐题记录）
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from kgqa import MODES, ROOT, SYSTEM, Fact, Retrieval, build_prompt
from llm import get_llm

CATS = {"direct": "图谱直接可答", "multihop": "需多跳/推理", "trap": "陷阱题", "outside": "图谱未覆盖"}
ZH = {"none": "不检索", "vector": "向量 RAG", "graph": "GraphRAG", "hybrid": "混合"}
OUT = ROOT / "outputs" / "qa"
RETRIEVAL = OUT / "retrieval.json"
DETAILS = OUT / "details.jsonl"


def load_questions(path: Path = ROOT / "data" / "qa_questions.tsv") -> list[dict]:
    qs = []
    for i, line in enumerate(l for l in path.read_text(encoding="utf-8").splitlines()
                             if l.strip() and not l.startswith("#")):
        cat, typ, q, ans = line.split("\t")
        item = {"id": i + 1, "cat": cat, "type": typ, "question": q}
        if typ == "mc":
            opts = ans.split("|")
            assert len(opts) == 4, line
            order = opts[:]
            random.Random(1000 + i).shuffle(order)      # 固定种子打乱选项顺序
            item["options"] = order
            item["answer"] = "ABCD"[order.index(opts[0])]
        else:
            assert ans in ("是", "否"), line
            item["answer"] = ans
        qs.append(item)
    return qs


def render(item: dict) -> tuple[str, str]:
    """返回 (问题文本, 作答要求)。"""
    if item["type"] == "mc":
        q = item["question"] + "\n" + "\n".join(f"{c}. {o}" for c, o in zip("ABCD", item["options"]))
        return q, "只输出正确选项的字母（A、B、C 或 D），不要输出其他内容。"
    return item["question"], "只输出一个字：是 或 否，不要输出其他内容。"


def parse(item: dict, out: str) -> str:
    m = re.search(r"[ABCD]", out.upper()) if item["type"] == "mc" else re.search(r"[是否]", out)
    return m.group(0) if m else ""


# ------------------------------------------------------------------ 第 1 步：检索
def build_retrieval():
    """在独立进程里运行：加载嵌入模型，做完检索后退出并释放内存。"""
    import torch
    torch.set_num_threads(2)
    from kgqa import Retriever
    retriever = Retriever()
    store = {}
    for item in load_questions():
        g = retriever.graph_search(item["question"])          # 只用题干检索，不用选项
        store[str(item["id"])] = {
            "entities": g.entities, "fuzzy": g.fuzzy,
            "facts": [{"head": f.head, "relation": f.relation, "tail": f.tail, "source": f.source,
                       "reason": f.reason, "score": f.score} for f in g.facts],
            "passages": retriever.vector_search(item["question"]),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    RETRIEVAL.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"检索完成：{len(store)} 道题 -> {RETRIEVAL}")


def retrieval_for(store: dict, qid: int, mode: str) -> Retrieval:
    d = store[str(qid)]
    r = Retrieval()
    if mode in ("graph", "hybrid"):
        r.entities, r.fuzzy = d["entities"], d["fuzzy"]
        r.facts = [Fact(**f) for f in d["facts"]]
    if mode in ("vector", "hybrid"):
        r.passages = d["passages"]
    return r


# ------------------------------------------------------------------ 第 2 步：作答与汇总
def summarize(qs, modes, model):
    rows = [json.loads(l) for l in DETAILS.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = {q["id"] for q in qs}
    rows = [r for r in rows if r["id"] in ids and r["mode"] in modes]
    acc = {m: defaultdict(lambda: [0, 0]) for m in modes}
    tokens = defaultdict(int)
    linked = 0
    for r in rows:
        for key in (r["cat"], "all"):
            acc[r["mode"]][key][0] += r["correct"]
            acc[r["mode"]][key][1] += 1
        tokens[r["mode"]] += r["prompt_tokens"]
        linked += r["mode"] == "graph" and bool(r["entities"]) and not r["fuzzy"]
    done = [m for m in modes if acc[m]["all"][1] == len(qs)]
    summary = {"model": model, "n": len(qs),
               "accuracy": {m: {k: round(v[0] / v[1], 4) for k, v in acc[m].items()} for m in done},
               "counts": {k: v[1] for k, v in acc[done[0]].items()} if done else {},
               "avg_prompt_tokens": {m: round(tokens[m] / len(qs)) for m in done}}
    if "graph" in done:
        summary["entity_link_rate"] = round(linked / len(qs), 4)
    (OUT / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["all"] + list(CATS)
    lines = ["| 方案 | 总体 | " + " | ".join(f"{CATS[c]}（{summary['counts'].get(c, 0)}）" for c in CATS)
             + " | 平均提示词 token |", "|" + "---|" * (len(cols) + 2)]
    for m in done:
        lines.append(f"| {ZH[m]} | " + " | ".join(f"{summary['accuracy'][m].get(c, 0):.1%}" for c in cols)
                     + f" | {summary['avg_prompt_tokens'][m]} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.retrieval_only:
        return build_retrieval()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for p in (RETRIEVAL, DETAILS):
            p.unlink(missing_ok=True)
    if not RETRIEVAL.exists():      # 检索放在子进程里，做完即释放嵌入模型占用的内存
        subprocess.run([sys.executable, __file__, "--retrieval-only"], check=True)
    store = json.loads(RETRIEVAL.read_text(encoding="utf-8"))

    qs = load_questions()[: args.limit]
    llm = get_llm()
    done = set()
    if DETAILS.exists():
        for l in DETAILS.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                if d.get("model") == llm.name:
                    done.add((d["id"], d["mode"]))
    todo = [(m, q) for m in args.modes for q in qs if (q["id"], m) not in done]
    print(f"模型: {llm.name}，题目 {len(qs)} 道，方案 {args.modes}；已完成 {len(done)} 条，待做 {len(todo)} 条")

    t0 = time.time()
    with open(DETAILS, "a", encoding="utf-8") as f:
        for n, (mode, item) in enumerate(todo, 1):
            q, inst = render(item)
            r = retrieval_for(store, item["id"], mode)
            out = llm.chat(SYSTEM, build_prompt(q, r, inst), max_tokens=8)
            pred = parse(item, out)
            f.write(json.dumps({
                "id": item["id"], "cat": item["cat"], "mode": mode, "model": llm.name,
                "question": item["question"], "gold": item["answer"], "pred": pred, "raw": out,
                "correct": pred == item["answer"], "prompt_tokens": llm.last_prompt_tokens,
                "entities": r.entities, "fuzzy": r.fuzzy, "facts": [x.text for x in r.facts],
                "passages": r.passages}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  用时 {time.time() - t0:.0f}s", flush=True)
    summarize(qs, args.modes, llm.name)


if __name__ == "__main__":
    main()
