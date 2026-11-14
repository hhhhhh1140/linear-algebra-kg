"""命令行问答演示：显示链接到的实体、检索到的图谱事实和大模型的回答。

用法：
  python qa/ask.py "正定矩阵一定可逆吗？"            # 默认 GraphRAG
  python qa/ask.py --mode hybrid "..."               # none / vector / graph / hybrid
  python qa/ask.py --compare "..."                   # 同一个问题，四种方案并排对比
  python qa/ask.py                                   # 进入交互模式，连续提问
"""
from __future__ import annotations

import argparse

from kgqa import MODES, SRC_ZH, Retriever, answer
from llm import get_llm

ZH = {"none": "不检索", "vector": "向量 RAG", "graph": "GraphRAG", "hybrid": "混合"}


def show(llm, retriever, question, mode):
    out, r, _ = answer(llm, retriever, question, mode)
    print(f"\n━━ {ZH[mode]} ━━")
    if mode in ("graph", "hybrid"):
        how = "（靠向量相似度兜底）" if r.fuzzy else ""
        print("链接到的实体：" + ("、".join(r.entities) if r.entities else "无") + how)
        for f in r.facts:
            print(f"  · {f.text}   [{SRC_ZH[f.source]}｜{f.reason}｜相似度 {f.score:.2f}]")
    if r.passages:
        print("检索到的教材句子：")
        for p in r.passages:
            print(f"  · {p}")
    print("回答：" + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", choices=MODES, default="graph")
    ap.add_argument("--compare", action="store_true", help="四种方案并排对比")
    args = ap.parse_args()

    llm = get_llm()
    retriever = Retriever()
    modes = MODES if args.compare else [args.mode]

    if args.question:
        for m in modes:
            show(llm, retriever, args.question, m)
        return
    print(f"使用模型 {llm.name}。输入问题后回车，直接回车退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except EOFError:
            break
        if not q:
            break
        for m in modes:
            show(llm, retriever, q, m)


if __name__ == "__main__":
    main()
