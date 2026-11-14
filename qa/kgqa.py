"""GraphRAG 数学问答的核心：实体链接 -> 图检索 -> 重排 -> 事实转文本 -> 大模型作答。

四种模式（用于对比实验）：
  none    不检索，直接问大模型
  vector  向量 RAG：从教材句子里检索最相似的几句
  graph   GraphRAG：从知识图谱里检索问题涉及实体的子图
  hybrid  两者都用
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kg"))
from common import ENTITY_TYPES, Matcher, RELATIONS, load_entities, load_lines  # noqa: E402

MODES = ["none", "vector", "graph", "hybrid"]

# 三元组 -> 中文句子
VERBALIZE = {
    "is_a": "{h}是一种{t}",
    "part_of": "{h}是{t}的组成部分",
    "depends_on": "{h}的定义依赖于{t}",
    "has_property": "{h}具有性质：{t}",
    "equivalent_to": "{h}与{t}等价",
    "solved_by": "{h}可以用{t}来求解或判定",
    "acts_on": "{h}的作用对象是{t}",
    "yields": "{h}的结果是{t}",
    "involves": "{h}涉及{t}",
}
SRC_ZH = {"seed": "人工种子", "extracted": "模型抽取", "inferred": "规则推理"}


@dataclass
class Fact:
    head: str
    relation: str
    tail: str
    source: str
    reason: str            # 为什么被检索到：邻居 / 路径 / 上位链
    score: float = 0.0

    @property
    def text(self) -> str:
        return VERBALIZE[self.relation].format(h=self.head, t=self.tail)

    @property
    def key(self):
        return (self.head, self.relation, self.tail)


@dataclass
class Retrieval:
    entities: list[str] = field(default_factory=list)       # 链接到的图谱实体
    fuzzy: bool = False                                      # 是否靠向量相似度兜底链接
    facts: list[Fact] = field(default_factory=list)
    passages: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ 向量模型
class Embedder:
    """bge-small-zh：取 [CLS] 向量并归一化，点积即余弦相似度。"""

    NAME = "BAAI/bge-small-zh-v1.5"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.NAME)
        self.model = AutoModel.from_pretrained(self.NAME).eval()
        self.cache: dict[str, object] = {}

    def encode(self, texts: list[str]):
        torch = self.torch
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            with torch.no_grad():
                enc = self.tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                vec = torch.nn.functional.normalize(self.model(**enc).last_hidden_state[:, 0], dim=-1)
            for t, v in zip(batch, vec):
                self.cache[t] = v
        return torch.stack([self.cache[t] for t in texts])


# ------------------------------------------------------------------ 图谱
class KnowledgeGraph:
    def __init__(self, path: Path = ROOT / "outputs" / "kg" / "kg.json"):
        kg = json.loads(path.read_text(encoding="utf-8"))
        self.type = {n["name"]: n["type"] for n in kg["nodes"]}
        self.edges = kg["edges"]
        self.out, self.inc = defaultdict(list), defaultdict(list)
        for e in self.edges:
            self.out[e["head"]].append(e)
            self.inc[e["tail"]].append(e)
        # 表面形式（含别名）-> 规范名
        _, surface = load_entities()
        self.surface = {s: n for s, n in surface.items() if n in self.type}
        for n in self.type:
            self.surface.setdefault(n, n)
        self.matcher = Matcher(self.surface)

    def link(self, question: str) -> list[str]:
        """词典最长匹配，把问题里的术语链接到图谱实体。"""
        return list(dict.fromkeys(sp["name"] for sp in self.matcher.find(question)))

    def neighbors(self, name: str) -> list[dict]:
        return self.out[name] + self.inc[name]

    def path(self, a: str, b: str, max_hops: int = 3) -> list[dict]:
        """a、b 之间的最短路径（把边看作无向），返回路径上的边。"""
        prev = {a: None}
        dq = deque([(a, 0)])
        while dq:
            x, d = dq.popleft()
            if x == b:
                break
            if d == max_hops:
                continue
            for e in self.neighbors(x):
                y = e["tail"] if e["head"] == x else e["head"]
                if y not in prev:
                    prev[y] = (x, e)
                    dq.append((y, d + 1))
        if b not in prev:
            return []
        edges, x = [], b
        while prev[x]:
            x, e = prev[x]
            edges.append(e)
        return edges[::-1]


# ------------------------------------------------------------------ 检索器
class Retriever:
    HUB_DEGREE = 30

    def __init__(self, kg: KnowledgeGraph | None = None, embedder: Embedder | None = None):
        self.kg = kg or KnowledgeGraph()
        self.emb = embedder or Embedder()
        # 向量 RAG 的文档库：教材风格句子
        self.passages = load_lines("natural_sentences.txt") + load_lines("new_text.txt")
        self.passage_vecs = self.emb.encode(self.passages)
        self.names = list(self.kg.type)
        self.name_vecs = self.emb.encode(self.names)

    def vector_search(self, question: str, k: int = 5) -> list[str]:
        sims = self.passage_vecs @ self.emb.encode([question])[0]
        return [self.passages[i] for i in sims.topk(k).indices.tolist()]

    def graph_search(self, question: str, k: int = 12) -> Retrieval:
        kg = self.kg
        r = Retrieval(entities=kg.link(question))
        # "矩阵""向量"这类枢纽实体连着几十条边，单独展开只会带来噪声
        hubs = {a for a in r.entities if len(kg.neighbors(a)) > self.HUB_DEGREE}
        if not set(r.entities) - hubs:
            # 兜底：没链接到具体实体时，用问题和实体名的向量相似度找最接近的实体
            sims = self.name_vecs @ self.emb.encode([question])[0]
            top = sims.topk(2)
            extra = [self.names[i] for i, s in zip(top.indices.tolist(), top.values.tolist())
                     if s > 0.6 and self.names[i] not in r.entities]
            r.entities += extra
            r.fuzzy = bool(extra)
        if not r.entities:
            return r

        cand: dict[tuple, Fact] = {}

        def add(e, reason):
            f = Fact(e["head"], e["relation"], e["tail"], e["source"], reason)
            cand.setdefault(f.key, f)

        # 1) 链接到的实体两两之间的路径：一定保留
        must = set()
        non_hub = [a for a in r.entities if a not in hubs]
        for i, a in enumerate(non_hub):
            for b in non_hub[i + 1:]:
                for e in kg.path(a, b):
                    add(e, "路径")
                    must.add((e["head"], e["relation"], e["tail"]))
        # 2) 每个实体的一跳邻居（图谱里已物化了推理出的继承关系）
        for a in r.entities:
            for e in kg.neighbors(a):
                add(e, "邻居")

        # 3) 用向量相似度对候选事实重排，保留最相关的 k 条；只经过枢纽实体的事实最多留 3 条
        facts = list(cand.values())
        specific = set(r.entities) - hubs
        if facts:
            sims = self.emb.encode([f.text for f in facts]) @ self.emb.encode([question])[0]
            for f, s in zip(facts, sims.tolist()):
                f.score = round(s, 4)
        keep = [f for f in facts if f.key in must]
        rest = sorted((f for f in facts if f.key not in must), key=lambda f: -f.score)
        hub_quota = 3
        for f in rest:
            if len(keep) >= k:
                break
            if specific and f.head not in specific and f.tail not in specific:
                if hub_quota == 0:
                    continue
                hub_quota -= 1
            keep.append(f)
        r.facts = keep
        return r

    def retrieve(self, question: str, mode: str) -> Retrieval:
        r = self.graph_search(question) if mode in ("graph", "hybrid") else Retrieval()
        if mode in ("vector", "hybrid"):
            r.passages = self.vector_search(question)
        return r


# ------------------------------------------------------------------ 提示词
SYSTEM = "你是一名严谨的线性代数助教。"


def build_context(r: Retrieval) -> str:
    parts = []
    if r.facts:
        parts.append("【知识图谱中的相关事实】\n" + "\n".join(f"- {f.text}" for f in r.facts))
    if r.passages:
        parts.append("【教材中的相关句子】\n" + "\n".join(f"- {p}" for p in r.passages))
    if not parts:
        return ""
    return ("以下参考资料可能不完整：没有列出的关系不代表不成立，资料也可能与问题无关。"
            "请结合资料和你自己的知识作答。\n\n" + "\n\n".join(parts) + "\n\n")


def build_prompt(question: str, r: Retrieval, instruction: str) -> str:
    return f"{build_context(r)}问题：{question}\n\n{instruction}"


EXPLAIN = "请先给出结论，再用两三句话说明理由。如果用到了参考资料中的事实，请指出是哪一条。"


def answer(llm, retriever: Retriever, question: str, mode: str = "graph", instruction: str = EXPLAIN,
           max_tokens: int = 300):
    r = retriever.retrieve(question, mode)
    prompt = build_prompt(question, r, instruction)
    return llm.chat(SYSTEM, prompt, max_tokens=max_tokens), r, prompt
