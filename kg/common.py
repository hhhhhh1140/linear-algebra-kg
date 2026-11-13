"""公共定义：本体（实体类型、关系类型）、数据读取、词典最长匹配标注。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
OUT = ROOT / "outputs"

ENTITY_TYPES = {
    "CON": "概念",
    "THM": "定理",
    "PROP": "性质",
    "OP": "运算",
    "METH": "方法",
}

# 关系：键 -> (中文名, 允许的头实体类型, 允许的尾实体类型)
RELATIONS = {
    "is_a": ("上位概念", {"CON", "OP", "METH"}, {"CON", "OP", "METH"}),
    "part_of": ("组成部分", {"CON"}, {"CON", "METH"}),
    "depends_on": ("定义依赖", {"CON", "PROP", "METH"}, {"CON", "PROP", "OP"}),
    "has_property": ("具有性质", {"CON", "OP"}, {"PROP"}),
    "equivalent_to": ("等价", {"PROP", "CON"}, {"PROP", "CON"}),
    "solved_by": ("求解方法", {"CON", "PROP", "METH"}, {"METH", "THM", "OP", "CON"}),
    "acts_on": ("作用对象", {"OP", "METH"}, {"CON"}),
    "yields": ("得到结果", {"OP", "METH"}, {"CON"}),
    "involves": ("定理涉及", {"THM"}, {"CON", "OP", "PROP"}),
}
NA = "NA"
REL_LABELS = [NA] + list(RELATIONS)

NER_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]


def _rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        yield line.split("\t")


def load_entities() -> tuple[dict[str, str], dict[str, str]]:
    """返回 (规范名 -> 类型, 表面形式 -> 规范名)。"""
    etype, surface = {}, {}
    for row in _rows(DATA / "entities.tsv"):
        name, t = row[0].strip(), row[1].strip()
        assert t in ENTITY_TYPES, (name, t)
        etype[name] = t
        surface[name] = name
        if len(row) > 2:
            for a in row[2].split(","):
                if a.strip():
                    surface[a.strip()] = name
    return etype, surface


def load_triples(etype: dict[str, str]) -> list[tuple[str, str, str]]:
    triples = []
    for row in _rows(DATA / "triples.tsv"):
        h, r, t = (x.strip() for x in row[:3])
        assert h in etype, f"未知实体: {h}"
        assert t in etype, f"未知实体: {t}"
        assert r in RELATIONS, f"未知关系: {r}"
        triples.append((h, r, t))
    assert len(set(triples)) == len(triples), "存在重复三元组"
    return triples


def load_lines(name: str) -> list[str]:
    return [l.strip() for l in (DATA / name).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


class Matcher:
    """基于词典的最长匹配，用于远程监督自动标注。"""

    def __init__(self, surface: dict[str, str]):
        self.surface = surface
        self.maxlen = max(map(len, surface))

    def find(self, text: str) -> list[dict]:
        """全局最长优先：先收集所有匹配，按长度从长到短选择互不重叠的片段。"""
        cands = []
        for i in range(len(text)):
            for L in range(1, min(self.maxlen, len(text) - i) + 1):
                s = text[i:i + L]
                if s in self.surface:
                    cands.append((i, i + L))
        cands.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        used, spans = [False] * len(text), []
        for a, b in cands:
            if not any(used[a:b]):
                for k in range(a, b):
                    used[k] = True
                spans.append({"start": a, "end": b, "text": text[a:b], "name": self.surface[text[a:b]]})
        return sorted(spans, key=lambda x: x["start"])


def bio_tags(n: int, spans: list[dict], etype: dict[str, str]) -> list[str]:
    tags = ["O"] * n
    for sp in spans:
        t = sp.get("type") or etype[sp["name"]]
        tags[sp["start"]] = f"B-{t}"
        for k in range(sp["start"] + 1, sp["end"]):
            tags[k] = f"I-{t}"
    return tags


def tags_to_spans(text: str, tags: list[str]) -> list[dict]:
    spans, cur = [], None
    for i, tg in enumerate(tags):
        if tg.startswith("B-") or (tg.startswith("I-") and (cur is None or cur["type"] != tg[2:])):
            if cur:
                spans.append(cur)
            cur = {"start": i, "end": i + 1, "type": tg[2:]}
        elif tg.startswith("I-") and cur is not None:
            cur["end"] = i + 1
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    for sp in spans:
        sp["text"] = text[sp["start"]:sp["end"]]
    return spans


@dataclass
class Example:
    text: str
    entities: list[dict]          # {start,end,text,name,type}
    relations: list[dict]         # {head: idx, tail: idx, label}
    source: str                   # template / natural / negative
    held: bool = False            # 是否只出现在测试集

    def to_json(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
