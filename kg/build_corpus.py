"""由种子三元组 + 教材句子构建 NER / 关系抽取语料（远程监督自动标注），并划分训练/验证/测试集。

划分策略（为了让测试有意义，而不是背答案）：
  * 留出实体：HELD_ENTITIES 中的实体在训练集中完全不出现，测试 NER 能否识别没见过的实体；
  * 留出三元组：与留出实体相关的三元组 + 随机 12% 的其他三元组只出现在测试集，测试能否抽出新三元组；
  * 留出句式：每种关系的最后一个模板只用于测试，测试对没见过的表述方式的泛化；
  * 教材自然句：一部分进训练，其余进验证/测试。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from common import (CORPUS, Example, Matcher, RELATIONS, NA, load_entities,
                    load_lines, load_triples)

SEED = 42

# 每种关系若干模板，最后一个只用于测试（"没见过的句式"）
TEMPLATES = {
    "is_a": ["{h}是一种特殊的{t}。", "{h}属于{t}的范畴。", "每个{h}都是{t}。",
             "作为{t}的一类，{h}在线性代数中十分常见。", "我们可以把{h}看作{t}的特例。",
             "{t}包括{h}等多种类型。", "从分类上看，{h}归入{t}。"],
    "part_of": ["{h}是{t}的组成部分。", "{t}中包含{h}。", "{h}构成了{t}的一部分。",
                "{t}由{h}等部分组成。", "{h}是构成{t}的要素之一。"],
    "depends_on": ["{h}的定义依赖于{t}。", "要理解{h}，首先需要掌握{t}。", "{h}是在{t}的基础上定义的。",
                   "学习{h}之前应先学习{t}。", "{h}这一概念建立在{t}之上。", "没有{t}就无法定义{h}。"],
    "has_property": ["{h}具有{t}。", "{h}满足{t}。", "可以证明{h}具有{t}这一性质。",
                     "{t}是{h}的重要性质。", "{h}的一个基本性质是{t}。", "{h}总是满足{t}的要求。"],
    "equivalent_to": ["{h}等价于{t}。", "{h}与{t}是等价的。", "对方阵而言，{h}与{t}互为充要条件。",
                      "{h}和{t}可以相互推出。"],
    "solved_by": ["{h}可以通过{t}求得。", "利用{t}可以求解{h}。", "{t}是计算{h}的常用方法。",
                  "计算{h}时常采用{t}。", "借助{t}能够得到{h}。", "实际中通常用{t}来确定{h}。"],
    "acts_on": ["{h}的作用对象是{t}。", "{h}适用于{t}。", "可以对{t}做{h}。",
                "{h}可以作用于{t}。", "{h}主要针对{t}进行处理。"],
    "yields": ["{h}的结果是{t}。", "通过{h}可以得到{t}。", "{h}最终得到{t}。",
               "经过{h}后得到{t}。", "{h}的输出是一个{t}。"],
    "involves": ["{h}涉及{t}。", "{h}刻画了{t}的性质。", "{h}描述了与{t}有关的结论。",
                 "在{h}中会用到{t}。", "{t}是{h}中的关键概念。", "{h}的表述离不开{t}。"],
}
NEG_TEMPLATES = ["{h}和{t}是两个不同的知识点。", "本节先介绍{h}，随后讨论{t}。",
                 "{h}与{t}都是线性代数中的重要内容。", "{h}和{t}之间没有直接的从属关系。",
                 "考试中{h}和{t}经常一起出现。"]
PREFIXES = ["", "", "", "在线性代数中，", "教材中提到，", "需要注意的是，", "一般地，"]

HELD_ENTITIES = {"反对称矩阵", "下三角矩阵", "酉空间", "楚列斯基分解", "舒尔定理", "几何重数",
                 "顺序主子式", "维数公式", "向量积", "规范形", "幂零矩阵", "正交补"}


def make_example(text, spans, etype, rel_lookup, source, held=False, force=None):
    """spans: 匹配到的实体；为每个有序实体对打关系标签（种子三元组中存在则为该关系，否则 NA）。"""
    ents = [dict(sp, type=etype[sp["name"]]) for sp in spans]
    rels = []
    for i, a in enumerate(ents):
        for j, b in enumerate(ents):
            if i == j or a["name"] == b["name"]:
                continue
            lab = rel_lookup.get((a["name"], b["name"]), NA)
            if force and (a["name"], b["name"]) == force[:2]:
                lab = force[2]
            rels.append({"head": i, "tail": j, "label": lab})
    return Example(text, ents, rels, source, held)


def surface_of(name, aliases, rng):
    if aliases.get(name) and rng.random() < 0.3:
        return rng.choice(aliases[name])
    return name


def main():
    rng = random.Random(SEED)
    etype, surface = load_entities()
    triples = load_triples(etype)
    matcher = Matcher(surface)
    aliases = defaultdict(list)
    for s, n in surface.items():
        if s != n:
            aliases[n].append(s)

    rel_lookup = {}
    for h, r, t in triples:
        assert (h, t) not in rel_lookup, f"同一实体对有多个关系: {h}, {t}"
        rel_lookup[(h, t)] = r

    others = [tr for tr in triples if tr[0] not in HELD_ENTITIES and tr[2] not in HELD_ENTITIES]
    rng.shuffle(others)
    held_triples = {tr for tr in triples if tr[0] in HELD_ENTITIES or tr[2] in HELD_ENTITIES}
    held_triples |= set(others[: int(0.12 * len(others))])
    held_pairs = {(h, t) for h, _, t in held_triples}

    split = {"train": [], "dev": [], "test": []}

    def render(tpl, h, t):
        hs, ts = surface_of(h, aliases, rng), surface_of(t, aliases, rng)
        text = rng.choice(PREFIXES) + tpl.format(h=hs, t=ts)
        spans = matcher.find(text)
        names = [sp["name"] for sp in spans]
        assert h in names and t in names, (text, names)
        return text, spans

    # 1) 模板句
    for h, r, t in triples:
        tpls = TEMPLATES[r]
        is_held = (h, r, t) in held_triples
        for k, tpl in enumerate(tpls):
            text, spans = render(tpl, h, t)
            ex = make_example(text, spans, etype, rel_lookup, "template", held=is_held)
            if is_held or k == len(tpls) - 1:
                split["test"].append(ex)
            else:
                split["train" if rng.random() < 0.9 else "dev"].append(ex)

    # 2) 负例句（无关系实体对）
    names = list(etype)
    n_neg = 0
    while n_neg < 400:
        h, t = rng.sample(names, 2)
        if (h, t) in rel_lookup or (t, h) in rel_lookup:
            continue
        text, spans = render(rng.choice(NEG_TEMPLATES), h, t)
        ex = make_example(text, spans, etype, rel_lookup, "negative")
        if h in HELD_ENTITIES or t in HELD_ENTITIES:
            split["test"].append(ex)
        else:
            u = rng.random()
            split["train" if u < 0.8 else "dev" if u < 0.9 else "test"].append(ex)
        n_neg += 1

    # 3) 教材自然句
    for text in load_lines("natural_sentences.txt"):
        spans = matcher.find(text)
        ex = make_example(text, spans, etype, rel_lookup, "natural")
        ns = {sp["name"] for sp in spans}
        touches_held = bool(ns & HELD_ENTITIES) or any(
            (a["name"], b["name"]) in held_pairs for a in ex.entities for b in ex.entities)
        if touches_held:
            ex.held = True
            split["test"].append(ex)
        else:
            u = rng.random()
            split["train" if u < 0.5 else "dev" if u < 0.65 else "test"].append(ex)

    # 校验：留出实体绝不出现在训练集
    for ex in split["train"] + split["dev"]:
        assert not ({e["name"] for e in ex.entities} & HELD_ENTITIES), ex.text

    CORPUS.mkdir(parents=True, exist_ok=True)
    for k, exs in split.items():
        rng.shuffle(exs)
        (CORPUS / f"{k}.jsonl").write_text("\n".join(e.to_json() for e in exs) + "\n", encoding="utf-8")
    (CORPUS / "held_triples.tsv").write_text(
        "\n".join("\t".join(tr) for tr in sorted(held_triples)) + "\n", encoding="utf-8")

    print(f"实体 {len(etype)} 个，种子三元组 {len(triples)} 条，留出三元组 {len(held_triples)} 条，留出实体 {len(HELD_ENTITIES)} 个")
    print("实体类型分布:", dict(Counter(etype.values())))
    print("关系分布:", dict(Counter(r for _, r, _ in triples)))
    for k, exs in split.items():
        rc = Counter(r["label"] for e in exs for r in e.relations)
        print(f"{k:5s}: {len(exs)} 句, 实体 {sum(len(e.entities) for e in exs)}, 关系对 {sum(rc.values())} (非NA {sum(v for l, v in rc.items() if l != NA)}),"
              f" 来源 {dict(Counter(e.source for e in exs))}")


if __name__ == "__main__":
    main()
