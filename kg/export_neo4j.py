"""把 outputs/kg/kg.json 导出为 Neo4j 导入脚本和交互式可视化页面（不依赖 PyTorch，可单独运行）。

为了在 Neo4j Browser 里直观：
  * 节点标签直接用中文类型（概念 / 定理 / 性质 / 运算 / 方法），每种类型自动显示不同颜色；
  * 关系类型直接用中文（上位概念、具有性质……），英文关系名保存在属性 relation 里；
  * 所有节点带属性 kg='linear_algebra'，便于整体清除和重新导入。

用法：python3 kg/export_neo4j.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ENTITY_TYPES, OUT, RELATIONS  # noqa: E402

KG_TAG = "linear_algebra"


def q(s):
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def bt(s):
    """中文标签 / 关系类型要用反引号括起来。"""
    return "`" + s.replace("`", "") + "`"


def export_cypher(path: Path, nodes: list[dict], edges: list[dict]):
    ntype = {n["name"]: n["type"] for n in nodes}
    lines = [
        "// 线性代数知识图谱（节点标签与关系类型均为中文）",
        "// 用法：cypher-shell -f neo4j_import.cypher，或粘贴到 Neo4j Browser 运行",
    ]
    for t, zh in ENTITY_TYPES.items():
        lines.append(f"CREATE CONSTRAINT la_{t.lower()}_name IF NOT EXISTS FOR (n:{bt(zh)}) REQUIRE n.name IS UNIQUE;")
    for n in nodes:
        zh = ENTITY_TYPES[n["type"]]
        lines.append(f"MERGE (n:{bt(zh)} {{name:{q(n['name'])}}}) "
                     f"SET n.kg={q(KG_TAG)}, n.type={q(n['type'])}, n.source={q(n['source'])};")
    for e in edges:
        rel_zh = RELATIONS[e["relation"]][0]
        props = f"r.relation={q(e['relation'])}, r.source={q(e['source'])}"
        if e.get("score") is not None:
            props += f", r.score={e['score']}"
        if e.get("rule"):
            props += f", r.rule={q(e['rule'])}"
        if e.get("via"):
            props += f", r.via={q(e['via'])}"
        lines.append(
            f"MATCH (a:{bt(ENTITY_TYPES[ntype[e['head']]])} {{name:{q(e['head'])}}}), "
            f"(b:{bt(ENTITY_TYPES[ntype[e['tail']]])} {{name:{q(e['tail'])}}}) "
            f"MERGE (a)-[r:{bt(rel_zh)}]->(b) SET {props};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_html(path: Path, nodes: list[dict], edges: list[dict]):
    tpl = (DATA.parent / "kg" / "viewer_template.html").read_text(encoding="utf-8")
    data = {"nodes": nodes, "edges": edges,
            "types": ENTITY_TYPES, "relations": {k: v[0] for k, v in RELATIONS.items()}}
    path.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


def main():
    kg = json.loads((OUT / "kg" / "kg.json").read_text(encoding="utf-8"))
    path = OUT / "kg" / "neo4j_import.cypher"
    export_cypher(path, kg["nodes"], kg["edges"])
    export_html(OUT / "kg" / "kg_view.html", kg["nodes"], kg["edges"])
    print(f"已导出 {len(kg['nodes'])} 个节点、{len(kg['edges'])} 条关系 -> {path} 和 kg_view.html")


if __name__ == "__main__":
    main()
