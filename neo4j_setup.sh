#!/usr/bin/env bash
# 在本机启动 Neo4j 社区版并导入线性代数知识图谱。
#
# 依赖：
#   * JDK 21：brew install openjdk@21
#   * Neo4j 社区版 + cypher-shell：首次运行时自动从官方地址下载到 .tools/，并校验 SHA256
# 本地数据库密码首次运行时随机生成，保存在 .neo4j.env（仅本人可读）。
#
# 用法：./neo4j_setup.sh          启动并导入
#       ./neo4j_setup.sh stop     停止服务
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

VER=2026.10.2
NEO4J_SHA=fcbccee832175fd8b873ca7d8228dbde4cb5cce43f9053833cc7517dc4dd6f31
CS_SHA=b60df0d21b3b236e4a4f359277477f5595b3d8a6253eaa2226f724641ff2677c

export JAVA_HOME="${JAVA_HOME:-$(brew --prefix openjdk@21 2>/dev/null)}"
[ -x "$JAVA_HOME/bin/java" ] || { echo "未找到 JDK 21，请先运行：brew install openjdk@21"; exit 1; }

TOOLS="$ROOT/.tools"
export NEO4J_HOME="$TOOLS/neo4j-community-$VER"
NEO4J="$NEO4J_HOME/bin/neo4j"
ADMIN="$NEO4J_HOME/bin/neo4j-admin"
CYPHER_SHELL="$TOOLS/cypher-shell-$VER/bin/cypher-shell"

# 0) 下载（如果还没有）
if [ ! -x "$NEO4J" ] || [ ! -x "$CYPHER_SHELL" ]; then
  mkdir -p "$TOOLS"; cd "$TOOLS"
  curl -fsSL -o neo4j.tar.gz "https://neo4j.com/artifact.php?name=neo4j-community-$VER-unix.tar.gz"
  curl -fsSL -o cypher-shell.zip "https://dist.neo4j.org/cypher-shell/cypher-shell-$VER.zip"
  printf '%s  neo4j.tar.gz\n%s  cypher-shell.zip\n' "$NEO4J_SHA" "$CS_SHA" | shasum -a 256 -c
  tar xzf neo4j.tar.gz && unzip -q -o cypher-shell.zip
  cd "$ROOT"
fi

if [ "${1:-}" = "stop" ]; then
  "$NEO4J" stop
  exit 0
fi

CYPHER=outputs/kg/neo4j_import.cypher
[ -f "$CYPHER" ] || { echo "未找到 $CYPHER，请先运行 ./run_all.sh"; exit 1; }

# 1) 首次运行：生成本地密码（必须在第一次启动 Neo4j 之前设置）
if [ ! -f .neo4j.env ]; then
  PW="la-kg-$(openssl rand -hex 6)"
  "$ADMIN" dbms set-initial-password "$PW" >/dev/null
  ( umask 077; printf 'NEO4J_USER=neo4j\nNEO4J_PASSWORD=%s\n' "$PW" > .neo4j.env )
  echo "已生成本地 Neo4j 密码，保存在 .neo4j.env"
fi
set -a; source .neo4j.env; set +a

# 2) 启动（如果还没运行）
if ! "$NEO4J" status >/dev/null 2>&1; then
  "$NEO4J" start
fi
CS=("$CYPHER_SHELL" -a neo4j://localhost:7687 -u "$NEO4J_USER" -p "$NEO4J_PASSWORD")
echo -n "等待 Neo4j 就绪"
for i in $(seq 1 60); do
  if "${CS[@]}" "RETURN 1" >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."; sleep 2
  [ "$i" = 60 ] && { echo " 超时，请查看日志：$NEO4J_HOME/logs/"; exit 1; }
done

# 3) 清掉旧的图谱节点（只删本图谱的节点，不影响库里其他数据），再导入
"${CS[@]}" "MATCH (n) WHERE n.kg = 'linear_algebra' OR n:Entity DETACH DELETE n" >/dev/null
"${CS[@]}" "DROP CONSTRAINT la_entity_name IF EXISTS" >/dev/null
"${CS[@]}" -f "$CYPHER" >/dev/null
echo "导入完成。"

# 4) 校验
"${CS[@]}" --format plain "MATCH (n {kg:'linear_algebra'}) RETURN labels(n)[0] AS 类型, count(*) AS 实体数 ORDER BY 实体数 DESC"
"${CS[@]}" --format plain "MATCH ({kg:'linear_algebra'})-[r]->() RETURN r.source AS 来源, count(*) AS 三元组数 ORDER BY 三元组数 DESC"
echo
echo "Neo4j Browser: http://localhost:7474  （用户名 neo4j，密码见 .neo4j.env）"
echo "查看全图：MATCH (n {kg:'linear_algebra'}) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m"
echo "停止服务：./neo4j_setup.sh stop"
