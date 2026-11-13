#!/usr/bin/env bash
# 一键复现：构建语料 -> 训练两种编码器 -> 抽取 + 推理 + 导出
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -r requirements.txt
fi
$PY kg/build_corpus.py
$PY kg/train.py --encoder char
$PY kg/train.py --encoder bert
$PY kg/build_kg.py
