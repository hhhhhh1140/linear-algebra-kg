#!/usr/bin/env bash
# 用无头 Chrome 为 README 生成图谱截图（需要联网加载 vis-network）。
# 用法：./docs/make_screenshots.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PAGE="file://$PWD/outputs/kg/kg_view.html"
OUTDIR=docs/images
mkdir -p "$OUTDIR"

shot() {  # 文件名 宽 高 查询参数
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1.5 \
    --window-size="$2,$3" --virtual-time-budget=25000 \
    --screenshot="$OUTDIR/$1" "$PAGE?$4" 2>/dev/null
  echo "生成 $OUTDIR/$1"
}

shot full_graph.png        1500 1000 "embed=1"
shot viewer_ui.png         1500  900 ""
shot focus_positive_definite.png 1200 800 "embed=1&focus=正定矩阵"
shot focus_eigenvalue.png        1200 800 "embed=1&focus=特征值"
shot focus_linear_system.png     1200 800 "embed=1&focus=线性方程组"
shot focus_gram_matrix.png       1200 800 "embed=1&focus=格拉姆矩阵"
