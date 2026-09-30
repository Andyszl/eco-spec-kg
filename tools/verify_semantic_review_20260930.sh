#!/usr/bin/env bash
set -euo pipefail
trap 'printf "\n操作停止：第 %s 行，请检查上方报错。\n" "$LINENO" >&2' ERR
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
BASE=/home/hello/szl/formula_review_20260929/data_candidate
OUT=/home/hello/szl/eco-spec-results/semantic_review_20260930
test -f "$BASE/manifest.json"
if [ -e "$OUT" ]; then
  printf '报告目录已存在，未覆盖：%s\n' "$OUT" >&2
  exit 1
fi
cd "$CODE"
git log -1 --oneline
python -m pytest tests -o addopts='' -p no:cacheprovider -q -rs
python tools/build_semantic_review.py --base "$BASE" --out "$OUT/data_candidate"
python tools/audit_semantic_review.py --base "$BASE" --reviewed "$OUT/data_candidate" --out "$OUT/audit"
python - "$OUT/audit/comparison.json" <<'PY'
import json
import sys
from pathlib import Path
actual=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
expected=json.loads(Path('deliveries/semantic_review_20260930/expected_comparison.json').read_text(encoding='utf-8'))
# Physical source-file line endings can differ between Windows and Linux.
# Data bytes, relation deltas and every matrix cell must agree exactly.
for report in (actual,expected):
    report.pop('source_hashes')
if actual!=expected:
    raise SystemExit('服务器与本地结果不一致；保留报告，不进入模型实验。')
print('语义复核服务器复验通过：rc2文件哈希、八组覆盖和逐单元关系差异一致。')
PY
printf '\n报告：%s/audit/comparison.json\n训练候选：%s/data_candidate\n本次未训练或推理。\n' "$OUT" "$OUT"
