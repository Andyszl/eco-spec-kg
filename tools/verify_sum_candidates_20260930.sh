#!/usr/bin/env bash
# Run from the new Git worktree; previous reports and reviewed data stay intact.
set -euo pipefail
trap 'printf "\n操作停止：第 %s 行，请检查上方报错。\n" "$LINENO" >&2' ERR
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hello/szl/eco-spec-results/paper_audit_20260929/session.env.sh
export PYTHONPATH="$CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
REVIEWED=/home/hello/szl/formula_review_20260929/data_candidate
OUT=/home/hello/szl/eco-spec-results/sum_candidates_20260930
test -f "$REVIEWED/manifest.json"
if [ -e "$OUT" ]; then
  printf '报告目录已存在，未覆盖：%s\n' "$OUT" >&2
  exit 1
fi
cd "$CODE"
git log -1 --oneline
python -m pytest tests -o addopts='' -p no:cacheprovider -q -rs
python tools/audit_sum_candidates.py --base "$DATA_PACKAGE" --reviewed "$REVIEWED" --out "$OUT"
python - "$OUT/comparison.json" <<'PY'
import json
import sys
from pathlib import Path
actual=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
expected=json.loads(Path('docs/audits/sum_candidates_20260930.json').read_text(encoding='utf-8'))
if actual['conditions'] != expected['conditions']:
    raise SystemExit('服务器与本地覆盖或逐单元关系差异不一致，请保留报告排查。')
print('\n求和修复服务器复验通过：覆盖统计与逐单元增删均与本地一致。')
PY
printf '\n报告：%s/comparison.json\n本次未进行模型训练或推理。\n' "$OUT"
