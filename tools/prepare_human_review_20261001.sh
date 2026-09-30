#!/usr/bin/env bash
set -euo pipefail
umask 077
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/hello/szl/eco-spec-results/paper_audit_20260929/session.env.sh
export PYTHONPATH="$CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
RC2=/home/hello/szl/eco-spec-results/semantic_review_20260930/data_candidate
OUT=/home/hello/szl/eco-spec-results/human_review_20261001/packet
test -f "$RC2/manifest.json"
test -f "$DATA_PACKAGE/manifest.json"
if [ -e "$OUT" ]; then
  printf '审核包已存在，未覆盖：%s\n' "$OUT" >&2
  exit 1
fi
cd "$CODE"
python -m pytest tests/test_human_review_gate.py -o addopts='' -p no:cacheprovider -q
python tools/prepare_human_review_v2.py prepare \
  --rc2 "$RC2" --frozen "$DATA_PACKAGE" --out "$OUT"
printf '\n待专家审核包：%s\n尚未冻结数据或运行模型。\n' "$OUT"
