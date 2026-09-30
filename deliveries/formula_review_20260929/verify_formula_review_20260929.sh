#!/usr/bin/env bash
# Verify the delivered training candidate. Does not train or call model services.
set -euo pipefail
trap 'printf "\n操作停止：第 %s 行，请检查上方报错。\n" "$LINENO" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$SCRIPT_DIR/formula_review_20260929.zip"
ROOT=/home/hello/szl
CODE="$ROOT/eco-spec-kg-symbols-20260929"
REV=9bc751ff7281faf41e75193975159d429db76e71
PACKAGE="$ROOT/formula_review_20260929"
REPORT="$ROOT/eco-spec-results/source_review_20260929_matrix"

printf '\n[1/5] 检查运行环境和原训练数据\n'
python -c 'import sys, pytest; print("Python:", sys.executable)'
source "$ROOT/eco-spec-results/paper_audit_20260929/session.env.sh"
test -f "${DATA_PACKAGE:?session.env.sh未提供DATA_PACKAGE}/blind/train_units.jsonl"
test -f "$DATA_PACKAGE/gold/train_annotations.jsonl"

printf '\n[2/5] 准备固定代码版本\n'
if [ ! -e "$CODE" ]; then
  git -C "$ROOT/eco-spec-kg" fetch https://github.com/Andyszl/eco-spec-kg.git main
  git -C "$ROOT/eco-spec-kg" cat-file -e "${REV}^{commit}"
  git -C "$ROOT/eco-spec-kg" worktree add --detach "$CODE" "$REV"
fi
test "$(git -C "$CODE" rev-parse HEAD)" = "$REV"
git -C "$CODE" diff --quiet HEAD -- src
export PYTHONPATH="$CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
git -C "$CODE" log -1 --oneline

printf '\n[3/5] 校验压缩包及交付文件\n'
cd "$ROOT"
printf '%s  %s\n' \
  e4dcdd48b55438dad74374cf222c77682fc7f79185810e45edcdaedac96cd1e4 \
  "$ARCHIVE" | sha256sum -c -
if [ ! -e "$PACKAGE" ]; then
  python -m zipfile -e "$ARCHIVE" "$ROOT"
fi
cd "$PACKAGE"
python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path.cwd()
manifest = json.loads((root / 'delivery_manifest.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    actual = hashlib.sha256((root / item['path']).read_bytes()).hexdigest()
    if actual != item['sha256']:
        raise SystemExit('交付文件哈希不符: ' + item['path'])
print('交付文件哈希校验通过')
PY

printf '\n[4/5] 验证修订器与训练候选包\n'
python -m pytest test_build_revision.py -o addopts='' -p no:cacheprovider -q
python validate_candidate.py --base "$DATA_PACKAGE" --candidate data_candidate

printf '\n[5/5] 复算四组候选覆盖\n'
if [ -e "$REPORT" ]; then
  printf '报告目录已存在，未覆盖：%s\n请核查已有结果，不要直接删除重跑。\n' "$REPORT" >&2
  exit 1
fi
python evaluate_candidate.py --base "$DATA_PACKAGE" --candidate data_candidate --out "$REPORT"
python - "$REPORT/coverage_matrix.json" <<'PY'
import json
import sys
from pathlib import Path
expected = json.loads(Path('analysis/coverage_matrix.json').read_text(encoding='utf-8'))
actual = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if actual != expected:
    raise SystemExit('服务器与本地四格统计不一致，请保留报告排查。')
print('\n服务器与本地四格统计完全一致，复验通过。')
PY
printf '\n报告位置：%s\n本次未运行模型训练或推理。\n' "$REPORT"
