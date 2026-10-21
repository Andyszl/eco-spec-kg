#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=$(date +%Y%m%d-%H%M%S)-$$
OUT="${ECOSPEC_FINAL_EVAL_RESULTS_ROOT:-/home/hello/szl/eco-spec-results}/final_source_evaluation_${TAG}"
TRANSFER="${OUT}_git"
REMOTE="${ECOSPEC_FINAL_EVAL_REMOTE:-https://github.com/Andyszl/eco-spec-kg.git}"
BRANCH="codex/final-source-evaluation-${TAG}"
PYTHON="${ECOSPEC_FINAL_EVAL_PYTHON:-python}"
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT/tools" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 TOKENIZERS_PARALLELISM=false
export WORLD_SIZE=1 NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
test ! -e "$OUT"
mkdir -p "$OUT"
printf 'EVALUATION_DIRECTORY=%s\n' "$OUT"
STATUS=0
"$PYTHON" "$TASK_ROOT/tools/final_source_evaluation_20261021.py" --out "$OUT/job" 2>&1 | tee "$OUT/launch.log" || STATUS=$?
test ! -e "$TRANSFER"
mkdir "$TRANSFER"
tar -czf "$TRANSFER/results.tar.gz" -C "$OUT" .
cd "$TRANSFER"
sha256sum results.tar.gz > reports.sha256
printf '%s\n' "$OUT" > result_directory.txt
printf '%s\n' "$STATUS" > evaluation_exit_code.txt
printf '* -text -filter\n' > .gitattributes
git -c init.defaultBranch=result-transfer init -q
git add -- .gitattributes results.tar.gz reports.sha256 result_directory.txt evaluation_exit_code.txt
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer six frozen final-source requests and partial-positive coverage; no training'
LOCAL_SHA=$(git rev-parse HEAD)
VERIFIED=0
for ATTEMPT in 1 2 3; do
  if git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$BRANCH"; then
    REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1) || REMOTE_SHA=
    if [ "$REMOTE_SHA" = "$LOCAL_SHA" ]; then VERIFIED=1; break; fi
  fi
  if [ "$ATTEMPT" -lt 3 ]; then sleep 5; fi
done
test "$VERIFIED" = 1 || { printf 'UPLOAD_RETRY_FAILED; do not repeat inference; results preserved in %s\n' "$TRANSFER" >&2; exit 1; }
printf 'RESULTS_UPLOAD_OK=%s\nEVALUATION_EXIT_CODE=%s\nEVALUATION_DIRECTORY=%s\n' "$BRANCH" "$STATUS" "$OUT"
exit "$STATUS"
