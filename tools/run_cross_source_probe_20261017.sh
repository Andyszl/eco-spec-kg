#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=$(date +%Y%m%d-%H%M%S)-$$
TRAIN=/home/hello/szl/eco-spec-runs/lora/cross_source_seed42_pair_20261016-151235-2370771
OUT="/home/hello/szl/eco-spec-results/cross_source_learning_probe_${TAG}"
TRANSFER="${OUT}_git"
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
BRANCH="codex/cross-source-learning-probe-${TAG}"
export PYTHONPATH="$TASK_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
test ! -e "$OUT"
mkdir -p "$OUT"
STATUS=0
python "$TASK_ROOT/tools/probe_cross_source_learning_20261017.py" --train "$TRAIN" --out "$OUT/job" 2>&1 | tee "$OUT/launch.log" || STATUS=$?
if [ ! -d "$OUT" ]; then
  printf 'PROBE_BLOCKED_BEFORE_LAUNCH\n' >&2
  if [ "$STATUS" -eq 0 ]; then STATUS=1; fi
  exit "$STATUS"
fi
test ! -e "$TRANSFER"
mkdir "$TRANSFER"
tar -czf "$TRANSFER/results.tar.gz" -C "$OUT" .
cd "$TRANSFER"
sha256sum results.tar.gz > reports.sha256
printf '%s\n' "$OUT" > result_directory.txt
printf '%s\n' "$STATUS" > probe_exit_code.txt
printf '* -text -filter\n' > .gitattributes
git -c init.defaultBranch=result-transfer init -q
git add -- .gitattributes results.tar.gz reports.sha256 result_directory.txt probe_exit_code.txt
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer fixed cross-source training-fit and validation diagnostic'
git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$BRANCH"
REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1)
test "$REMOTE_SHA" = "$(git rev-parse HEAD)"
printf 'RESULTS_UPLOAD_OK=%s\nPROBE_EXIT_CODE=%s\nPROBE_DIRECTORY=%s\n' "$BRANCH" "$STATUS" "$OUT"
exit "$STATUS"
