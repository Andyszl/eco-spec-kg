#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=$(date +%Y%m%d-%H%M%S)-$$
TRAIN=/home/hello/szl/eco-spec-runs/lora/full_graph_scope_seed42_pair_20261019-192829-3887179
OUT="/home/hello/szl/eco-spec-results/full_graph_scope_learning_probe_${TAG}"
TRANSFER="${OUT}_git"
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
BRANCH="codex/full-graph-scope-learning-probe-${TAG}"
export PYTHONPATH="$TASK_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
test ! -e "$OUT"
mkdir -p "$OUT"
STATUS=0
python "$TASK_ROOT/tools/probe_full_graph_scope_20261019.py" --train "$TRAIN" --out "$OUT/job" 2>&1 | tee "$OUT/launch.log" || STATUS=$?
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
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer fixed full-graph scoped train and internal-validation diagnosis'
LOCAL_SHA=$(git rev-parse HEAD)
VERIFIED=0
for ATTEMPT in 1 2 3; do
  if git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$BRANCH"; then
    REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1) || REMOTE_SHA=
    if [ "$REMOTE_SHA" = "$LOCAL_SHA" ]; then VERIFIED=1; break; fi
  fi
  if [ "$ATTEMPT" -lt 3 ]; then sleep 5; fi
done
test "$VERIFIED" = 1 || { printf 'UPLOAD_RETRY_FAILED; results preserved in %s\n' "$TRANSFER" >&2; exit 1; }
printf 'RESULTS_UPLOAD_OK=%s\nPROBE_EXIT_CODE=%s\nPROBE_DIRECTORY=%s\n' "$BRANCH" "$STATUS" "$OUT"
exit "$STATUS"
