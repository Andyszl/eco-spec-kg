#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test "$#" -eq 1 || { printf 'Usage: bash run_semantic_pair_20261015.sh /absolute/preflight_report.json\n' >&2; exit 2; }
PREFLIGHT=$1
test -s "$PREFLIGHT"
TAG=$(date +%Y%m%d-%H%M%S)-$$
PAIR="/home/hello/szl/eco-spec-runs/lora/semantic_seed42_pair_${TAG}"
OUT="/home/hello/szl/eco-spec-results/semantic_seed42_pair_${TAG}"
TRANSFER="${OUT}_git"
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
BRANCH="codex/semantic-seed42-pair-${TAG}"
export PYTHONPATH="$TASK_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
STATUS=0
python "$TASK_ROOT/tools/semantic_pair_20261015.py" train \
  --model /home/hello/szl/models/Qwen3.5-9B --preflight "$PREFLIGHT" --out "$PAIR" --run || STATUS=$?
if [ ! -s "$PAIR/pair_manifest.json" ]; then
  printf 'TRAINING_BLOCKED_BEFORE_LAUNCH\n' >&2
  if [ "$STATUS" -eq 0 ]; then STATUS=1; fi
  exit "$STATUS"
fi
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$PAIR/pair_manifest.json" "$OUT/"
for ARM in control repair; do
  if [ -f "$PAIR/$ARM.log" ]; then cp "$PAIR/$ARM.log" "$OUT/"; fi
done
test ! -e "$TRANSFER"
mkdir "$TRANSFER"
tar -czf "$TRANSFER/results.tar.gz" -C "$OUT" .
cd "$TRANSFER"
sha256sum results.tar.gz > reports.sha256
printf '%s\n' "$PAIR" > pair_directory.txt
printf '%s\n' "$STATUS" > training_exit_code.txt
printf '* -text -filter\n' > .gitattributes
git -c init.defaultBranch=result-transfer init -q
git add -- .gitattributes results.tar.gz reports.sha256 pair_directory.txt training_exit_code.txt
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer fixed seed42 semantic supervision paired reports'
git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$BRANCH"
REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1)
test "$REMOTE_SHA" = "$(git rev-parse HEAD)"
printf 'RESULTS_UPLOAD_OK=%s\nTRAINING_EXIT_CODE=%s\nPAIR_DIRECTORY=%s\n' "$BRANCH" "$STATUS" "$PAIR"
exit "$STATUS"
