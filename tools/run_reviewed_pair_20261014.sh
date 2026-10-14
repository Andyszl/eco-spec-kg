#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_TAG=$(date +%Y%m%d-%H%M%S)-$$
PAIR_DIR="/home/hello/szl/eco-spec-runs/lora/reviewed_seed42_pair_${RUN_TAG}"
REPORT_DIR="/home/hello/szl/eco-spec-results/reviewed_seed42_pair_${RUN_TAG}"
PREFLIGHT=/home/hello/szl/eco-spec-results/scope_preflight_20261014-092238-1073350/preflight_report.json
MODEL=/home/hello/szl/models/Qwen3.5-9B
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
RESULT_BRANCH="codex/seed42-pair-${RUN_TAG}"
export PYTHONPATH="$TASK_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0

# The launcher validates the review, real preflight, unchanged inputs, installed
# Swift options, explicit partitions and an idle BF16-capable GPU before training.
TRAIN_STATUS=0
python "$TASK_ROOT/tools/train_reviewed_pair_20261014.py" \
  --model "$MODEL" --preflight "$PREFLIGHT" --out "$PAIR_DIR" --run || TRAIN_STATUS=$?

if [ ! -s "$PAIR_DIR/pair_manifest.json" ]; then
  printf 'LAUNCH_BLOCKED_BEFORE_TRAINING; no training manifest created\n' >&2
  if [ "$TRAIN_STATUS" -eq 0 ]; then TRAIN_STATUS=1; fi
  exit "$TRAIN_STATUS"
fi
test ! -e "$REPORT_DIR"
mkdir -p "$REPORT_DIR"
cp "$PAIR_DIR/pair_manifest.json" "$REPORT_DIR/"
for ARM in control repair; do
  if [ -f "$PAIR_DIR/$ARM.log" ]; then cp "$PAIR_DIR/$ARM.log" "$REPORT_DIR/"; fi
done
cd "$REPORT_DIR"
sha256sum pair_manifest.json > reports.sha256
for ARM in control repair; do
  if [ -f "$ARM.log" ]; then sha256sum "$ARM.log" >> reports.sha256; fi
done
printf '* -text -filter\n' > .gitattributes
git -c init.defaultBranch=result-transfer init -q
git add -- .gitattributes pair_manifest.json reports.sha256
for ARM in control repair; do
  if [ -f "$ARM.log" ]; then git add -- "$ARM.log"; fi
done
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer reviewed seed42 paired training reports'
git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$RESULT_BRANCH"
REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$RESULT_BRANCH" | cut -f1)
test "$REMOTE_SHA" = "$(git rev-parse HEAD)"
printf 'RESULTS_UPLOAD_OK=%s\nTRAINING_EXIT_CODE=%s\nPAIR_DIRECTORY=%s\n' "$RESULT_BRANCH" "$TRAIN_STATUS" "$PAIR_DIR"
exit "$TRAIN_STATUS"
