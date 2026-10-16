#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=$(date +%Y%m%d-%H%M%S)-$$
PAIR="/home/hello/szl/eco-spec-runs/lora/cross_source_seed42_pair_${TAG}"
OUT="/home/hello/szl/eco-spec-results/cross_source_seed42_pair_${TAG}"
TRANSFER="${OUT}_git"
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
BRANCH="codex/cross-source-seed42-pair-${TAG}"
export PYTHONPATH="$TASK_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
test ! -e "$OUT"
test ! -e "$PAIR"
mkdir -p "$OUT"
STATUS=0
python "$TASK_ROOT/tools/train_cross_source_pair_20261016.py" \
  --model /home/hello/szl/models/Qwen3.5-9B --out "$PAIR" --run 2>&1 | tee "$OUT/launch.log" || STATUS=$?
if [ -s "$PAIR/pair_manifest.json" ]; then
  cp "$PAIR/pair_manifest.json" "$OUT/"
else
  printf 'LAUNCH_BLOCKED_BEFORE_TRAINING\n' >&2
  printf '{"status":"blocked_before_training","training_started":false}\n' > "$OUT/launch_status.json"
  if [ "$STATUS" -eq 0 ]; then STATUS=1; fi
fi
python - "$PAIR" "$OUT" <<'PY'
from pathlib import Path
import sys,shutil
pair,out=map(Path,sys.argv[1:])
if pair.exists():
    for p in pair.rglob("*"):
        if p.is_file() and (p.name in {"trainer_state.json","args.json","sft_args.json","adapter_config.json"} or p.parent==pair and p.suffix==".log"):
            dest=out/"metadata"/p.relative_to(pair)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
PY
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
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer guarded cross-source seed42 pair metadata'
git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$BRANCH"
REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1)
test "$REMOTE_SHA" = "$(git rev-parse HEAD)"
printf 'RESULTS_UPLOAD_OK=%s\nTRAINING_EXIT_CODE=%s\nPAIR_DIRECTORY=%s\n' "$BRANCH" "$STATUS" "$PAIR"
exit "$STATUS"
