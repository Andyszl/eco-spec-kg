#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TAG=$(date +%Y%m%d-%H%M%S)-$$
OUT="/home/hello/szl/eco-spec-results/full_graph_scope_preflight_${TAG}"
TRANSFER="${OUT}_git"
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
BRANCH="codex/full-graph-scope-preflight-${TAG}"
export PYTHONPATH="$TASK_ROOT/src" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$TASK_ROOT/experiments/full_graph_scope_pair_20261018/input_manifest.json" "$OUT/"
STATUS=0
python "$TASK_ROOT/tools/full_graph_scope_pair_20261018.py" preflight \
  --model /home/hello/szl/models/Qwen3.5-9B --out "$OUT/preflight" 2>&1 | tee "$OUT/preflight.log" || STATUS=$?
if [ ! -d "$OUT" ]; then
  printf 'MISSING_RESULT_DIRECTORY\n' >&2
  exit 2
fi
test ! -e "$TRANSFER"
mkdir "$TRANSFER"
tar -czf "$TRANSFER/results.tar.gz" -C "$OUT" .
cd "$TRANSFER"
sha256sum results.tar.gz > reports.sha256
printf '%s\n' "$OUT" > result_directory.txt
printf '%s\n' "$STATUS" > preflight_exit_code.txt
printf '* -text -filter\n' > .gitattributes
git -c init.defaultBranch=result-transfer init -q
git add -- .gitattributes results.tar.gz reports.sha256 result_directory.txt preflight_exit_code.txt
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer full-graph scoped input CPU tokenizer preflight'
git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$BRANCH"
REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1)
test "$REMOTE_SHA" = "$(git rev-parse HEAD)"
printf 'RESULTS_UPLOAD_OK=%s\nPREFLIGHT_EXIT_CODE=%s\nPREFLIGHT_DIRECTORY=%s\n' "$BRANCH" "$STATUS" "$OUT"
exit "$STATUS"
