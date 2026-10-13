#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_TAG=$(date +%Y%m%d-%H%M%S)-$$
REPORT_DIR="/home/hello/szl/eco-spec-results/scope_preflight_${RUN_TAG}"
REMOTE=https://github.com/Andyszl/eco-spec-kg.git
RESULT_BRANCH="codex/scope-preflight-${RUN_TAG}"

CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TASK_ROOT/src" python "$TASK_ROOT/tools/check_scope_inputs_20261013.py" \
  --model /home/hello/szl/models/Qwen3.5-9B \
  --original-prep /home/hello/szl/eco-spec-results/bindings_v26_20261012_110721 \
  --output "$REPORT_DIR"

test -s "$REPORT_DIR/preflight_report.json"
cd "$REPORT_DIR"
sha256sum preflight_report.json coverage_audit.json > reports.sha256
printf '* -text -filter\n' > .gitattributes
git -c init.defaultBranch=result-transfer init -q
git add -- .gitattributes preflight_report.json coverage_audit.json reports.sha256
git -c user.name='EcoSpec results transfer' -c user.email='results@localhost' -c commit.gpgsign=false commit -qm 'Transfer CPU scope-input preflight reports'
git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 push "$REMOTE" "HEAD:refs/heads/$RESULT_BRANCH"
REMOTE_SHA=$(git -c credential.helper= -c credential.username=Andyszl -c http.version=HTTP/1.1 ls-remote "$REMOTE" "refs/heads/$RESULT_BRANCH" | cut -f1)
test "$REMOTE_SHA" = "$(git rev-parse HEAD)"
printf 'RESULTS_UPLOAD_OK=%s\n' "$RESULT_BRANCH"
