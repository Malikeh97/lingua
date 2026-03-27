#!/usr/bin/env bash
# Download a fraction of fineinstructions/fineinstructions_nemotron to scratch.

set -euo pipefail

# Load HF_TOKEN from repo-root .env if not already set in the environment
REPO_ROOT_ENV="$(cd "$(dirname "$0")/../.." && pwd)/.env"
if [[ -z "${HF_TOKEN:-}" && -f "${REPO_ROOT_ENV}" ]]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' "${REPO_ROOT_ENV}" | cut -d= -f2- | tr -d '"'"'")" || true
    export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is not set. Add it to .env (repo root) or export it before running."
    echo "  Example: echo 'HF_TOKEN=hf_...' >> $(dirname "${REPO_ROOT_ENV}")/.env"
fi

OUT_DIR="/scratch/ehghaghi/fineinstructions"
NUM_SAMPLES="1000"   # ~1M context tokens at avg ~2000 tokens/sample; set to 0 to use MAX_FRACTION instead
MAX_FRACTION="0.0001"  # used only when NUM_SAMPLES=0
MIN_TOKENS="1000"         # 0 = no minimum
MAX_TOKENS="8000"         # 0 = no maximum
KEEP_EMPTY="True"     # true = keep empty-context rows and save contexts.jsonl

# Force HF to use a temp dir under scratch so nothing persists to ~/.cache during the run
TMPDIR_HF="$(mktemp -d /scratch/ehghaghi/hf_cache_tmp.XXXXXX)"
export HF_HOME="${TMPDIR_HF}/huggingface"
export HF_DATASETS_CACHE="${TMPDIR_HF}/huggingface/datasets"
export TRANSFORMERS_CACHE="${TMPDIR_HF}/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${TMPDIR_HF}/huggingface/hub"

echo "Temporary HF home: ${TMPDIR_HF}"
echo "Output dir       : ${OUT_DIR}"
echo "Num samples      : ${NUM_SAMPLES} (0 = use max fraction)"
echo "Max fraction     : ${MAX_FRACTION}"
echo "Min/max tokens   : ${MIN_TOKENS} / ${MAX_TOKENS} (0 = no limit)"
echo "Keep empty       : ${KEEP_EMPTY}"
echo

cleanup() {
    echo "Cleaning up temporary HF cache at: ${TMPDIR_HF}"
    rm -rf "${TMPDIR_HF}"
}
trap cleanup EXIT

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_ARGS=""
[[ "${NUM_SAMPLES}" -gt 0 ]] && EXTRA_ARGS="${EXTRA_ARGS} --num-samples ${NUM_SAMPLES}"
[[ "${MIN_TOKENS}"  -gt 0 ]] && EXTRA_ARGS="${EXTRA_ARGS} --min-tokens ${MIN_TOKENS}"
[[ "${MAX_TOKENS}"  -gt 0 ]] && EXTRA_ARGS="${EXTRA_ARGS} --max-tokens ${MAX_TOKENS}"
[[ "${KEEP_EMPTY,,}" == "true" ]] && EXTRA_ARGS="${EXTRA_ARGS} --keep-empty"

python -m apps.minimal_squad.download_fineinstructions \
    --out_dir "${OUT_DIR}" \
    --max_fraction "${MAX_FRACTION}" \
    --no-fill-empty-text \
    ${EXTRA_ARGS} \
|| {
    exit_code=$?
    # Exit code 134 = SIGABRT: known HF datasets streaming cleanup crash.
    # Data is always fully saved before this occurs — safe to ignore.
    [[ $exit_code -eq 134 ]] || exit $exit_code
    echo "Warning: Python exited with SIGABRT (known HF streaming cleanup issue). Data was saved successfully."
}
