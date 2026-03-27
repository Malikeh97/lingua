#!/usr/bin/env bash
# Scan token_count distribution across fineinstructions/fineinstructions_nemotron.
# Streams the dataset without downloading — saves a histogram to OUT_DIR.

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
SCAN_MAX_ROWS="10000000"  # 0 = entire dataset

# Force HF to use a temp dir under scratch so nothing persists to ~/.cache during the run
TMPDIR_HF="$(mktemp -d /scratch/ehghaghi/hf_cache_tmp.XXXXXX)"
export HF_HOME="${TMPDIR_HF}/huggingface"
export HF_DATASETS_CACHE="${TMPDIR_HF}/huggingface/datasets"
export TRANSFORMERS_CACHE="${TMPDIR_HF}/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${TMPDIR_HF}/huggingface/hub"

echo "Temporary HF home: ${TMPDIR_HF}"
echo "Output dir       : ${OUT_DIR}"
echo "Scan max rows    : ${SCAN_MAX_ROWS} (0 = full dataset)"
echo

cleanup() {
    echo "Cleaning up temporary HF cache at: ${TMPDIR_HF}"
    rm -rf "${TMPDIR_HF}"
}
trap cleanup EXIT

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

python -m apps.minimal_squad.download_fineinstructions \
    --out_dir "${OUT_DIR}" \
    --scan-token-counts \
    --scan-max-rows "${SCAN_MAX_ROWS}"
