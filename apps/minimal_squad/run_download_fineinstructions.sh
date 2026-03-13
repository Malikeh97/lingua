#!/usr/bin/env bash
# Run download_fineinstructions.py with a specific output dir and fraction,
# then wipe the HuggingFace cache so nothing lingers locally.

set -euo pipefail

OUT_DIR="${1:-/scratch/ehghaghi/fineinstructions}"
MAX_FRACTION="${2:-0.0001}"

# # Clear HuggingFace cache before download (streaming still writes temp files)
# HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
# echo "Clearing HuggingFace cache at: ${HF_CACHE}"
# rm -rf "${HF_CACHE}"

# Force HF to use a temp dir under scratch so nothing persists to ~/.cache during the run
TMPDIR_HF="$(mktemp -d /scratch/ehghaghi/hf_cache_tmp)"
export HF_HOME="${TMPDIR_HF}/huggingface"
export HF_DATASETS_CACHE="${TMPDIR_HF}/huggingface/datasets"
export TRANSFORMERS_CACHE="${TMPDIR_HF}/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${TMPDIR_HF}/huggingface/hub"

echo "Temporary HF home: ${TMPDIR_HF}"
echo "Output dir       : ${OUT_DIR}"
echo "Max fraction     : ${MAX_FRACTION}"
echo

cleanup() {
    echo "Cleaning up temporary HF cache at: ${TMPDIR_HF}"
    rm -rf "${TMPDIR_HF}"
}
trap cleanup EXIT

# Run from the repo root so the -m flag resolves correctly
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

python -m apps.minimal_squad.download_fineinstructions \
    --out_dir "${OUT_DIR}" \
    --max_fraction "${MAX_FRACTION}"
