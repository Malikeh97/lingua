#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Export script for converting DCP checkpoints to consolidated .pth format.

This script consolidates distributed checkpoints saved during training into a
single .pth file that can be easily loaded for inference.

Usage:
    python -m apps.enc_dec.export_checkpoint --checkpoint_dir /path/to/checkpoint/0000001000

    # Extract only model weights (no optimizer state)
    python -m apps.enc_dec.export_checkpoint --checkpoint_dir /path/to/checkpoint/0000001000 --model_only

Output:
    Creates /path/to/checkpoint/0000001000/consolidated/consolidated.pth
"""

import argparse
import logging
from pathlib import Path

import torch

from lingua.checkpoint import consolidate_checkpoints, CONSOLIDATE_FOLDER, CONSOLIDATE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EXPORT_CHECKPOINT")


def export_model_only(consolidated_path: Path, output_path: Path = None):
    """
    Extract only model weights from a consolidated checkpoint.

    Args:
        consolidated_path: Path to the consolidated.pth file
        output_path: Optional path for the model-only checkpoint.
                     Defaults to model_weights.pth in the same directory.
    """
    if output_path is None:
        output_path = consolidated_path.parent / "model_weights.pth"

    logger.info(f"Loading consolidated checkpoint from {consolidated_path}")
    checkpoint = torch.load(consolidated_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
        logger.info(f"Extracted model state dict with {len(model_state_dict)} keys")
        torch.save(model_state_dict, output_path)
        logger.info(f"Saved model weights to {output_path}")
    else:
        logger.warning("No 'model' key found in checkpoint. Available keys: %s", list(checkpoint.keys()))
        logger.info("Saving full checkpoint as model weights")
        torch.save(checkpoint, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export DCP checkpoint to consolidated .pth format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., /path/to/checkpoints/0000001000)"
    )
    parser.add_argument(
        "--model_only",
        action="store_true",
        help="Extract only model weights (without optimizer state)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for model_only export (default: <checkpoint_dir>/consolidated/model_weights.pth)"
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Check if it's a valid DCP checkpoint
    metadata_file = checkpoint_dir / ".metadata"
    if not metadata_file.exists():
        raise ValueError(
            f"Invalid checkpoint directory: {checkpoint_dir}\n"
            f"Expected to find .metadata file for DCP checkpoint"
        )

    # Consolidate the checkpoint
    logger.info(f"Consolidating checkpoint from {checkpoint_dir}")
    consolidate_path = consolidate_checkpoints(str(checkpoint_dir))
    consolidated_file = consolidate_path / CONSOLIDATE_NAME
    logger.info(f"Consolidated checkpoint saved to {consolidated_file}")

    # Optionally extract model weights only
    if args.model_only:
        output_path = Path(args.output) if args.output else None
        model_path = export_model_only(consolidated_file, output_path)
        logger.info(f"Model-only weights saved to {model_path}")

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
