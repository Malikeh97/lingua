#!/usr/bin/env python3
"""
Download a fraction of fineinstructions/fineinstructions_nemotron to scratch.

Dataset structure:
  - Some rows have non-empty `text` (the source document) but may share a warc_record_id
    with instruction rows that have empty `text`.
  - Instruction rows: {warc_record_id, text="", instantiated_instruction="...", answer="..."}
  - Document rows: {warc_record_id, text="<doc>", instantiated_instruction=..., answer=...}
  We join by warc_record_id so every saved sample has (context, question, answer).

Usage:
    python -m apps.minimal_squad.download_fineinstructions
    python -m apps.minimal_squad.download_fineinstructions --max_fraction 0.01 --out_dir /scratch/ehghaghi/fineinstructions
    python -m apps.minimal_squad.download_fineinstructions --dry_run
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

DATASET_NAME = "fineinstructions/fineinstructions_nemotron"
TOTAL_SIZE_TB = 1.71
DEFAULT_SCRATCH = "/scratch/ehghaghi/fineinstructions"
DEFAULT_MAX_FRACTION = 0.003

# ── Estimated total rows ──────────────────────────────────────────────────────
# We fetch from dataset info if available; this is the fallback.
# (1.71 TB / ~2 KB avg row ≈ 855 M rows; conservative estimate: 500 M)
FALLBACK_TOTAL_ROWS = 500_000_000


def get_total_rows() -> int:
    """Try to read exact total rows from dataset metadata without downloading data."""
    try:
        from datasets import load_dataset_builder

        print("Fetching dataset info (no data downloaded)...")
        builder = load_dataset_builder(DATASET_NAME)
        info = builder.info
        if info.splits and "train" in info.splits:
            n = info.splits["train"].num_examples
            if n:
                print(f"  Dataset has {n:,} train rows ({info.dataset_size / 1e12:.2f} TB)")
                return n
    except Exception as e:
        print(f"  Could not fetch dataset info: {e}")
    print(f"  Falling back to estimated {FALLBACK_TOTAL_ROWS:,} total rows")
    return FALLBACK_TOTAL_ROWS


def stream_and_join(target_samples: int, val_fraction: float = 0.05):
    """
    Stream the dataset and join document rows with instruction rows by warc_record_id.

    Strategy (single pass):
      - text_lookup: {warc_record_id -> text} for rows where text is non-empty
      - pending:     {warc_record_id -> [rows waiting for their text]}
      When we see a row with non-empty text:
        1. Store it in text_lookup.
        2. Flush any pending rows for that warc_record_id.
      When we see a row with empty text:
        1. If warc_record_id already in text_lookup → emit immediately.
        2. Otherwise → add to pending.
      We stop once we have collected `target_samples` complete samples.

    Returns:
        (train_samples, val_samples): lists of dicts with keys
            {context, question, answer, warc_record_id}
    """
    from datasets import load_dataset

    print(f"Streaming {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    text_lookup: dict[str, str] = {}    # warc_record_id -> text
    pending: dict[str, list] = defaultdict(list)  # warc_record_id -> list of incomplete rows
    complete: list[dict] = []

    rows_scanned = 0
    rows_skipped_no_instruction = 0
    rows_skipped_no_text_found = 0

    def emit(row: dict, text: str):
        """Build a complete sample from a row + resolved text."""
        question = (row.get("instantiated_instruction") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            return  # nothing useful
        complete.append(
            {
                "context": text.strip(),
                "question": question,
                "answer": answer,
                "warc_record_id": row.get("warc_record_id", ""),
            }
        )

    for row in ds:
        rows_scanned += 1
        wid = row.get("warc_record_id", "")
        text = (row.get("text") or "").strip()
        instruction = (row.get("instantiated_instruction") or "").strip()
        answer = (row.get("answer") or "").strip()

        if text:
            # Store / update text for this warc_record_id
            text_lookup[wid] = text
            # Flush any pending rows that were waiting for this wid
            if wid in pending:
                for pending_row in pending.pop(wid):
                    emit(pending_row, text)
            # If this row itself also has instruction+answer, emit it directly
            if instruction and answer:
                emit(row, text)

        else:
            # No text on this row — look up or defer
            if wid in text_lookup:
                if instruction and answer:
                    emit(row, text_lookup[wid])
                else:
                    rows_skipped_no_instruction += 1
            else:
                if instruction and answer:
                    pending[wid].append(row)
                else:
                    rows_skipped_no_instruction += 1

        # Progress log every 100k rows
        if rows_scanned % 100_000 == 0:
            print(
                f"  scanned {rows_scanned:,} | complete {len(complete):,} "
                f"| pending wids {len(pending):,} | text_lookup {len(text_lookup):,}"
            )

        if len(complete) >= target_samples:
            print(f"  Reached target of {target_samples:,} complete samples.")
            break

    # Drain any remaining pending rows whose warc_record_id text was never found
    rows_skipped_no_text_found = sum(len(v) for v in pending.values())

    print(f"\nStream summary:")
    print(f"  Rows scanned          : {rows_scanned:,}")
    print(f"  Complete samples      : {len(complete):,}")
    print(f"  Skipped (no instr/ans): {rows_skipped_no_instruction:,}")
    print(f"  Skipped (text missing): {rows_skipped_no_text_found:,}")
    print(f"  text_lookup size      : {len(text_lookup):,}")

    # Split into train / val
    val_n = max(1, int(len(complete) * val_fraction))
    train_samples = complete[val_n:]
    val_samples = complete[:val_n]
    return train_samples, val_samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {len(samples):,} rows → {path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download fineinstructions dataset slice")
    parser.add_argument(
        "--max_fraction",
        type=float,
        default=DEFAULT_MAX_FRACTION,
        help=f"Fraction of total dataset to download (default: {DEFAULT_MAX_FRACTION})",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_SCRATCH,
        help=f"Output directory on scratch (default: {DEFAULT_SCRATCH})",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of collected samples to reserve for validation (default: 0.05)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan (row targets, paths) without downloading",
    )
    args = parser.parse_args()

    if args.max_fraction > 0.03:
        print(
            f"WARNING: max_fraction={args.max_fraction} exceeds the 0.03 (3%) safety limit. "
            "The dataset is 1.71 TB. Proceeding anyway — make sure you have enough disk space."
        )

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    # Estimate target row count
    total_rows = get_total_rows()
    target_samples = int(total_rows * args.max_fraction)
    target_size_gb = TOTAL_SIZE_TB * 1000 * args.max_fraction

    print(f"\nDownload plan:")
    print(f"  Dataset        : {DATASET_NAME}")
    print(f"  Total rows est.: {total_rows:,}")
    print(f"  Fraction       : {args.max_fraction:.1%}")
    print(f"  Target samples : {target_samples:,}")
    print(f"  Estimated size : ~{target_size_gb:.1f} GB")
    print(f"  Output dir     : {out_dir}")
    print(f"  Train          : {train_path}")
    print(f"  Val            : {val_path}")

    # Check scratch disk space
    try:
        import shutil
        free_gb = shutil.disk_usage(out_dir.parent if not out_dir.exists() else out_dir).free / 1e9
        print(f"  Scratch free   : {free_gb:.1f} GB")
        if free_gb < target_size_gb * 1.2:
            print(
                f"WARNING: Only {free_gb:.1f} GB free but need ~{target_size_gb * 1.2:.1f} GB "
                "(estimated + 20% buffer). Proceed with caution."
            )
    except Exception:
        pass

    if args.dry_run:
        print("\nDry run complete — no data downloaded.")
        return

    print()
    train_samples, val_samples = stream_and_join(target_samples, args.val_fraction)

    print(f"\nSaving...")
    save_jsonl(train_samples, train_path)
    save_jsonl(val_samples, val_path)

    # Write a metadata file alongside the data
    meta = {
        "dataset": DATASET_NAME,
        "max_fraction": args.max_fraction,
        "total_rows_estimate": total_rows,
        "target_samples": target_samples,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "columns": ["context", "question", "answer", "warc_record_id"],
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata       → {meta_path}")

    print(f"\nDone. {len(train_samples):,} train + {len(val_samples):,} val samples saved.")


if __name__ == "__main__":
    main()
