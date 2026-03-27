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
import math
import os
import sqlite3
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


def scan_token_counts(max_rows: int, out_dir: Path):
    """
    Scan token_count distribution by reading parquet files directly via pyarrow.
    Only the token_count column is decoded — all large text columns are skipped,
    keeping memory usage minimal regardless of dataset size.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    limit_str = f"up to {max_rows:,}" if max_rows > 0 else "all"
    print(f"\nScanning token_count distribution ({limit_str} rows) ...")

    fs = HfFileSystem()
    parquet_files = sorted(fs.glob(f"datasets/{DATASET_NAME}/data/train-*.parquet"))
    print(f"  Found {len(parquet_files):,} parquet files")

    NUM_BINS = 100
    BIN_MAX = 50_000
    bin_width = BIN_MAX / NUM_BINS
    bin_counts = [0] * NUM_BINS
    overflow_count = 0

    n = 0
    mean = 0.0
    M2 = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    done = False
    for filepath in parquet_files:
        if done:
            break
        with fs.open(filepath, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["token_count"]):
                col = batch.column("token_count")
                for tc_val in col:
                    if tc_val is None:
                        continue
                    tc = int(tc_val.as_py())
                    if tc == 0:
                        continue  # skip empty-context rows (token_count=0 means instruction row, not truly empty)
                    n += 1
                    delta = tc - mean
                    mean += delta / n
                    M2 += delta * (tc - mean)
                    if tc < min_val:
                        min_val = tc
                    if tc > max_val:
                        max_val = tc
                    if tc >= BIN_MAX:
                        overflow_count += 1
                    else:
                        bin_counts[min(int(tc / bin_width), NUM_BINS - 1)] += 1

                    if n % 1_000_000 == 0:
                        print(f"  scanned {n:,} rows ...")

                    if max_rows > 0 and n >= max_rows:
                        done = True
                        break
                if done:
                    break

    _report_token_stats(n, mean, M2, min_val, max_val, bin_counts, overflow_count, bin_width, BIN_MAX, out_dir)


def _report_token_stats(n, mean, M2, min_val, max_val, bin_counts, overflow_count, bin_width, BIN_MAX, out_dir):
    """Print token count statistics and save a histogram. Called after the streaming pass."""
    if n == 0:
        print("  No token_count values found — skipping distribution analysis.")
        return

    std = math.sqrt(M2 / (n - 1)) if n > 1 else 0.0

    half = n / 2
    cumsum = 0
    median_est = float("nan")
    for i, count in enumerate(bin_counts):
        if cumsum + count >= half:
            fraction = (half - cumsum) / count if count > 0 else 0.5
            median_est = (i + fraction) * bin_width
            break
        cumsum += count

    lines = [
        f"Token count statistics (n={n:,} rows scanned):",
        f"  Mean   : {mean:.1f}",
        f"  Median : ~{median_est:.0f}  (estimated from histogram)",
        f"  Std    : {std:.1f}",
        f"  Min    : {min_val:,}",
        f"  Max    : {max_val:,}",
    ]
    if overflow_count:
        lines.append(f"  (Note: {overflow_count:,} rows had token_count > {BIN_MAX:,} and are shown in overflow bin)")

    print("\n" + "\n".join(lines))

    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"token_count_stats_{n // 1_000_000}M.txt"
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"  Stats saved      → {stats_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bin_edges = [i * bin_width for i in range(len(bin_counts) + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_counts))]
        bin_centers_plot = bin_centers + [BIN_MAX + bin_width / 2]
        bin_counts_plot = bin_counts + [overflow_count]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(bin_centers_plot, bin_counts_plot, width=bin_width * 0.9, color="steelblue", edgecolor="none")
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean:.0f}")
        ax.axvline(median_est, color="orange", linestyle="--", linewidth=1.5, label=f"median≈{median_est:.0f}")
        ax.set_xlabel("token_count")
        ax.set_ylabel("row count")
        ax.set_title(f"Token count distribution  (n={n:,})")
        ax.legend()
        ax.set_xlim(left=0)
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "token_count_dist.png"
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Histogram saved → {plot_path}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")


def _resolve_contexts(need_context_write: set[str], contexts_db: sqlite3.Connection) -> tuple[int, int]:
    """
    Second lightweight pass: scan just warc_record_id + text columns via pyarrow
    to find document texts for wids not resolved during the main pass.
    Inserts rows into the contexts SQLite DB.
    Returns (resolved, unresolved).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    remaining = set(need_context_write)
    resolved = 0
    print(f"\n  Second pass: resolving {len(remaining):,} remaining contexts ...")

    fs = HfFileSystem()
    parquet_files = sorted(fs.glob(f"datasets/{DATASET_NAME}/data/train-*.parquet"))

    for filepath in parquet_files:
        if not remaining:
            break
        with fs.open(filepath, "rb") as pf_file:
            pf = pq.ParquetFile(pf_file)
            for batch in pf.iter_batches(columns=["warc_record_id", "text"]):
                rows = []
                for wid_val, text_val in zip(batch.column("warc_record_id"), batch.column("text")):
                    wid = (wid_val.as_py() or "")
                    text = (text_val.as_py() or "").strip()
                    if text and wid in remaining:
                        rows.append((wid, text))
                        remaining.discard(wid)
                        resolved += 1
                if rows:
                    contexts_db.executemany(
                        "INSERT OR IGNORE INTO contexts (warc_record_id, text) VALUES (?, ?)", rows
                    )
                    contexts_db.commit()
                if not remaining:
                    break

    return resolved, len(remaining)


def stream_and_join(
    target_samples: int,
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    val_fraction: float = 0.05,
    fill_empty_text: bool = True,
    keep_empty: bool = False,
    min_tokens: int = 0,
    max_tokens: int = 0,
) -> tuple[int, int]:
    """
    Stream the dataset and join document rows with instruction rows by warc_record_id.

    Writes samples directly to disk (no in-memory accumulation).
    val_n is pre-computed from target_samples so the first val_n samples go to val,
    the rest to train.

    keep_empty=True: include samples with empty context and save contexts.jsonl
        mapping warc_record_id → text so training code can populate them later.
    min_tokens / max_tokens: skip samples outside this token_count range (0 = no limit).

    Returns:
        (train_count, val_count)
    """
    from datasets import load_dataset

    val_n = max(1, int(target_samples * val_fraction))

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    contexts_db_path = out_dir / "contexts.db"

    print(f"Streaming {DATASET_NAME} (single pass) ...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True).select_columns(
        ["warc_record_id", "text", "instantiated_instruction", "answer", "token_count"]
    )

    # text_lookup only needed for fill_empty_text join path
    text_lookup: dict[str, str] = {} if fill_empty_text else {}
    pending: dict[str, list] = defaultdict(list)
    # wids of emitted empty-context samples whose text hasn't been written to contexts.jsonl yet
    need_context_write: set[str] = set()
    # keep_empty token-count filtering: wid → document token_count (int only, cheap memory)
    text_tc_lookup: dict[str, int] = {}
    # instruction rows waiting for their matching document row (keep_empty + fill_empty_text=False)
    pending_empty: dict[str, list] = defaultdict(list)

    rows_scanned = 0
    complete_count = 0
    sample_counter = 0
    rows_skipped_no_instruction = 0
    rows_skipped_token_filter = 0

    ctx_db = None
    if keep_empty:
        ctx_db = sqlite3.connect(str(contexts_db_path))
        ctx_db.execute("PRAGMA journal_mode=WAL")
        ctx_db.execute("PRAGMA synchronous=NORMAL")
        ctx_db.execute(
            "CREATE TABLE IF NOT EXISTS contexts "
            "(warc_record_id TEXT PRIMARY KEY, text TEXT)"
        )
        ctx_db.commit()

    try:
        with open(val_path, "w") as val_f, open(train_path, "w") as train_f:

            def emit(row: dict, text: str, override_tc: int | None = None):
                nonlocal complete_count, sample_counter
                nonlocal rows_skipped_token_filter
                question = (row.get("instantiated_instruction") or "").strip()
                answer = (row.get("answer") or "").strip()
                if not question or not answer:
                    return

                # Token count filter — override_tc lets empty-context rows use the
                # matching document row's token_count instead of their own (which is 0)
                tc = override_tc if override_tc is not None else row.get("token_count")
                tc_int = int(tc) if tc is not None else None
                if tc_int is not None:
                    if min_tokens > 0 and tc_int < min_tokens:
                        rows_skipped_token_filter += 1
                        return
                    if max_tokens > 0 and tc_int > max_tokens:
                        rows_skipped_token_filter += 1
                        return

                wid = row.get("warc_record_id", "")
                sample = json.dumps({
                    "sample_id": sample_counter,
                    "context": text.strip(),
                    "question": question,
                    "answers": {"text": [answer]},
                    "warc_record_id": wid,
                    "token_count": tc,
                })
                if complete_count < val_n:
                    val_f.write(sample + "\n")
                else:
                    train_f.write(sample + "\n")
                complete_count += 1
                sample_counter += 1

                # If context is empty, mark this wid as needing a contexts.jsonl entry
                if keep_empty and not text.strip():
                    need_context_write.add(wid)

            for row in ds:
                rows_scanned += 1
                wid = row.get("warc_record_id", "")
                text = (row.get("text") or "").strip()
                instruction = (row.get("instantiated_instruction") or "").strip()
                answer = (row.get("answer") or "").strip()

                if text:
                    if keep_empty and not fill_empty_text:
                        # Store document tc so instruction rows arriving later can filter correctly
                        doc_tc = row.get("token_count")
                        doc_tc_int = int(doc_tc) if doc_tc is not None else 0
                        text_tc_lookup[wid] = doc_tc_int
                        # Flush any instruction rows that were waiting for this document
                        if wid in pending_empty:
                            for pending_row in pending_empty.pop(wid):
                                emit(pending_row, "", override_tc=doc_tc_int)
                    if fill_empty_text:
                        # Only store text in memory when needed for the join
                        text_lookup[wid] = text
                        if wid in pending:
                            for pending_row in pending.pop(wid):
                                emit(pending_row, text)
                    if instruction and answer:
                        emit(row, text)
                    # Write context for any empty-context samples whose wid just arrived
                    if keep_empty and wid in need_context_write:
                        ctx_db.execute(
                            "INSERT OR IGNORE INTO contexts (warc_record_id, text) VALUES (?, ?)",
                            (wid, text),
                        )
                        need_context_write.discard(wid)
                else:
                    if fill_empty_text:
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
                    else:
                        if keep_empty:
                            # Use document token_count for filter if available, else defer
                            if instruction and answer:
                                if wid in text_tc_lookup:
                                    emit(row, "", override_tc=text_tc_lookup[wid])
                                else:
                                    pending_empty[wid].append(row)
                            else:
                                rows_skipped_no_instruction += 1
                        else:
                            if instruction and answer:
                                emit(row, "")
                            else:
                                rows_skipped_no_instruction += 1

                if rows_scanned % 100_000 == 0:
                    print(
                        f"  scanned {rows_scanned:,} | complete {complete_count:,} "
                        f"| pending wids {len(pending):,}"
                        + (f" | text_lookup {len(text_lookup):,}" if fill_empty_text else "")
                    )

                if complete_count >= target_samples:
                    print(f"  Reached target of {target_samples:,} complete samples.")
                    break

    finally:
        if ctx_db:
            ctx_db.commit()
        del ds  # explicitly release streaming iterator to avoid C++ destructor crash on shutdown

    # Second pass: resolve any contexts not found during the main pass
    if keep_empty and need_context_write:
        resolved, unresolved = _resolve_contexts(need_context_write, ctx_db)
        print(f"  Contexts resolved : {resolved:,}  |  not found: {unresolved:,}")

    if ctx_db:
        ctx_db.execute("ANALYZE")
        ctx_db.close()

    rows_skipped_no_text_found = sum(len(v) for v in pending.values())
    rows_skipped_no_doc_seen = sum(len(v) for v in pending_empty.values())
    train_count = max(0, complete_count - val_n)
    actual_val_count = min(complete_count, val_n)

    print(f"\nStream summary:")
    print(f"  Rows scanned          : {rows_scanned:,}")
    print(f"  Complete samples      : {complete_count:,}")
    print(f"  Skipped (no instr/ans): {rows_skipped_no_instruction:,}")
    print(f"  Skipped (token filter): {rows_skipped_token_filter:,}")
    print(f"  Skipped (text missing): {rows_skipped_no_text_found:,}")
    if keep_empty:
        print(f"  Dropped (doc not seen): {rows_skipped_no_doc_seen:,}")
    print(f"  text_lookup size      : {len(text_lookup):,}")
    if keep_empty:
        ctx_size_mb = contexts_db_path.stat().st_size / 1e6 if contexts_db_path.exists() else 0
        print(f"  Contexts saved        → {contexts_db_path}  ({ctx_size_mb:.1f} MB)")

    return train_count, actual_val_count


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
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to download. Overrides --max_fraction when set (default: 0 = use --max_fraction)",
    )
    parser.add_argument(
        "--max_fraction",
        type=float,
        default=DEFAULT_MAX_FRACTION,
        help=f"Fraction of total dataset to download, used when --num-samples is not set (default: {DEFAULT_MAX_FRACTION})",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Skip samples with token_count below this value (default: 0 = no minimum)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Skip samples with token_count above this value (default: 0 = no maximum)",
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
        "--fill-empty-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Join empty-context rows with their document text via warc_record_id "
             "(default: True). Use --no-fill-empty-text to emit empty context as-is.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="When using --no-fill-empty-text, include samples with empty context and "
             "save contexts.jsonl (warc_record_id → text) for population at training time.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan (row targets, paths) without downloading",
    )
    parser.add_argument(
        "--scan-token-counts",
        action="store_true",
        help="Scan the full dataset for token_count distribution only (no download). "
             "Use --scan-max-rows to cap the scan.",
    )
    parser.add_argument(
        "--scan-max-rows",
        type=int,
        default=0,
        help="Max rows to scan when --scan-token-counts is set (0 = entire dataset, default: 0)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.scan_token_counts:
        scan_token_counts(args.scan_max_rows, out_dir)
        return

    if args.max_fraction > 0.03 and args.num_samples == 0:
        print(
            f"WARNING: max_fraction={args.max_fraction} exceeds the 0.03 (3%) safety limit. "
            "The dataset is 1.71 TB. Proceeding anyway — make sure you have enough disk space."
        )

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    # Determine target sample count
    total_rows = get_total_rows()
    if args.num_samples > 0:
        target_samples = args.num_samples
        fraction = target_samples / total_rows
    else:
        target_samples = int(total_rows * args.max_fraction)
        fraction = args.max_fraction
    target_size_gb = TOTAL_SIZE_TB * 1000 * fraction

    print(f"\nDownload plan:")
    print(f"  Dataset        : {DATASET_NAME}")
    print(f"  Total rows est.: {total_rows:,}")
    print(f"  Fraction       : {fraction:.1%}")
    print(f"  Target samples : {target_samples:,}")
    print(f"  Min tokens     : {args.min_tokens if args.min_tokens else 'none'}")
    print(f"  Max tokens     : {args.max_tokens if args.max_tokens else 'none'}")
    print(f"  Estimated size : ~{target_size_gb:.1f} GB")
    print(f"  Output dir     : {out_dir}")
    print(f"  Train          : {train_path}")
    print(f"  Val            : {val_path}")
    print(f"  Fill empty text: {args.fill_empty_text}")
    print(f"  Keep empty     : {args.keep_empty}")

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
    train_count, val_count = stream_and_join(
        target_samples, train_path, val_path, out_dir,
        val_fraction=args.val_fraction,
        fill_empty_text=args.fill_empty_text,
        keep_empty=args.keep_empty,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    train_size_mb = train_path.stat().st_size / 1e6 if train_path.exists() else 0
    val_size_mb = val_path.stat().st_size / 1e6 if val_path.exists() else 0
    print(f"\nSaved {train_count:,} train rows → {train_path}  ({train_size_mb:.1f} MB)")
    print(f"Saved {val_count:,} val rows   → {val_path}  ({val_size_mb:.1f} MB)")

    # Write a metadata file alongside the data
    meta = {
        "dataset": DATASET_NAME,
        "target_samples": target_samples,
        "fraction": fraction,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "fill_empty_text": args.fill_empty_text,
        "keep_empty": args.keep_empty,
        "total_rows_estimate": total_rows,
        "train_samples": train_count,
        "val_samples": val_count,
        "columns": ["context", "question", "answers", "warc_record_id", "token_count"],
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata       → {meta_path}")

    print(f"\nDone. {train_count:,} train + {val_count:,} val samples saved.")


if __name__ == "__main__":
    main()
