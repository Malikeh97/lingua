"""
Test data pipeline: build the same pipeline as training, generate batches, dump for inspection.

Usage:
    python -m apps.addon_tests.test_data_pipeline \
        config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_train.yaml \
        num_batches=5

    # Multi-task
    python -m apps.addon_tests.test_data_pipeline \
        config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_multitask.yaml \
        num_batches=5

Outputs:
    - Console: per-batch stats (num_examples, enc/dec token counts, dtypes)
    - File: outputs/test_data_pipeline/batches.jsonl (text-level inspection)
"""

import json
import os
import sys
from pathlib import Path

import ray
import torch

from addons.data.config import DataArgs
from addons.models.config import ModelArgs
from apps.finesearch.config_utils import parse_cli, load_yaml, deep_merge, dataclass_defaults, dict_to_dataclass

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TestConfig:
    data: DataArgs = field(default_factory=DataArgs)
    model: ModelArgs = field(default_factory=ModelArgs)
    doc_max_tokens: int = 8192
    target_max_tokens: int = 2048
    num_batches: int = 5
    output_dir: str = "outputs/test_data_pipeline"


def main():
    cli = parse_cli()
    if "config" not in cli:
        print("Usage: python -m apps.addon_tests.test_data_pipeline config=<yaml>[,<yaml>] [num_batches=N]")
        sys.exit(1)

    config_arg = cli.pop("config")
    config_paths = config_arg if isinstance(config_arg, list) else \
        [p.strip() for p in str(config_arg).split(",")] if "," in str(config_arg) else [config_arg]

    merged = dataclass_defaults(TestConfig)
    for path in config_paths:
        merged = deep_merge(merged, load_yaml(path))
    merged = deep_merge(merged, cli)
    cfg = dict_to_dataclass(TestConfig, merged)

    # Import tasks so they register
    import addons.tasks.squad  # noqa: F401
    import addons.tasks.hotpotqa  # noqa: F401
    import addons.tasks.fineinstructions  # noqa: F401

    # Build tokenizer factory
    from addons.models.encoder_decoder import EncoderDecoder
    from addons.models.decoder import Decoder
    model_cls = EncoderDecoder if cfg.model.model_type == "encdec" else Decoder
    tokenizer_factory = model_cls.tokenizer_factory(cfg.model)
    encoder_tokenizer, decoder_tokenizer = tokenizer_factory()  # local copy for decoding

    # Build pipeline
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    from apps.finesearch.train import build_data_pipeline
    pipeline = build_data_pipeline(cfg.data, split="train", tokenizer_factory=tokenizer_factory)

    # Output directory
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "batches.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Data Pipeline Test")
    print(f"{'='*60}")
    print(f"Sources: {[s.get('task', s) for s in cfg.data.sources]}")
    print(f"Target tokens: {cfg.data.target_tokens}")
    print(f"Doc max tokens: {cfg.data.doc_max_tokens}")
    print(f"Target max tokens: {cfg.data.target_max_tokens}")
    print(f"Generating {cfg.num_batches} batches...")
    print(f"{'='*60}\n")

    with open(jsonl_path, "w") as f:
        for batch_idx in range(cfg.num_batches):
            print(f"--- Batch {batch_idx} ---")
            batch = ray.get(pipeline.get_batch.remote())

            if batch is None:
                print(f"  Pipeline exhausted at batch {batch_idx}")
                break

            num_examples = batch.batch_size
            enc_tokens = batch.encoder_tokens.tokens.numel()
            dec_tokens = batch.decoder_tokens.tokens.numel()
            num_labels = (batch.labels != -100).sum().item()

            print(f"  Examples: {num_examples}")
            print(f"  Unique docs: {batch.encoder_tokens.num_seqs}")
            print(f"  Encoder tokens: {enc_tokens} ({batch.encoder_tokens.num_seqs} docs, "
                  f"lengths: min={min(batch.encoder_tokens.lengths)}, "
                  f"max={max(batch.encoder_tokens.lengths)})")
            print(f"  Decoder tokens: {dec_tokens} ({batch.decoder_tokens.num_seqs} seqs, "
                  f"lengths: min={min(batch.decoder_tokens.lengths)}, "
                  f"max={max(batch.decoder_tokens.lengths)})")
            print(f"  Label tokens: {num_labels} (non -100)")
            print(f"  Total tokens: {enc_tokens + dec_tokens}")

            # Show first 2 examples (decode from token IDs)
            for j in range(min(2, num_examples)):
                doc_idx = batch.example_doc_indices[j][0] if batch.example_doc_indices[j] else 0
                enc_s = batch.encoder_tokens.cu_seqlens[doc_idx].item()
                enc_e = batch.encoder_tokens.cu_seqlens[doc_idx + 1].item()
                enc_text = encoder_tokenizer.decode(
                    batch.encoder_tokens.tokens[enc_s:enc_e], skip_special_tokens=False
                )

                dec_s = batch.decoder_tokens.cu_seqlens[j].item()
                dec_e = batch.decoder_tokens.cu_seqlens[j + 1].item()
                dec_text = decoder_tokenizer.decode(
                    batch.decoder_tokens.tokens[dec_s:dec_e], skip_special_tokens=False
                )

                label_ids = batch.labels[dec_s:dec_e]
                label_text = decoder_tokenizer.decode(
                    label_ids[label_ids != -100], skip_special_tokens=False
                )

                if len(enc_text) > 400:
                    enc_text = enc_text[:200] + " ... " + enc_text[-200:]

                print(f"  Example[{j}]:")
                print(f"    enc_decoded: {enc_text!r}")
                print(f"    dec_decoded: {dec_text!r}")
                print(f"    label_decoded: {label_text!r}")
                print(f"    doc_indices: {batch.example_doc_indices[j]}")

            # Write to jsonl
            batch_record = {
                "batch_idx": batch_idx,
                "num_examples": num_examples,
                "enc_tokens": enc_tokens,
                "dec_tokens": dec_tokens,
                "label_tokens": num_labels,
                "enc_doc_lengths": batch.encoder_tokens.lengths,
                "dec_seq_lengths": batch.decoder_tokens.lengths,
            }
            f.write(json.dumps(batch_record) + "\n")

    # Pipeline stats
    stats = ray.get(pipeline.get_stats.remote())
    print(f"\n{'='*60}")
    print(f"Pipeline Stats")
    print(f"{'='*60}")
    print(f"  Source counts: {stats['source_counts']}")
    print(f"  Batches emitted: {stats['batches_emitted']}")
    print(f"  Buffer size: {stats['buffer_size']}")
    print(f"  Output: {jsonl_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
