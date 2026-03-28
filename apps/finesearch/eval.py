"""
Evaluation script for FineSearch encoder-decoder.

Usage:
    python -m apps.finesearch.eval config=configs/finesearch/eval.yaml

Supports:
- Task-specific metrics (F1, EM, ROUGE, etc.)
- Multiple tasks in single run
- Checkpoint loading with config inference
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from lingua.args import dataclass_from_dict, dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)

from addons.distributed import DistributedArgs

from transformers import AutoTokenizer

from addons.models.config import ModelArgs, GenerationArgs
from addons.models.encoder_decoder import EncoderDecoder
from addons.models.decoder import Decoder
from addons.data.collate import TokenizedBatch
from addons.tasks.registry import get_task, list_tasks
from addons.tasks.base import BaseTask
from addons.tasks.schema import ContextBasedExample

logger = logging.getLogger()


# ==================== Configuration ====================


@dataclass
class EvalArgs:
    """Evaluation-specific configuration."""

    # Checkpoint
    ckpt_dir: str = ""

    # Tasks
    tasks: List[str] = field(default_factory=lambda: ["squad"])
    split: str = "validation"

    # Batching
    batch_size: int = 8
    max_samples: Optional[int] = None  # None = all samples

    # Output
    dump_dir: Optional[str] = None
    metric_log_dir: Optional[str] = None

    # Device
    device: str = "cuda"

    # For tracking (set by train script)
    global_step: Optional[int] = None


@dataclass
class EvalConfig:
    """Full evaluation configuration."""

    name: str = "finesearch-eval"
    model: ModelArgs = field(default_factory=ModelArgs)
    generation: GenerationArgs = field(default_factory=GenerationArgs)
    eval: EvalArgs = field(default_factory=EvalArgs)

    # Fallback decoder tokenizer (used when model.decoder_name is empty)
    default_decoder_tokenizer: str = "meta-llama/Llama-3.2-1B"


# ==================== Model Loading ====================


def load_model_and_tokenizers(cfg: EvalConfig):
    """Load model from checkpoint and build tokenizers."""
    ckpt_path = Path(cfg.eval.ckpt_dir)

    # Check for consolidated checkpoint
    if (ckpt_path / "params.json").exists():
        consolidate_path = ckpt_path
    else:
        consolidate_path = ckpt_path / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(str(ckpt_path))

    # Load training config to infer model args
    params_path = consolidate_path / "params.json"
    if params_path.exists():
        train_cfg = OmegaConf.load(params_path)
        # Override model args from training config
        if hasattr(train_cfg, "model"):
            cfg.model = dataclass_from_dict(ModelArgs, train_cfg.model, strict=False)

    # Build model
    if cfg.model.model_type == "encdec":
        model = EncoderDecoder(cfg.model)
    else:
        model = Decoder(cfg.model)

    # Load weights
    ckpt_file = consolidate_path / "consolidated.pth"
    if ckpt_file.exists():
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        elif "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

    model = model.to(cfg.eval.device).eval()

    # Build tokenizers from model args
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_name)
    decoder_name = cfg.model.decoder_name or cfg.default_decoder_tokenizer
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    return model, encoder_tokenizer, decoder_tokenizer


# ==================== Generation ====================


@torch.inference_mode()
def generate_predictions(
    model: torch.nn.Module,
    examples: List[ContextBasedExample],
    encoder_tokenizer,
    decoder_tokenizer,
    cfg: EvalConfig,
) -> List[str]:
    """
    Generate predictions for a batch of examples.

    Returns list of generated strings.
    """
    device = torch.device(cfg.eval.device)
    gen_args = cfg.generation

    # Tokenize inputs
    # For encoder: tokenize documents
    # For decoder: tokenize query as prompt

    predictions = []

    for example in examples:
        # Encode documents
        doc_texts = example.documents
        doc_tokens = [
            encoder_tokenizer.encode(doc, truncation=True, max_length=cfg.model.encoder_max_len)
            for doc in doc_texts
        ]

        # Encode query as decoder prompt
        query_tokens = decoder_tokenizer.encode(
            example.query,
            truncation=True,
            max_length=cfg.model.decoder_max_len // 2,
        )

        # Create minimal batch for single example
        # This is a simplified version - full implementation would use TokenizedBatch
        encoder_input = torch.tensor(
            [t for doc in doc_tokens for t in doc], device=device
        ).unsqueeze(0)
        decoder_input = torch.tensor(query_tokens, device=device).unsqueeze(0)

        # Generate
        if hasattr(model, "generate"):
            output_ids = model.generate(
                encoder_input=encoder_input,
                decoder_input=decoder_input,
                max_new_tokens=gen_args.max_new_tokens,
                temperature=gen_args.temperature,
                top_p=gen_args.top_p,
                do_sample=gen_args.do_sample,
            )
            # Decode output
            pred_text = decoder_tokenizer.decode(
                output_ids[0, len(query_tokens):],
                skip_special_tokens=True,
            )
        else:
            # Fallback: use forward pass (for models without generate)
            pred_text = "[generation not implemented]"

        predictions.append(pred_text)

    return predictions


# ==================== Metrics ====================


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    f1_scores = []
    em_scores = []

    for pred, ref in zip(predictions, references):
        f1_scores.append(compute_f1(pred, ref))
        em_scores.append(compute_exact_match(pred, ref))

    return {
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "num_samples": len(predictions),
    }


# ==================== Evaluation ====================


def evaluate_task(
    model: torch.nn.Module,
    task: BaseTask,
    split: str,
    encoder_tokenizer,
    decoder_tokenizer,
    cfg: EvalConfig,
) -> Dict[str, float]:
    """Evaluate model on a single task."""
    # Initialize task state
    state = task.init_state(split)

    all_predictions = []
    all_references = []

    num_samples = 0
    max_samples = cfg.eval.max_samples or float("inf")

    pbar = tqdm(desc=f"Evaluating {task.__class__.__name__}")

    while not task.is_exhausted(state) and num_samples < max_samples:
        # Read batch of examples
        examples, state = task.read(state, cfg.eval.batch_size)

        if not examples:
            break

        # Generate predictions
        predictions = generate_predictions(
            model,
            examples,
            encoder_tokenizer,
            decoder_tokenizer,
            cfg,
        )

        # Collect predictions and references
        for example, pred in zip(examples, predictions):
            all_predictions.append(pred)
            # Use first answer as reference (for multi-answer, would need to handle differently)
            if example.answers:
                all_references.append(example.answers[0])
            else:
                all_references.append("")

        num_samples += len(examples)
        pbar.update(len(examples))
        pbar.set_postfix({"samples": num_samples})

    pbar.close()

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)

    # Add task-specific metrics if available
    if hasattr(task, "compute_metrics"):
        task_metrics = task.compute_metrics(all_predictions, all_references)
        metrics.update(task_metrics)

    return metrics


def launch_eval(cfg: EvalConfig):
    """Main evaluation function."""
    # Setup distributed if needed
    if not torch.distributed.is_initialized():
        try:
            setup_torch_distributed(DistributedArgs())
        except Exception:
            pass  # Single GPU mode

    # Create output directory
    if cfg.eval.dump_dir:
        Path(cfg.eval.dump_dir).mkdir(parents=True, exist_ok=True)
        dump_config(cfg, Path(cfg.eval.dump_dir) / "config.yaml", log_config=False)

    # Load model
    logger.info("Loading model...")
    model, encoder_tokenizer, decoder_tokenizer = load_model_and_tokenizers(cfg)
    logger.info("Model loaded")

    # Run evaluation on each task
    all_results = {}

    for task_name in cfg.eval.tasks:
        logger.info(f"Evaluating on {task_name}...")

        try:
            task = get_task(task_name)
        except KeyError:
            logger.warning(f"Task {task_name} not found, skipping")
            continue

        metrics = evaluate_task(
            model,
            task,
            cfg.eval.split,
            encoder_tokenizer,
            decoder_tokenizer,
            cfg,
        )

        all_results[task_name] = metrics
        logger.info(f"{task_name}: {metrics}")

    # Save results
    if cfg.eval.dump_dir and get_global_rank() == 0:
        results_path = Path(cfg.eval.dump_dir) / "results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    # Log to metrics file
    if cfg.eval.metric_log_dir and get_global_rank() == 0:
        metric_log_path = Path(cfg.eval.metric_log_dir) / "metrics.eval.jsonl"
        timestamp = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if cfg.eval.global_step is not None:
            timestamp["global_step"] = cfg.eval.global_step

        log_entry = timestamp | {"results": all_results}
        with open(metric_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return all_results


def main():
    """
    CLI uses OmegaConf for config loading with overrides.

    Usage:
        python -m apps.finesearch.eval config=configs/finesearch/eval.yaml
        python -m apps.finesearch.eval config=eval.yaml eval.tasks=[squad,hotpotqa]
    """
    cli_args = OmegaConf.from_cli()

    if not hasattr(cli_args, "config"):
        print("Usage: python -m apps.finesearch.eval config=<config.yaml> [overrides]")
        print(f"\nAvailable tasks: {list_tasks()}")
        sys.exit(1)

    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalConfig())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    results = launch_eval(cfg)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for task_name, metrics in results.items():
        print(f"\n{task_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")


if __name__ == "__main__":
    main()
