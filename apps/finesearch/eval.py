"""
Evaluation script for FineSearch encoder-decoder.

Usage:
    python -m apps.finesearch.eval config=apps/finesearch/configs/debug.yaml \
        eval.ckpt_dir=outputs/finesearch_debug/checkpoints/step_2000

Supports:
- Task-specific metrics (F1, EM, ROUGE, etc.)
- Multiple tasks in single run
- Checkpoint loading with config inference
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from lingua.checkpoint import consolidate_checkpoints
from lingua.distributed import (
    get_global_rank,
    setup_torch_distributed,
)

from addons.distributed import DistributedArgs

from transformers import AutoTokenizer

from addons.models.config import ModelArgs, GenerationArgs
from addons.models.encoder_decoder import EncoderDecoder
from addons.models.decoder import Decoder
from addons.data.collate import TokenizedBatch
from addons.tasks.registry import get_task, list_tasks
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample

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

    consolidate_path = consolidate_checkpoints(str(ckpt_path))

    # Load training config to infer model args
    params_path = consolidate_path / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            train_params = json.load(f)
        if "model" in train_params and isinstance(train_params["model"], dict):
            from apps.finesearch.config_utils import dict_to_dataclass
            cfg.model = dict_to_dataclass(ModelArgs, train_params["model"])

    # Build model
    if cfg.model.model_type == "encdec":
        model = EncoderDecoder(cfg.model)
    else:
        model = Decoder(cfg.model)

    # Load weights
    ckpt_file = consolidate_path / "consolidated.pth"
    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    elif "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model = model.to(device=cfg.eval.device, dtype=torch.bfloat16).eval()

    # Build tokenizers
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_name)
    if cfg.model.decoder_name:
        decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.model.decoder_name)
    else:
        decoder_tokenizer = encoder_tokenizer

    return model, encoder_tokenizer, decoder_tokenizer


# ==================== Generation ====================


@torch.inference_mode()
def generate_predictions(
    model: torch.nn.Module,
    batch: BatchedContextBasedExamples,
    encoder_tokenizer,
    decoder_tokenizer,
    cfg: EvalConfig,
) -> List[str]:
    """
    Generate predictions for a batch of examples.

    Tokenizes into TokenizedBatch, runs model.generate(), decodes output.
    """
    device = torch.device(cfg.eval.device)

    tokenized = TokenizedBatch.from_batched_examples(
        batch,
        encoder_tokenizer,
        decoder_tokenizer,
        cfg.model.encoder_max_len,
        cfg.model.decoder_max_len,
        device,
    ).prompt_only()

    eos_id = getattr(decoder_tokenizer, "sep_token_id", None) or getattr(decoder_tokenizer, "eos_token_id", None)

    if hasattr(model, "generate"):
        output_ids = model.generate(tokenized, cfg.generation, eos_token_id=eos_id)
        # output_ids: [1, generated_len] (prompt already stripped)
        predictions = []
        for i in range(output_ids.shape[0]):
            pred = decoder_tokenizer.decode(
                output_ids[i], skip_special_tokens=True
            )
            predictions.append(pred)
    else:
        # Fallback: use forward pass to get greedy predictions
        logits = model(tokenized)
        pred_ids = logits.argmax(dim=-1)

        # Split by cu_seqlens
        cu = tokenized.decoder_tokens.cu_seqlens
        predictions = []
        for i in range(tokenized.batch_size):
            start = cu[i].item()
            end = cu[i + 1].item()
            pred = decoder_tokenizer.decode(
                pred_ids[start:end], skip_special_tokens=True
            )
            predictions.append(pred)

    return predictions


# ==================== Evaluation ====================


def evaluate_task(
    model: torch.nn.Module,
    task_cls: type,
    split: str,
    encoder_tokenizer,
    decoder_tokenizer,
    cfg: EvalConfig,
) -> Dict[str, float]:
    """Evaluate model on a single task."""
    state = task_cls.init_state(split, shuffle=False)

    all_predictions: List[str] = []
    all_miscs: List[Dict[str, Any]] = []

    num_samples = 0
    max_samples = cfg.eval.max_samples or float("inf")

    pbar = tqdm(desc=f"Evaluating {task_cls.__name__}")

    while not task_cls.is_exhausted(state) and num_samples < max_samples:
        examples, state = task_cls.read(state, cfg.eval.batch_size)

        if not examples:
            break

        # Flatten to individual examples for misc collection
        flat_examples: List[ContextBasedExample] = []
        for unit in examples:
            if isinstance(unit, ContextBasedExample):
                flat_examples.append(unit)
            else:
                flat_examples.extend(list(unit))

        # Generate one example at a time (generate() doesn't support packed batches)
        predictions = []
        for ex in flat_examples:
            single_batch = BatchedContextBasedExamples.from_examples([ex])
            preds = generate_predictions(
                model, single_batch, encoder_tokenizer, decoder_tokenizer, cfg
            )
            predictions.extend(preds)

        all_predictions.extend(predictions)
        all_miscs.extend(ex.misc for ex in flat_examples)

        num_samples += len(flat_examples)
        pbar.update(len(flat_examples))
        pbar.set_postfix({"samples": num_samples})

    pbar.close()

    # Print first 5 predictions for debugging
    for i in range(min(5, len(all_predictions))):
        gold = all_miscs[i].get("all_answers", ["?"])
        print(f"[{i}] pred: {all_predictions[i]!r}")
        print(f"     gold: {gold}")

    if not all_predictions:
        return {}

    # Use task's evaluate() which knows how to interpret misc
    metrics = task_cls.evaluate(all_predictions, all_miscs)
    metrics["num_samples"] = len(all_predictions)
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

    # Load model
    logger.info("Loading model...")
    model, encoder_tokenizer, decoder_tokenizer = load_model_and_tokenizers(cfg)
    logger.info("Model loaded")

    # Import all task modules so they register
    import addons.tasks.squad  # noqa: F401
    import addons.tasks.hotpotqa  # noqa: F401

    # Run evaluation on each task
    all_results = {}

    for task_name in cfg.eval.tasks:
        logger.info(f"Evaluating on {task_name}...")

        try:
            task = get_task(task_name)
            task_cls = task.__class__
        except ValueError:
            logger.warning(f"Task {task_name} not found, skipping. Available: {list_tasks()}")
            continue

        metrics = evaluate_task(
            model,
            task_cls,
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
        log_entry: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "results": all_results,
        }
        if cfg.eval.global_step is not None:
            log_entry["global_step"] = cfg.eval.global_step

        with open(metric_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return all_results


def main():
    """
    Usage:
        python -m apps.finesearch.eval config=apps/finesearch/configs/debug.yaml \
            eval.ckpt_dir=outputs/finesearch_debug/checkpoints/step_2000
        python -m apps.finesearch.eval config=debug.yaml eval.tasks='[squad,hotpotqa]'
    """
    from apps.finesearch.config_utils import load_config

    cfg = load_config(EvalConfig)

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
