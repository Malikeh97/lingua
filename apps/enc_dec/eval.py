# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from lingua.args import dump_config
from lingua.checkpoint import consolidate_checkpoints, load_from_checkpoint
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    get_is_master,
    get_world_size,
    setup_env,
    setup_torch_distributed,
    get_device_mesh,
)
from lingua.logger import init_logger
from lingua.metrics import LoggingArgs, MetricLogger, WandbArgs, get_num_params
from lingua.tokenizer import build_tokenizer

from apps.enc_dec.enc_dec import (
    EncDecTransformerArgs,
    EncDecTransformer,
)
from apps.enc_dec.data import (
    EncDecDataArgs,
    build_qa_dataloader,
)

logger = logging.getLogger()

EVAL_FOLDER_NAME = "eval_{:010d}"


@dataclass
class ValidationArgs:
    """Validation configuration."""
    max_steps: Optional[int] = None  # Limit number of validation steps


@dataclass
class GenerationArgs:
    """Generation configuration."""
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 1.0
    top_k: int = 0


@dataclass
class EncDecEvalArgs:
    """Evaluation arguments for encoder-decoder model."""

    dump_dir: str = ""
    ckpt_dir: str = ""
    metric_log_dir: Optional[str] = None

    # Data configuration
    data: EncDecDataArgs = field(default_factory=EncDecDataArgs)

    # Model configuration
    model: EncDecTransformerArgs = field(default_factory=EncDecTransformerArgs)

    # Distributed configuration
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    # Validation configuration
    validation: ValidationArgs = field(default_factory=ValidationArgs)

    # Generation configuration
    generation: GenerationArgs = field(default_factory=GenerationArgs)

    # Logging
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    wandb: Optional[WandbArgs] = None

    # Metadata
    global_step: Optional[int] = None


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score based on token overlap."""
    pred_tokens = prediction.strip().lower().split()
    gt_tokens = ground_truth.strip().lower().split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    num_common = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


@torch.no_grad()
def generate_answer(
    model: EncDecTransformer,
    encoder_input_ids: torch.Tensor,
    decoder_prefix_ids: torch.Tensor,
    encoder_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """Generate answer tokens autoregressively.

    Args:
        model: Encoder-decoder model
        encoder_input_ids: [B, enc_seq] encoder input tokens
        decoder_prefix_ids: [B, prefix_len] decoder prefix (question tokens)
        encoder_mask: [B, enc_seq] encoder padding mask
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        eos_token_id: End of sequence token ID

    Returns:
        Generated token IDs [B, generated_len]
    """
    model.eval()
    bsz = encoder_input_ids.shape[0]
    device = encoder_input_ids.device

    # Encode the document once
    encoder_output = model.encoder(
        encoder_input_ids,
        padding_mask=encoder_mask,
        attn_impl="sdpa",
    )

    # Start with decoder prefix
    decoder_input = decoder_prefix_ids.clone()
    generated_tokens = []

    for _ in range(max_new_tokens):
        # Get next token prediction
        logits = model.decoder(
            decoder_input,
            encoder_output,
            encoder_mask=encoder_mask,
            target=None,
            attn_impl="sdpa",
        )

        # Get logits for last position
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = (
                next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Sample or take argmax
        if temperature > 0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_tokens.append(next_token)

        # Check for EOS
        if (next_token == eos_token_id).all():
            break

        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    else:
        return torch.empty(bsz, 0, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_validation(
    model: EncDecTransformer,
    dataloader,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on validation set (loss only)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_steps = 0

    for batch in dataloader:
        encoder_input_ids = batch["encoder_input_ids"].cuda()
        decoder_input_ids = batch["decoder_input_ids"].cuda()
        labels = batch["labels"].cuda()
        encoder_mask = batch["encoder_padding_mask"].cuda()

        loss = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_target=labels,
            encoder_padding_mask=encoder_mask,
        )

        # Count non-masked tokens
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        num_steps += 1

        if max_steps is not None and num_steps >= max_steps:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_tokens": total_tokens,
        "val_steps": num_steps,
    }


def launch_eval(args: EncDecEvalArgs):
    """Launch evaluation."""
    # Setup
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        dump_config(args, Path(args.dump_dir) / "config.yaml")

    init_logger(Path(args.dump_dir) / "eval.log" if args.dump_dir else None)
    setup_env(args.env)
    setup_torch_distributed(args.distributed)

    world_mesh = get_device_mesh(args.distributed)
    dp_mesh = world_mesh["dp_replicate"]
    dp_rank = dp_mesh.get_local_rank()
    dp_degree = dp_mesh.size()

    if args.distributed.dp_shard > 1:
        dp_rank = (
            dp_rank * world_mesh["dp_shard"].size()
            + world_mesh["dp_shard"].get_local_rank()
        )
        dp_degree *= world_mesh["dp_shard"].size()

    # Build tokenizer
    tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
    args.model.vocab_size = tokenizer.n_words

    # Set sequence lengths
    args.model.max_encoder_seqlen = args.data.max_encoder_len
    args.model.max_decoder_seqlen = args.data.max_decoder_len

    # Consolidate checkpoint if needed
    if args.ckpt_dir:
        consolidate_checkpoints(args.ckpt_dir)

    # Build model
    logger.info("Building model...")
    model = EncDecTransformer(args.model)
    model = model.cuda()

    # Load checkpoint
    if args.ckpt_dir:
        logger.info(f"Loading checkpoint from {args.ckpt_dir}")
        load_from_checkpoint(args.ckpt_dir, model, model_key="model")

    model.eval()

    # Build validation dataloader
    val_dataloader = build_qa_dataloader(
        args.data,
        dp_rank,
        dp_degree,
        split="validation",
    )

    # Run validation
    logger.info("Running validation...")
    val_metrics = evaluate_validation(
        model,
        val_dataloader,
        max_steps=args.validation.max_steps,
    )

    # Log results
    if get_is_master():
        logger.info(f"Validation results: {val_metrics}")

        if args.dump_dir:
            results_path = Path(args.dump_dir) / "results.json"
            with open(results_path, "w") as f:
                json.dump(val_metrics, f, indent=2)
            logger.info(f"Results saved to {results_path}")

        # Log to metrics file if specified
        if args.metric_log_dir:
            metrics_path = Path(args.metric_log_dir) / "metrics.jsonl"
            with open(metrics_path, "a") as f:
                metrics = {
                    "global_step": args.global_step,
                    **{f"eval/{k}": v for k, v in val_metrics.items()},
                }
                f.write(json.dumps(metrics) + "\n")

    return val_metrics


def main():
    """Main entry point."""
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(EncDecEvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    launch_eval(cfg)


if __name__ == "__main__":
    main()
