"""
Training script for FineSearch encoder-decoder.

Usage:
    python -m apps.finesearch.train config=configs/finesearch/train.yaml

Uses YAML config with CLI overrides.
"""

import gc
import logging
import os
import sys
import dataclasses
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from lingua.checkpoint import CheckpointArgs, CheckpointManager
from lingua.distributed import (
    get_is_master,
    get_world_size,
    init_signal_handler,
    setup_env,
    setup_torch_distributed,
    requeue_slurm_job,
    EnvironmentArgs,
)

from addons.distributed import DistributedArgs
from lingua.logger import init_logger
from lingua.metrics import GPUMemoryMonitor, LoggingArgs, MetricLogger, get_num_params
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler


import ray

from addons.models.config import ModelArgs
from addons.models.encoder_decoder import EncoderDecoder
from addons.models.decoder import Decoder
from addons.data.collate import TokenizedBatch
from addons.data.ray_pipeline import PipelineConfig, create_pipeline
from addons.trainer import Trainer, TrainerArgs

logger = logging.getLogger()


# ==================== Configuration ====================


from addons.data.config import DataArgs


@dataclass
class TrainConfig:
    """Full training configuration."""

    name: str = "finesearch"
    dump_dir: str = ""
    seed: int = 42

    # Training
    steps: int = 10000
    grad_acc_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 1000
    gc_collect_freq: int = 1000

    # Components
    model: ModelArgs = field(default_factory=ModelArgs)
    data: DataArgs = field(default_factory=DataArgs)
    trainer: TrainerArgs = field(default_factory=TrainerArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # Validation during training
    run_val: bool = True


# ==================== Signal Handler ====================


preemption_flag = dict(flag=False)


class TrainState:
    """Simple train state compatible with CheckpointManager."""

    def __init__(self, step: int = 0):
        self.step = step

    def state_dict(self):
        return {"step": self.step}


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption! Checkpointing and exiting.")
    preemption_flag["flag"] = True


# ==================== Model Building ====================


def build_model(args: ModelArgs) -> torch.nn.Module:
    """Build encoder-decoder or decoder-only model."""
    if args.model_type == "encdec":
        model = EncoderDecoder(args)
    else:
        model = Decoder(args)

    return model


# ==================== Data Pipeline ====================


def build_val_pipeline(args: DataArgs, tokenizer_factory):
    """Build validation pipeline using only tasks that have a validation split."""
    from addons.tasks.registry import get_task

    val_sources = []
    for source in args.sources:
        source = dict(source)
        name = source["task"]
        task_cls = get_task(name).__class__
        try:
            data = task_cls.prepare_data("validation", **{k: v for k, v in source.items() if k not in ("task", "weight")})
            del data
            val_sources.append(source)
        except Exception:
            logger.info(f"Task '{name}' has no validation split, skipping for validation")

    if not val_sources:
        return None

    val_args = DataArgs(
        sources=val_sources,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        packer_buffer_size=args.packer_buffer_size,
        seed=args.seed,
        doc_max_tokens=args.doc_max_tokens,
        target_max_tokens=args.target_max_tokens,
        enc_token_cost=args.enc_token_cost,
        dec_token_cost=args.dec_token_cost,
    )
    return build_data_pipeline(val_args, split="validation", tokenizer_factory=tokenizer_factory)


def build_data_pipeline(args: DataArgs, split: str = "train", tokenizer_factory=None):
    """Build Ray data pipeline."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    config = PipelineConfig(
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        packer_buffer_size=args.packer_buffer_size,
        seed=args.seed,
        enc_token_cost=args.enc_token_cost,
        dec_token_cost=args.dec_token_cost,
    )

    # Build task configs and weights from sources
    task_configs = {}
    weight_dict = {}
    for source in args.sources:
        source = dict(source)  # copy
        name = source.pop("task")
        weight = source.pop("weight", 1.0)
        task_configs[name] = {"task_name": name, **source}
        weight_dict[name] = weight

    pipeline = create_pipeline(
        task_configs=task_configs,
        weights=weight_dict,
        config=config,
        split=split,
        tokenizer_factory=tokenizer_factory,
        doc_max_tokens=args.doc_max_tokens,
        target_max_tokens=args.target_max_tokens,
    )

    return pipeline


class DataIterator:
    """Wraps Ray pipeline for training iteration.

    Pipeline returns TokenizedBatch on CPU. Iterator moves it to device.
    """

    def __init__(self, pipeline, device: torch.device):
        self.pipeline = pipeline
        self.device = device

    def __iter__(self):
        return self

    def __next__(self) -> TokenizedBatch:
        batch = ray.get(self.pipeline.get_batch.remote())
        if batch is None:
            raise StopIteration
        return batch.to(self.device)


# ==================== Training Loop ====================


def validate_config(cfg: TrainConfig):
    """Validate and adjust configuration."""
    assert cfg.dump_dir, "dump_dir must be set"

    if cfg.checkpoint.path is None:
        cfg.checkpoint.path = str(Path(cfg.dump_dir) / "checkpoints")

    # Validate sources
    assert cfg.data.sources, "data.sources must not be empty"
    for i, src in enumerate(cfg.data.sources):
        assert "task" in src, f"data.sources[{i}] must have 'task' key"


def train(cfg: TrainConfig):
    """Main training function."""
    with ExitStack() as context_stack:
        validate_config(cfg)

        # Setup distributed
        if get_is_master():
            os.makedirs(cfg.dump_dir, exist_ok=True)
            from apps.finesearch.config_utils import dump_yaml
            dump_yaml(cfg, Path(cfg.dump_dir) / "config.yaml")

        init_logger(Path(cfg.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(cfg.env)
        setup_torch_distributed(cfg.distributed)

        logger.info(f"Starting job: {cfg.name}")

        # Set seed
        torch.manual_seed(cfg.seed)

        # Build tokenizer factory (model owns the special token logic)
        logger.info("Building tokenizers")
        model_cls = EncoderDecoder if cfg.model.model_type == "encdec" else Decoder
        tokenizer_factory = model_cls.tokenizer_factory(cfg.model)

        # Build model
        logger.info("Building model")
        model = build_model(cfg.model)

        # Apply model dtype from distributed config
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        model_dtype = dtype_map.get(cfg.distributed.model_dtype, torch.float32)
        model = model.to(device=cfg.trainer.device, dtype=model_dtype)

        # Activation checkpointing
        if cfg.model.activation_checkpointing:
            from addons.models.encoder_decoder import apply_activation_checkpointing
            logger.info("Enabling activation checkpointing on decoder layers")
            apply_activation_checkpointing(model)

        model_param_count = get_num_params(model)
        logger.info(f"Model size: {model_param_count:,} parameters")

        # GPU monitoring
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU: {gpu_memory_monitor.device_name} "
            f"({gpu_memory_monitor.device_capacity_gib:.2f} GiB)"
        )

        # Build optimizer
        optimizer, scheduler = build_optimizer(model, cfg.optim, cfg.steps)

        # Sequence parallelism setup
        from addons.distributed import get_sp_group, get_dp_group
        sp_size = cfg.distributed.sp_size
        sp_group = get_sp_group(sp_size)
        dp_group = get_dp_group(sp_size)  # for FSDP process_group when sp_size > 1
        sp_rank = dist.get_rank(sp_group) if sp_group else 0

        if sp_group:
            logger.info(f"Sequence parallelism: sp_size={sp_size}, sp_rank={sp_rank}, "
                        f"dp_group_size={dist.get_world_size(dp_group)}")

        # Build data pipeline — only sp_rank 0 in each sp_group fetches data
        device = torch.device(cfg.trainer.device)
        train_pipeline = None
        train_iter = None
        if sp_rank == 0:
            logger.info("Building data pipeline")
            train_pipeline = build_data_pipeline(cfg.data, split="train", tokenizer_factory=tokenizer_factory)
            train_iter = DataIterator(train_pipeline, device)

        val_pipeline = None
        if cfg.run_val and cfg.eval_interval > 0 and sp_rank == 0:
            val_pipeline = build_val_pipeline(cfg.data, tokenizer_factory=tokenizer_factory)
            if val_pipeline is None:
                logger.warning("No tasks with validation split found, disabling validation")

        # Checkpoint manager
        checkpoint = CheckpointManager.instantiate_and_make_dir(cfg.checkpoint)

        cfg_dict = dataclasses.asdict(cfg)

        # Metric logger
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(cfg.dump_dir) / "metrics.jsonl", cfg)
        )

        # Profiler
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(cfg.dump_dir, model, cfg.profiling)
        )

        gc.disable()

        # Training state
        step = 0
        acc_step = 0
        total_loss = 0.0
        time_last_log = timer()
        tokens_since_log = 0
        total_examples = 0
        total_enc_tokens = 0
        total_dec_tokens = 0

        model.train()
        logger.info("Starting training")

        pbar = tqdm(total=cfg.steps, desc="Training")

        while step < cfg.steps:
            acc_step += 1
            acc_step = acc_step % cfg.grad_acc_steps

            # Garbage collection
            if step > 0 and step % cfg.gc_collect_freq == 0:
                gc.collect()

            # Get batch
            if sp_rank == 0:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = DataIterator(train_pipeline, device)
                    batch = next(train_iter)
            else:
                batch = None

            # Distribute batch across sp_group
            if sp_group is not None:
                from addons.data.collate import broadcast_batch, shard_batch
                batch = broadcast_batch(batch, sp_group)
                batch = shard_batch(batch, sp_rank, sp_size)

            # Track data stats
            total_examples += batch.batch_size
            total_enc_tokens += batch.encoder_tokens.tokens.numel()
            total_dec_tokens += batch.decoder_tokens.tokens.numel()

            # Forward pass
            optimizer.zero_grad()
            logits = model(batch, sp_group=sp_group)

            # Compute loss
            shift_logits = logits[:-1].contiguous()
            shift_labels = batch.labels[1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # All-reduce loss across sp_group (each rank has a shard)
            if sp_group is not None:
                dist.all_reduce(loss, group=sp_group)
                loss = loss / sp_size

            # Scale loss for gradient accumulation
            scaled_loss = loss / cfg.grad_acc_steps
            scaled_loss.backward()

            total_loss += loss.item()
            tokens_since_log += batch.decoder_tokens.tokens.numel()

            # Optimizer step
            if acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip
                )
                optimizer.step()
                scheduler.step()
                step += 1
                pbar.update(1)

                # Logging
                if step % cfg.log_interval == 0:
                    time_delta = timer() - time_last_log
                    avg_loss = total_loss / cfg.log_interval
                    tps = tokens_since_log / time_delta

                    grad_norm_val = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

                    metrics = {
                        "global_step": step,
                        "loss": avg_loss,
                        "grad_norm": grad_norm_val,
                        "lr": optimizer.param_groups[0]["lr"],
                        "tokens_per_sec": tps,
                        "total_examples": total_examples,
                        "total_enc_tokens": total_enc_tokens,
                        "total_dec_tokens": total_dec_tokens,
                        "total_tokens": total_enc_tokens + total_dec_tokens,
                    }

                    # Pipeline stats (per-source example counts)
                    pipeline_stats = ray.get(train_pipeline.get_stats.remote())
                    metrics["packer_buffer_size"] = pipeline_stats["buffer_size"]
                    for source, count in pipeline_stats["source_counts"].items():
                        metrics[f"examples/{source}"] = count

                    if get_is_master():
                        metric_logger.log(metrics)

                    logger.info(
                        f"step: {step:>6}  "
                        f"loss: {avg_loss:.4f}  "
                        f"lr: {metrics['lr']:.2e}  "
                        f"tps: {tps:.0f}"
                    )

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    total_loss = 0.0
                    tokens_since_log = 0
                    time_last_log = timer()

                # Checkpointing
                saved = False
                if step % cfg.checkpoint.dump.every == 0:
                    saved = checkpoint.save(model, optimizer, TrainState(step=step), cfg_dict)

                # Evaluation
                if val_pipeline is not None and cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
                    logger.info("Running evaluation...")
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0

                    val_iter = DataIterator(val_pipeline, device)

                    with torch.no_grad():
                        for _ in range(100):  # Max 100 val batches
                            try:
                                val_batch = next(val_iter)
                            except StopIteration:
                                break

                            val_logits = model(val_batch)
                            val_shift_logits = val_logits[:-1].contiguous()
                            val_shift_labels = val_batch.labels[1:].contiguous()
                            batch_loss = torch.nn.functional.cross_entropy(
                                val_shift_logits.view(-1, val_shift_logits.size(-1)),
                                val_shift_labels.view(-1),
                                ignore_index=-100,
                            )
                            val_loss += batch_loss.item()
                            val_batches += 1

                    if val_batches > 0:
                        val_loss /= val_batches
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        if get_is_master():
                            metric_logger.log({"global_step": step, "val_loss": val_loss})

                    model.train()

                # Preemption handling
                if preemption_flag["flag"]:
                    if not saved:
                        checkpoint.save(model, optimizer, TrainState(step=step), cfg_dict)
                    requeue_slurm_job()
                    sys.exit(0)

        pbar.close()

        # Final checkpoint
        checkpoint.save(model, optimizer, TrainState(step=step), cfg_dict)
        logger.info(f"Training complete. Final step: {step}")

    if dist.is_initialized():
        dist.barrier()
    if ray.is_initialized():
        ray.shutdown()
    gc.collect()


def main():
    """
    Usage:
        python -m apps.finesearch.train config=apps/finesearch/configs/debug.yaml
        python -m apps.finesearch.train config=debug.yaml model.encoder_name=bert-base
    """
    from apps.finesearch.config_utils import load_config

    cfg = load_config(TrainConfig)
    train(cfg)


if __name__ == "__main__":
    main()
