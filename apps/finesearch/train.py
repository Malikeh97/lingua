"""
Training script for FineSearch encoder-decoder.

Usage:
    python -m apps.finesearch.train config=configs/finesearch/train.yaml

Uses OmegaConf for YAML config with CLI overrides (same pattern as apps/main).
"""

import gc
import logging
import os
import sys
import types
import dataclasses
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Type, TypeVar, get_type_hints

import yaml
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
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

from transformers import AutoTokenizer
import ray

from addons.models.config import ModelArgs
from addons.models.encoder_decoder import EncoderDecoder
from addons.models.decoder import Decoder
from addons.data.collate import TokenizedBatch, PackedSequences
from addons.data.ray_pipeline import PipelineConfig, create_pipeline_from_names
from addons.trainer import Trainer, TrainerArgs

logger = logging.getLogger()


# ==================== Configuration ====================


@dataclass
class DataArgs:
    """Data pipeline configuration."""

    tasks: List[str] = field(default_factory=lambda: ["squad"])
    weights: List[float] = field(default_factory=lambda: [1.0])
    target_tokens: int = 65536
    batch_size: int = 32  # Examples per reader fetch
    packer_buffer_size: int = 50
    seed: int = 42

    # Fallback decoder tokenizer (used when model.decoder_name is empty)
    default_decoder_tokenizer: str = "meta-llama/Llama-3.2-1B"


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

    # Evaluation during training
    eval: Optional[Any] = None


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


def build_tokenizers(model_args: ModelArgs, data_args: DataArgs):
    """Build encoder and decoder tokenizers from model args."""
    # Encoder tokenizer from model.encoder_name
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.encoder_name)

    # Decoder tokenizer: use pretrained decoder's tokenizer if available,
    # otherwise reuse encoder tokenizer (from-scratch decoder shares encoder embeddings)
    if model_args.decoder_name:
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_args.decoder_name)
    else:
        decoder_tokenizer = encoder_tokenizer

    return encoder_tokenizer, decoder_tokenizer


# ==================== Data Pipeline ====================


def build_data_pipeline(args: DataArgs, split: str = "train"):
    """Build Ray data pipeline."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    config = PipelineConfig(
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        packer_buffer_size=args.packer_buffer_size,
        seed=args.seed,
    )

    pipeline = create_pipeline_from_names(
        task_names=args.tasks,
        weights=args.weights,
        config=config,
        split=split,
    )

    return pipeline


class DataIterator:
    """Wraps Ray pipeline for training iteration."""

    def __init__(
        self,
        pipeline,
        encoder_tokenizer,
        decoder_tokenizer,
        encoder_max_len: int,
        decoder_max_len: int,
        device: torch.device,
    ):
        self.pipeline = pipeline
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.device = device

    def __iter__(self):
        return self

    def __next__(self) -> TokenizedBatch:
        batch = ray.get(self.pipeline.get_batch.remote())
        if batch is None:
            raise StopIteration

        # Tokenize batch
        tokenized = TokenizedBatch.from_batched_examples(
            batch,
            self.encoder_tokenizer,
            self.decoder_tokenizer,
            self.encoder_max_len,
            self.decoder_max_len,
            self.device,
        )
        return tokenized


# ==================== Training Loop ====================


def validate_config(cfg: TrainConfig):
    """Validate and adjust configuration."""
    assert cfg.dump_dir, "dump_dir must be set"

    if cfg.checkpoint.path is None:
        cfg.checkpoint.path = str(Path(cfg.dump_dir) / "checkpoints")

    # Validate task/weight lengths
    assert len(cfg.data.tasks) == len(cfg.data.weights), (
        f"tasks ({len(cfg.data.tasks)}) and weights ({len(cfg.data.weights)}) must match"
    )


def train(cfg: TrainConfig):
    """Main training function."""
    with ExitStack() as context_stack:
        validate_config(cfg)

        # Setup distributed
        if get_is_master():
            os.makedirs(cfg.dump_dir, exist_ok=True)
            _dump_config_yaml(cfg, Path(cfg.dump_dir) / "config.yaml")

        init_logger(Path(cfg.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(cfg.env)
        setup_torch_distributed(cfg.distributed)

        logger.info(f"Starting job: {cfg.name}")

        # Set seed
        torch.manual_seed(cfg.seed)

        # Build tokenizers
        logger.info("Building tokenizers")
        encoder_tokenizer, decoder_tokenizer = build_tokenizers(cfg.model, cfg.data)

        # Build model
        logger.info("Building model")
        model = build_model(cfg.model)

        # Apply model dtype from distributed config
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        model_dtype = dtype_map.get(cfg.distributed.model_dtype, torch.float32)
        model = model.to(device=cfg.trainer.device, dtype=model_dtype)

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

        # Build data pipeline
        logger.info("Building data pipeline")
        train_pipeline = build_data_pipeline(cfg.data, split="train")
        val_pipeline = None
        if cfg.eval_interval > 0:
            val_pipeline = build_data_pipeline(cfg.data, split="validation")

        # Create data iterators
        device = torch.device(cfg.trainer.device)
        train_iter = DataIterator(
            train_pipeline,
            encoder_tokenizer,
            decoder_tokenizer,
            cfg.model.encoder_max_len,
            cfg.model.decoder_max_len,
            device,
        )

        # Checkpoint manager
        checkpoint = CheckpointManager.instantiate_and_make_dir(cfg.checkpoint)

        # Convert config to OmegaConf-safe dict for checkpointing
        # (Literal type annotations not supported by OmegaConf.structured)
        cfg_dict = OmegaConf.create(dataclasses.asdict(cfg))

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
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator
                train_iter = DataIterator(
                    train_pipeline,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    cfg.model.encoder_max_len,
                    cfg.model.decoder_max_len,
                    device,
                )
                batch = next(train_iter)

            # Forward pass
            optimizer.zero_grad()
            logits = model(batch)

            # Compute loss
            shift_logits = logits[:-1].contiguous()
            shift_labels = batch.labels[1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

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

                    gpu_stats = gpu_memory_monitor.get_peak_stats()

                    metrics = {
                        "step": step,
                        "loss": avg_loss,
                        "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                        "lr": optimizer.param_groups[0]["lr"],
                        "tokens_per_sec": tps,
                        "gpu_mem_pct": gpu_stats.max_active_pct,
                    }

                    if get_is_master():
                        metric_logger.log(metrics)

                    logger.info(
                        f"step: {step:>6}  "
                        f"loss: {avg_loss:.4f}  "
                        f"grad: {metrics['grad_norm']:.2e}  "
                        f"lr: {metrics['lr']:.2e}  "
                        f"tps: {tps:.0f}  "
                        f"mem: {gpu_stats.max_active_pct:.0f}%"
                    )

                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    total_loss = 0.0
                    tokens_since_log = 0
                    time_last_log = timer()
                    gpu_memory_monitor.reset_peak_stats()

                # Checkpointing
                saved = False
                if step % cfg.checkpoint.dump.every == 0:
                    saved = checkpoint.save(model, optimizer, TrainState(step=step), cfg_dict)

                # Evaluation
                if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
                    logger.info("Running evaluation...")
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0

                    val_iter = DataIterator(
                        val_pipeline,
                        encoder_tokenizer,
                        decoder_tokenizer,
                        cfg.model.encoder_max_len,
                        cfg.model.decoder_max_len,
                        device,
                    )

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
                            metric_logger.log({"step": step, "val_loss": val_loss})

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

    gc.collect()


T = TypeVar("T")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dataclass_defaults(cls) -> dict:
    """Extract defaults from a dataclass, recursing into nested dataclasses."""
    result = {}
    for f in dataclasses.fields(cls):
        if dataclasses.is_dataclass(f.type):
            result[f.name] = _dataclass_defaults(f.type)
        elif f.default is not dataclasses.MISSING:
            result[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            result[f.name] = f.default_factory()
    return result


def _dict_to_dataclass(cls: Type[T], data: dict) -> T:
    """Convert a dict to a dataclass, recursing into nested dataclasses."""
    kwargs = {}
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    for k, v in data.items():
        ft = field_types.get(k)
        if ft and dataclasses.is_dataclass(ft) and isinstance(v, dict):
            kwargs[k] = _dict_to_dataclass(ft, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _dump_config_yaml(cfg, path):
    """Dump dataclass config to YAML without OmegaConf.structured()."""
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    yaml_str = yaml.dump(d, default_flow_style=False, sort_keys=False)
    logger.info("Using the following config for this run:")
    logger.info(yaml_str)
    with open(path, "w") as f:
        f.write(yaml_str)


def main():
    """
    CLI uses OmegaConf only for YAML loading and CLI parsing (no structured()).

    Usage:
        python -m apps.finesearch.train config=apps/finesearch/configs/debug.yaml
        python -m apps.finesearch.train config=debug.yaml model.encoder_name=bert-base
    """
    cli_args = OmegaConf.from_cli()

    if not hasattr(cli_args, "config"):
        print("Usage: python -m apps.finesearch.train config=<config.yaml> [overrides]")
        sys.exit(1)

    file_cfg = OmegaConf.to_container(OmegaConf.load(cli_args.config), resolve=True)
    del cli_args.config
    cli_overrides = OmegaConf.to_container(cli_args, resolve=True)

    defaults = _dataclass_defaults(TrainConfig)
    merged = _deep_merge(defaults, file_cfg)
    merged = _deep_merge(merged, cli_overrides)

    cfg = _dict_to_dataclass(TrainConfig, merged)
    train(cfg)


if __name__ == "__main__":
    main()
