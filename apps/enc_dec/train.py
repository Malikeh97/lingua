# Copyright (c) Meta Platforms, Inc. and affiliates.

import gc
import logging
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Optional

from omegaconf import OmegaConf
import torch
import torch.distributed
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from lingua.args import dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer

from apps.enc_dec.enc_dec import (
    EncDecTransformerArgs,
    EncDecTransformer,
    EncoderType,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token_enc_dec,
    load_pretrained_decoder_weights,
)
from apps.enc_dec.data import (
    EncDecDataArgs,
    build_infinite_qa_dataloader,
    build_train_val_dataloaders,
    EncDecDataLoaderState,
    init_dataloader_state,
)
from apps.enc_dec.eval import evaluate_validation

import wandb

logger = logging.getLogger()


@dataclass
class EvalArgs:
    """Evaluation configuration for validation during training."""
    every: int = 100  # Evaluate every N optimizer steps
    max_steps: Optional[int] = None  # Limit number of validation steps (None = full val set)


@dataclass
class EncDecTrainArgs:
    """Training arguments for encoder-decoder model."""

    name: str = "enc_dec"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000

    # Number of optimizer steps to take (if None, use max_epochs)
    steps: Optional[int] = None
    # Number of epochs to train (used when steps is None)
    max_epochs: Optional[int] = None

    # Data configuration
    data: EncDecDataArgs = field(default_factory=EncDecDataArgs)

    # Optimizer configuration
    optim: OptimArgs = field(default_factory=OptimArgs)

    # Model configuration
    model: EncDecTransformerArgs = field(default_factory=EncDecTransformerArgs)

    # Distributed training configuration
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    # Checkpointing
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)

    # Profiling
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)

    # Logging
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # Evaluation during training
    eval: EvalArgs = field(default_factory=EvalArgs)


@dataclass
class EncDecTrainState(Stateful):
    """Training state for encoder-decoder model."""

    step: int  # Number of optimizer steps taken
    acc_step: int  # Number of accumulation steps since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: EncDecDataLoaderState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": {
                "epoch": self.data_loader_state.epoch,
                "step_in_epoch": self.data_loader_state.step_in_epoch,
            },
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = EncDecDataLoaderState(
            epoch=state_dict["data_loader_state"]["epoch"],
            step_in_epoch=state_dict["data_loader_state"]["step_in_epoch"],
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: EncDecTrainArgs, vocab_size: int):
    """Validate training arguments."""
    if args.model.vocab_size < 0:
        logger.info(f"Setting model vocab_size to {vocab_size}")
        args.model.vocab_size = vocab_size

    assert args.model.vocab_size == vocab_size, "Vocab size mismatch"
    assert args.dump_dir, "dump_dir must be specified"

    if args.checkpoint.path is None:
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")
        logger.info(f"Setting checkpoint path to {args.checkpoint.path}")

    # Validate distributed settings
    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )

    # Validate steps/epochs configuration
    if args.steps is None and args.max_epochs is None:
        raise ValueError("Either 'steps' or 'max_epochs' must be specified")
    if args.steps is not None and args.max_epochs is not None:
        logger.warning("Both 'steps' and 'max_epochs' specified. 'steps' takes priority.")

    # Set max sequence lengths
    args.model.max_encoder_seqlen = args.data.max_encoder_len
    args.model.max_decoder_seqlen = args.data.max_decoder_len

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption! Checkpointing ASAP and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def should_continue_training(step: int, epoch: int, args: EncDecTrainArgs) -> bool:
    """Check if training should continue based on steps or epochs."""
    if args.steps is not None:
        return step < args.steps
    return epoch < args.max_epochs


def train(args: EncDecTrainArgs):
    """Main training function."""
    with ExitStack() as context_stack:
        # Build tokenizer and validate args
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(args, tokenizer.n_words)

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")

        init_logger(Path(args.dump_dir) / "train.log")
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)

        logger.info(f"Starting job: {args.name}")

        # Get data parallel info
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()

        if args.distributed.dp_shard > 1:
            dp_rank = (
                dp_rank * world_mesh["dp_shard"].size()
                + world_mesh["dp_shard"].get_local_rank()
            )
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank: {dp_rank}")
        logger.info(f"Running on dp size: {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Check encoder type
        encoder_type = EncoderType(args.model.encoder_type)
        logger.info(f"Using encoder type: {encoder_type.value}")

        # Check if using pretrained decoder
        use_pretrained_decoder = (
            args.model.pretrained_decoder.model_name
            and len(args.model.pretrained_decoder.model_name) > 0
        )

        # Initialize model
        # For pretrained encoder or pretrained decoder, we can't use meta device
        if encoder_type == EncoderType.PRETRAINED or use_pretrained_decoder:
            if encoder_type == EncoderType.PRETRAINED:
                logger.info(f"Loading pretrained encoder: {args.model.pretrained_encoder.model_name}")
            # Build model directly - pretrained components load weights from HuggingFace
            model = EncDecTransformer(args.model)

            # Load pretrained decoder weights BEFORE parallelization
            if use_pretrained_decoder:
                logger.info(f"Loading pretrained decoder from: {args.model.pretrained_decoder.model_name}")
                load_pretrained_decoder_weights(
                    decoder=model.decoder,
                    model_name=args.model.pretrained_decoder.model_name,
                    freeze_pretrained=args.model.pretrained_decoder.freeze_pretrained,
                    init_base_std=args.model.decoder.init_base_std,
                    seed=args.model.seed,
                    init_mode=args.model.pretrained_decoder.init_mode,
                )
        else:
            # Initialize on meta device for efficiency
            with torch.device("meta"):
                model = EncDecTransformer(args.model)

        logger.info("Model is built!")

        # Parameter counting
        model_param_count = get_num_params(model)
        encoder_param_count = get_num_params(model.encoder)
        decoder_param_count = get_num_params(model.decoder)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = model_param_count - trainable_params

        logger.info(f"Total parameters: {model_param_count:,}")
        logger.info(f"Encoder parameters: {encoder_param_count:,}")
        logger.info(f"Decoder parameters: {decoder_param_count:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")

        # Parallelize model
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,  # TP not implemented yet for enc-dec
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Initialize weights
        # For pretrained encoder or pretrained decoder, weights are already loaded before parallelization
        if encoder_type == EncoderType.PRETRAINED or use_pretrained_decoder:
            # Move model to cuda (encoder already has weights loaded, decoder may have pretrained weights)
            model = model.cuda()

            if not use_pretrained_decoder:
                # Only initialize decoder if we didn't load pretrained weights
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.model.seed)
                    # Initialize decoder weights
                    model.decoder.init_weights(shared_embeddings=False)

            # Initialize projection layer if it exists (for pretrained encoder)
            if encoder_type == EncoderType.PRETRAINED:
                if hasattr(model.encoder, 'projection') and model.encoder.projection is not None:
                    model.encoder.reset_parameters()
        else:
            model = model.to_empty(device="cuda")

            if args.checkpoint.init_ckpt_path:
                logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
                load_from_checkpoint(
                    args.checkpoint.init_ckpt_path, model, model_key="model"
                )
                model.encoder.rope_embeddings.reset_parameters()
                model.decoder.rope_embeddings.reset_parameters()
            else:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.model.seed)
                    model.init_weights()

        check_model_value_range(model, range=10.0, std=1.0)

        # GPU memory monitoring
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # Build data loaders (train and optionally validation)
        data_loader, val_loader = build_train_val_dataloaders(
            args.data,
            dp_rank,
            dp_degree,
        )

        if val_loader is not None:
            logger.info(f"Validation enabled: evaluating every {args.eval.every} steps")

        # Compute total steps for scheduler
        if args.steps is not None:
            total_steps = args.steps
        else:
            batches_per_epoch = len(data_loader)
            steps_per_epoch = batches_per_epoch // args.grad_acc_steps
            total_steps = steps_per_epoch * args.max_epochs
            logger.info(
                f"Using max_epochs={args.max_epochs}, batches_per_epoch={batches_per_epoch}, "
                f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}"
            )

        # Build optimizer
        optimizer, scheduler = build_optimizer(model, args.optim, total_steps)

        # Initialize data loader state
        data_loader_state = init_dataloader_state()

        train_state = EncDecTrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        # Checkpoint manager
        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        gc.disable()

        # Training loop
        model.train()

        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )

        # Set data loader epoch from checkpoint state
        data_loader.set_epoch(train_state.data_loader_state.epoch)

        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()

        saved = False
        while should_continue_training(train_state.step, data_loader.epoch, args):
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # Get batch
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch = next(data_loader)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("Garbage collection")
                gc.collect()

            # Move batch to GPU
            encoder_input_ids = batch["encoder_input_ids"].cuda()
            decoder_input_ids = batch["decoder_input_ids"].cuda()
            labels = batch["labels"].cuda()
            encoder_padding_mask = batch["encoder_padding_mask"].cuda()

            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += decoder_input_ids.numel()

            bsz, dec_seqlen = decoder_input_ids.shape
            _, enc_seqlen = encoder_input_ids.shape

            # Forward pass
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            loss = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_target=labels,
                encoder_padding_mask=encoder_padding_mask,
            )

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # Scale loss for gradient accumulation
            loss = loss / args.grad_acc_steps
            loss.backward()
            loss = loss.detach() * args.grad_acc_steps

            # Optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )
                grad_norm = (
                    grad_norm.full_tensor()
                    if isinstance(grad_norm, DTensor)
                    else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # Profiler step
            if torch_profiler:
                import xformers.profiler
                xformers.profiler.step()

            # Logging
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = total_acc_steps * args.data.batch_size * dec_seqlen
                total_tokens = dp_degree * tokens_per_gpu

                # FLOPS calculation
                FLOPS = (
                    get_num_flop_per_token_enc_dec(
                        encoder_param_count,
                        decoder_param_count,
                        args.model.encoder.n_layers,
                        args.model.decoder.n_layers,
                        args.model.dim,
                        enc_seqlen,
                        dec_seqlen,
                    )
                    * wps
                )

                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "wps": wps,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {"loss/out": loss.item()}
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()

                epoch_progress = data_loader.get_epoch_progress()
                logger.info(
                    f"global_step: {train_state.step}"
                    f"  epoch: {data_loader.epoch}"
                    f"  epoch_pct: {epoch_progress:.1f}%"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(), 4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                )

            # Validation evaluation
            if val_loader is not None and every_n_steps(train_state, args.eval.every, acc_step=0):
                logger.info(f"Running validation at step {train_state.step}...")
                val_metrics = evaluate_validation(
                    model, val_loader, max_steps=args.eval.max_steps
                )
                model.train()  # Switch back to training mode

                # Log validation metrics
                val_metrics_flat = {f"eval/{k}": v for k, v in val_metrics.items()}
                val_metrics_flat["global_step"] = train_state.step
                if get_is_master():
                    metric_logger.log(val_metrics_flat)

                logger.info(
                    f"Validation: step={train_state.step}"
                    f"  val_loss={val_metrics['val_loss']:.4f}"
                    f"  val_ppl={val_metrics['val_perplexity']:.2f}"
                )

            # Checkpointing
            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                # Update data loader state
                train_state.data_loader_state.epoch = data_loader.epoch
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            # Handle preemption
            if preemption_flag["flag"]:
                if not saved:
                    train_state.data_loader_state.epoch = data_loader.epoch
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                requeue_slurm_job()
                sys.exit(0)

    # Final checkpoint
    if not saved:
        train_state.data_loader_state.epoch = data_loader.epoch
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    """Main entry point with OmegaConf CLI."""
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(EncDecTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
