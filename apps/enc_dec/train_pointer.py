# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Training script for Pointer Mechanism Encoder-Decoder.

Similar to train.py but uses EncDecPointerTransformer and
position-based labels from data_pointer.py.
"""

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

from apps.enc_dec.enc_dec_pointer import (
    EncDecPointerTransformer,
    PointerMechanismArgs,
    build_pointer_fsdp_grouping_plan,
)
from apps.enc_dec.enc_dec import (
    EncDecTransformerArgs,
    EncoderType,
    get_no_recompute_ops,
    get_num_flop_per_token_enc_dec,
)
from apps.enc_dec.data_pointer import (
    PointerDataArgs,
    build_pointer_train_val_dataloaders,
)
from apps.enc_dec.data import (
    EncDecDataLoaderState,
    init_dataloader_state,
)

logger = logging.getLogger()


@torch.no_grad()
def evaluate_pointer_validation(
    model: EncDecPointerTransformer,
    dataloader,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate pointer model on validation set.

    Args:
        model: Pointer mechanism encoder-decoder model
        dataloader: Validation dataloader
        max_steps: Maximum number of steps (None = full validation set)

    Returns:
        Dictionary with validation metrics:
        - val_loss: Average cross-entropy loss over positions
        - val_accuracy: Average pointer accuracy
        - val_steps: Number of validation steps
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    num_steps = 0

    for batch in dataloader:
        encoder_input_ids = batch["encoder_input_ids"].cuda()
        decoder_input_ids = batch["decoder_input_ids"].cuda()
        target_positions = batch["target_positions"].cuda()
        encoder_mask = batch["encoder_padding_mask"].cuda()

        # Forward pass returns (loss, aux_dict)
        loss, aux = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            target_positions=target_positions,
            encoder_padding_mask=encoder_mask,
        )

        # Accumulate loss
        valid_mask = (target_positions != -100)
        num_valid = valid_mask.sum().item()
        total_loss += loss.item() * num_valid
        total_valid += num_valid

        # Accumulate accuracy
        total_correct += aux.get("pointer_accuracy", 0.0) * num_valid

        num_steps += 1

        if max_steps is not None and num_steps >= max_steps:
            break

    avg_loss = total_loss / max(total_valid, 1)
    avg_accuracy = total_correct / max(total_valid, 1)

    return {
        "val_loss": avg_loss,
        "val_accuracy": avg_accuracy,
        "val_steps": num_steps,
    }


@dataclass
class EvalArgs:
    """Evaluation configuration."""
    every: int = 100
    max_steps: Optional[int] = None


@dataclass
class PointerTrainArgs:
    """Training arguments for pointer mechanism encoder-decoder."""

    name: str = "pointer_enc_dec"
    dump_dir: str = ""

    seed: int = 42
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000

    steps: Optional[int] = None
    max_epochs: Optional[int] = None

    data: PointerDataArgs = field(default_factory=PointerDataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: EncDecTransformerArgs = field(default_factory=EncDecTransformerArgs)
    pointer: PointerMechanismArgs = field(default_factory=PointerMechanismArgs)

    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    eval: EvalArgs = field(default_factory=EvalArgs)


@dataclass
class PointerTrainState(Stateful):
    """Training state."""

    step: int
    acc_step: int
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


def validate_train_args(args: PointerTrainArgs, vocab_size: int):
    """Validate training arguments."""
    if args.model.vocab_size < 0:
        logger.info(f"Setting model vocab_size to {vocab_size}")
        args.model.vocab_size = vocab_size

    assert args.model.vocab_size == vocab_size, "Vocab size mismatch"
    assert args.dump_dir, "dump_dir must be specified"

    if args.checkpoint.path is None:
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard
        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate //= args.distributed.tp_size

    if args.steps is None and args.max_epochs is None:
        raise ValueError("Either 'steps' or 'max_epochs' must be specified")

    args.model.max_encoder_seqlen = args.data.max_encoder_len
    args.model.max_decoder_seqlen = args.data.max_decoder_len


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Preemption signal received")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def should_continue_training(step: int, epoch: int, args: PointerTrainArgs) -> bool:
    if args.steps is not None:
        return step < args.steps
    return epoch < args.max_epochs


def train(args: PointerTrainArgs):
    """Main training function for pointer mechanism."""
    with ExitStack() as context_stack:
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

        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()

        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        torch.manual_seed(args.seed)
        logger.info("Building Pointer Mechanism model")

        encoder_type = EncoderType(args.model.encoder_type)
        logger.info(f"Encoder type: {encoder_type.value}")

        # Build model
        if encoder_type == EncoderType.PRETRAINED:
            logger.info(f"Loading pretrained encoder: {args.model.pretrained_encoder.model_name}")
            model = EncDecPointerTransformer(args.model, args.pointer)
        else:
            with torch.device("meta"):
                model = EncDecPointerTransformer(args.model, args.pointer)

        logger.info("Model built!")

        model_param_count = get_num_params(model)
        encoder_param_count = get_num_params(model.encoder)
        decoder_param_count = get_num_params(model.decoder)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {model_param_count:,}")
        logger.info(f"Encoder parameters: {encoder_param_count:,}")
        logger.info(f"Decoder parameters: {decoder_param_count:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Parallelize
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_pointer_fsdp_grouping_plan(args.model, args.pointer),
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        )

        # Initialize weights
        if encoder_type == EncoderType.PRETRAINED:
            model = model.cuda()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.decoder.init_weights(shared_embeddings=False)
            if hasattr(model.encoder, 'projection') and model.encoder.projection is not None:
                model.encoder.reset_parameters()
        else:
            model = model.to_empty(device="cuda")
            if args.checkpoint.init_ckpt_path:
                load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
                model.encoder.rope_embeddings.reset_parameters()
                model.decoder.rope_embeddings.reset_parameters()
            else:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(args.model.seed)
                    model.init_weights()

        check_model_value_range(model, range=10.0, std=1.0)

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(f"GPU: {gpu_memory_monitor.device_name} with {gpu_memory_monitor.device_capacity_gib:.2f}GiB")

        # Data loaders (using pointer data module)
        data_loader, val_loader = build_pointer_train_val_dataloaders(args.data, dp_rank, dp_degree)

        # Compute total steps
        if args.steps is not None:
            total_steps = args.steps
        else:
            batches_per_epoch = len(data_loader)
            steps_per_epoch = batches_per_epoch // args.grad_acc_steps
            total_steps = steps_per_epoch * args.max_epochs

        optimizer, scheduler = build_optimizer(model, args.optim, total_steps)
        data_loader_state = init_dataloader_state()

        train_state = PointerTrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        gc.disable()
        model.train()

        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )

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

            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            batch = next(data_loader)

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                gc.collect()

            encoder_input_ids = batch["encoder_input_ids"].cuda()
            decoder_input_ids = batch["decoder_input_ids"].cuda()
            target_positions = batch["target_positions"].cuda()
            encoder_padding_mask = batch["encoder_padding_mask"].cuda()

            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += decoder_input_ids.numel()

            bsz, dec_seqlen = decoder_input_ids.shape
            _, enc_seqlen = encoder_input_ids.shape

            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            # Forward pass returns (loss, aux_dict)
            loss, aux = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                target_positions=target_positions,
                encoder_padding_mask=encoder_padding_mask,
            )

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            loss = loss / args.grad_acc_steps
            loss.backward()
            loss = loss.detach() * args.grad_acc_steps

            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )
                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            end_timer.record()
            torch.cuda.synchronize()
            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

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

                total_acc_steps = args.grad_acc_steps * train_state.step + train_state.acc_step
                tokens_per_gpu = total_acc_steps * args.data.batch_size * dec_seqlen
                total_tokens = dp_degree * tokens_per_gpu

                FLOPS = (
                    get_num_flop_per_token_enc_dec(
                        encoder_param_count, decoder_param_count,
                        args.model.encoder.n_layers, args.model.decoder.n_layers,
                        args.model.dim, enc_seqlen, dec_seqlen,
                    ) * wps
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
                        "pointer": aux,  # Pointer mechanism stats
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
                    f"step: {train_state.step}"
                    f"  epoch: {data_loader.epoch}"
                    f"  pct: {epoch_progress:.1f}%"
                    f"  loss: {round(loss.item(), 4):>7}"
                    f"  ptr_acc: {aux.get('pointer_accuracy', 0):.3f}"
                    f"  grad: {grad_norm:.2e}"
                    f"  lr: {curr_lr:.2e}"
                )

            # Validation evaluation
            if val_loader is not None and every_n_steps(train_state, args.eval.every, acc_step=0):
                logger.info(f"Running validation at step {train_state.step}...")
                val_metrics = evaluate_pointer_validation(
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
                    f"  val_acc={val_metrics['val_accuracy']:.4f}"
                )

            # Checkpointing
            saved = False
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0) or \
               every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                train_state.data_loader_state.epoch = data_loader.epoch
                saved = checkpoint.save(model, optimizer, train_state, args, device_mesh=world_mesh)

            if preemption_flag["flag"]:
                if not saved:
                    train_state.data_loader_state.epoch = data_loader.epoch
                    checkpoint.save(model, optimizer, train_state, args, device_mesh=world_mesh)
                requeue_slurm_job()
                sys.exit(0)

    if not saved:
        train_state.data_loader_state.epoch = data_loader.epoch
        checkpoint.save(model, optimizer, train_state, args, device_mesh=world_mesh)
    gc.collect()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_cfg = OmegaConf.structured(PointerTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
