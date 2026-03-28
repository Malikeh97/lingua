"""
Training loop for encoder-decoder and decoder-only models.

Reference: apps/minimal_squad/main.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from addons.data.collate import TokenizedBatch, PackedSequences


@dataclass
class TrainerArgs:
    """Training configuration."""

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Training
    epochs: int = 3
    log_interval: int = 10
    eval_interval: int = 500

    # Pretrained weight handling
    pretrained_lr_scale: Optional[float] = None  # None = freeze, >0 = scale LR

    # Device
    device: str = "cuda"


class Trainer:
    """
    Training loop for models accepting TokenizedBatch.

    Supports:
    - Encoder-decoder and decoder-only models
    - Distributed training via sp_group
    - Separate LR for pretrained vs new parameters
    - Gradient clipping
    - wandb logging (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainerArgs,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        sp_group: Optional[dist.ProcessGroup] = None,
        wandb_run: Optional[Any] = None,
    ):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sp_group = sp_group
        self.wandb = wandb_run

        self.device = torch.device(args.device)
        self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.global_step = 0
        self.epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate LR for pretrained parameters."""
        pretrained_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if getattr(param, "is_pretrained", False):
                pretrained_params.append(param)
            else:
                other_params.append(param)

        if self.args.pretrained_lr_scale is None or self.args.pretrained_lr_scale == 0.0:
            # Freeze pretrained
            for p in pretrained_params:
                p.requires_grad = False
            param_groups = [{"params": other_params, "lr": self.args.lr}]
        else:
            # Scale LR for pretrained
            param_groups = [
                {"params": pretrained_params, "lr": self.args.lr * self.args.pretrained_lr_scale},
                {"params": other_params, "lr": self.args.lr},
            ]

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {total:,} total, {trainable:,} trainable")

        return torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate scheduler."""
        if self.args.warmup_steps == 0:
            return None

        def lr_lambda(step):
            if step < self.args.warmup_steps:
                return step / self.args.warmup_steps
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _batch_to_device(self, batch: TokenizedBatch) -> TokenizedBatch:
        """Move batch tensors to device."""
        # Move encoder tokens
        enc = batch.encoder_tokens
        encoder_tokens = PackedSequences(
            tokens=enc.tokens.to(self.device),
            cu_seqlens=enc.cu_seqlens.to(self.device),
            lengths=enc.lengths,
        )

        # Move decoder tokens
        dec = batch.decoder_tokens
        decoder_tokens = PackedSequences(
            tokens=dec.tokens.to(self.device),
            cu_seqlens=dec.cu_seqlens.to(self.device),
            lengths=dec.lengths,
        )

        return TokenizedBatch(
            encoder_tokens=encoder_tokens,
            doc_hash_to_idx=batch.doc_hash_to_idx,
            decoder_tokens=decoder_tokens,
            labels=batch.labels.to(self.device),
            example_doc_indices=batch.example_doc_indices,
        )

    def train_step(self, batch: TokenizedBatch) -> torch.Tensor:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(batch, sp_group=self.sp_group)

        # Compute loss
        loss = self._compute_loss(logits, batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm,
            )

        # Optimizer step
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def _compute_loss(self, logits: torch.Tensor, batch: TokenizedBatch) -> torch.Tensor:
        """Compute cross-entropy loss with label shifting."""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:-1].contiguous()
        shift_labels = batch.labels[1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def eval_step(self, batch: TokenizedBatch) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()

        logits = self.model(batch, sp_group=self.sp_group)
        loss = self._compute_loss(logits, batch)

        return {"loss": loss.item()}

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            batch = self._batch_to_device(batch)

            loss = self.train_step(batch)
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Logging
            if self.global_step % self.args.log_interval == 0:
                if self.wandb is not None:
                    self.wandb.log({
                        "train/loss": loss.item(),
                        "train/step": self.global_step,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    })

            # Evaluation
            if self.val_loader and self.global_step % self.args.eval_interval == 0:
                val_metrics = self.evaluate()
                if self.wandb is not None:
                    self.wandb.log({
                        f"val/{k}": v for k, v in val_metrics.items()
                    })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            batch = self._batch_to_device(batch)
            metrics = self.eval_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1

        return {"loss": total_loss / num_batches}

    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.args.epochs + 1):
            self.epoch = epoch

            train_loss = self.train_epoch()
            history["train_loss"].append(train_loss)
            print(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

            if self.val_loader:
                val_metrics = self.evaluate()
                history["val_loss"].append(val_metrics["loss"])
                print(f"Epoch {epoch} - Val loss: {val_metrics['loss']:.4f}")

                if self.wandb is not None:
                    self.wandb.log({
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                        "val/epoch_loss": val_metrics["loss"],
                    })

        return history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
