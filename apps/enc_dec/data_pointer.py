# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Data module for Pointer-based Extractive QA.

This module provides datasets and data loaders for training the pointer mechanism.
Instead of token-level labels, it provides position labels indicating which encoder
positions correspond to the answer span.

For SQuAD-style datasets:
- Encoder input: context/document
- Decoder input: question (prompting the model to point to answer)
- Target: positions in encoder that correspond to answer tokens
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Any, Iterator, Tuple
import logging

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F

from lingua.tokenizer import build_tokenizer, TokenizerArgs

logger = logging.getLogger()


@dataclass
class PointerDataArgs:
    """Data arguments for pointer-based extractive QA training."""

    # HuggingFace dataset configuration
    dataset_name: str = "squad"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None

    # Column names (SQuAD format)
    question_column: str = "question"
    answer_column: str = "answers"  # SQuAD has "answers" with "text" and "answer_start"
    context_column: str = "context"

    # Sequence length limits
    max_encoder_len: int = 2048
    max_decoder_len: int = 128  # Shorter for pointer (just question + answer positions)

    # Training parameters
    batch_size: int = 8
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True

    # Tokenizer configuration (decoder tokenizer)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)

    # Encoder tokenizer (for pretrained encoders like ModernBERT)
    encoder_tokenizer_name: Optional[str] = None

    # Train/validation split
    val_split_ratio: float = 0.0

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2


class PointerQADataset(Dataset):
    """Dataset for pointer-based extractive QA.

    Encoder input: context tokens
    Decoder input: question tokens
    Target: encoder positions for each answer token

    The key difference from standard QA dataset is that targets are positions
    in the encoder sequence, not token IDs.
    """

    def __init__(
        self,
        args: PointerDataArgs,
        tokenizer,
        split: str = "train",
        hf_dataset=None,
    ):
        self.args = args
        self.tokenizer = tokenizer  # Decoder tokenizer
        self.split = split

        # Setup encoder tokenizer
        self.encoder_tokenizer = None
        self.use_hf_encoder_tokenizer = args.encoder_tokenizer_name is not None

        if self.use_hf_encoder_tokenizer:
            from transformers import AutoTokenizer
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
            logger.info(f"Using HuggingFace encoder tokenizer: {args.encoder_tokenizer_name}")

        # Load dataset
        if hf_dataset is not None:
            self.dataset = hf_dataset
            logger.info(f"Using provided dataset with {len(self.dataset)} examples ({split})")
        else:
            from datasets import load_dataset

            if args.dataset_config:
                self.dataset = load_dataset(args.dataset_name, args.dataset_config, split=split)
            else:
                self.dataset = load_dataset(args.dataset_name, split=split)

            if args.max_samples is not None and args.max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(args.max_samples))
                logger.info(f"Limited to {args.max_samples} samples")
            else:
                logger.info(f"Loaded {len(self.dataset)} examples from {args.dataset_name} ({split})")

    def __len__(self) -> int:
        return len(self.dataset)

    def _find_answer_positions_in_encoder(
        self,
        context: str,
        answer_text: str,
        answer_start_char: int,
        encoder_output: dict,
    ) -> List[int]:
        """Find which encoder token positions correspond to the answer.

        Args:
            context: The full context text
            answer_text: The answer text
            answer_start_char: Character position where answer starts in context
            encoder_output: Output from HuggingFace tokenizer with offset_mapping

        Returns:
            List of encoder positions that correspond to answer tokens
        """
        answer_end_char = answer_start_char + len(answer_text)

        # Get offset mapping: list of (start_char, end_char) for each token
        offset_mapping = encoder_output.get("offset_mapping", None)

        if offset_mapping is None:
            # Fallback: try to find by substring matching (less accurate)
            logger.warning("No offset mapping available, using fallback position finding")
            return self._find_answer_positions_fallback(
                encoder_output["input_ids"], answer_text
            )

        answer_positions = []
        for token_idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens (they have (0, 0) offset)
            if start == end == 0:
                continue

            # Check if this token overlaps with answer span
            if start < answer_end_char and end > answer_start_char:
                answer_positions.append(token_idx)

        return answer_positions

    def _find_answer_positions_fallback(
        self,
        encoder_ids: List[int],
        answer_text: str,
    ) -> List[int]:
        """Fallback method to find answer positions by decoding and matching."""
        if self.encoder_tokenizer is None:
            return []

        # Decode each token and try to find the answer
        # This is a simple substring match - not perfect but works for simple cases
        decoded_tokens = [
            self.encoder_tokenizer.decode([tid], skip_special_tokens=False)
            for tid in encoder_ids
        ]

        # Build cumulative text to find positions
        answer_lower = answer_text.lower()
        cumulative = ""
        answer_positions = []

        for idx, token_text in enumerate(decoded_tokens):
            cumulative += token_text
            if answer_lower in cumulative.lower():
                # Found potential match, mark this position
                answer_positions.append(idx)

        return answer_positions

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Get text fields
        context = item[self.args.context_column]
        question = item[self.args.question_column]

        # Handle SQuAD answer format
        answers = item[self.args.answer_column]
        if isinstance(answers, dict):
            answer_text = answers["text"][0] if answers["text"] else ""
            answer_start_char = answers["answer_start"][0] if answers["answer_start"] else 0
        else:
            answer_text = answers[0] if isinstance(answers, list) else answers
            answer_start_char = 0  # Unknown, will use fallback

        # Tokenize encoder input (context)
        if self.use_hf_encoder_tokenizer:
            enc_output = self.encoder_tokenizer(
                context,
                truncation=True,
                max_length=self.args.max_encoder_len,
                return_tensors=None,
                add_special_tokens=True,
                return_offsets_mapping=True,  # Important for position finding
            )
            encoder_input_ids = enc_output["input_ids"]

            # Find answer positions in encoder
            answer_positions = self._find_answer_positions_in_encoder(
                context, answer_text, answer_start_char, enc_output
            )
        else:
            # Using lingua tokenizer (no offset mapping available)
            encoder_input_ids = self.tokenizer.encode(
                context,
                add_bos=self.args.add_bos,
                add_eos=self.args.add_eos,
            )
            # Use fallback position finding
            answer_positions = self._find_answer_positions_fallback(
                encoder_input_ids, answer_text
            )

        # Truncate encoder input
        encoder_input_ids = encoder_input_ids[:self.args.max_encoder_len]

        # Filter answer positions that are within truncated range
        answer_positions = [p for p in answer_positions if p < len(encoder_input_ids)]

        # Tokenize decoder input (question)
        question_with_prompt = question + "\nAnswer:"
        decoder_input_ids = self.tokenizer.encode(
            question_with_prompt,
            add_bos=self.args.add_bos,
            add_eos=False,
        )

        # Build target positions
        # For each decoder output position, we need a target encoder position
        # Structure: [question positions (masked)] + [answer positions]

        question_len = len(decoder_input_ids)

        # Target positions: -100 for question, then answer positions
        # The decoder outputs one position per output token
        # For extractive QA, we want to output the answer span positions
        target_positions = [-100] * question_len + answer_positions

        # Decoder input includes question + (previous answer tokens for autoregressive)
        # For training, we can use the answer tokens as decoder input after question
        # and target the NEXT position in the sequence
        answer_tokens = self.tokenizer.encode(
            answer_text,
            add_bos=False,
            add_eos=self.args.add_eos,
        )

        # Full decoder input: question + answer tokens
        full_decoder_input = decoder_input_ids + answer_tokens

        # Target positions: masked for question, position labels for answer
        # Shift for next-position prediction
        if len(answer_positions) > 0:
            # Extend positions if answer has fewer positions than tokens
            # (can happen due to tokenization differences)
            while len(target_positions) < len(full_decoder_input):
                # Repeat last position or use -100
                target_positions.append(
                    answer_positions[-1] if answer_positions else -100
                )

        # Truncate decoder
        if len(full_decoder_input) > self.args.max_decoder_len:
            full_decoder_input = full_decoder_input[:self.args.max_decoder_len]
            target_positions = target_positions[:self.args.max_decoder_len]

        # Shift for next-token prediction (decoder input[:-1], target[1:])
        if len(full_decoder_input) > 1:
            decoder_input_shifted = full_decoder_input[:-1]
            target_positions_shifted = target_positions[1:]
        else:
            decoder_input_shifted = full_decoder_input
            target_positions_shifted = target_positions

        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_shifted, dtype=torch.long),
            "target_positions": torch.tensor(target_positions_shifted, dtype=torch.long),
        }


def pointer_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    encoder_pad_id: int,
    decoder_pad_id: int,
    max_encoder_len: int,
    max_decoder_len: int,
) -> Dict[str, torch.Tensor]:
    """Collate function for pointer QA with dynamic padding."""

    # Find max lengths in batch
    max_enc_len_batch = min(
        max(item["encoder_input_ids"].shape[0] for item in batch),
        max_encoder_len,
    )
    max_dec_len_batch = min(
        max(item["decoder_input_ids"].shape[0] for item in batch),
        max_decoder_len - 1,
    )

    encoder_inputs = []
    decoder_inputs = []
    target_positions = []
    encoder_masks = []

    for item in batch:
        # Truncate
        enc_input = item["encoder_input_ids"][:max_enc_len_batch]
        dec_input = item["decoder_input_ids"][:max_dec_len_batch]
        targets = item["target_positions"][:max_dec_len_batch]

        enc_len = enc_input.shape[0]
        dec_len = dec_input.shape[0]
        targets_len = targets.shape[0]

        # Pad encoder
        enc_padded = F.pad(enc_input, (0, max_enc_len_batch - enc_len), value=encoder_pad_id)
        encoder_inputs.append(enc_padded)

        # Create encoder mask
        enc_mask = torch.zeros(max_enc_len_batch, dtype=torch.bool)
        enc_mask[:enc_len] = True
        encoder_masks.append(enc_mask)

        # Pad decoder
        dec_padded = F.pad(dec_input, (0, max_dec_len_batch - dec_len), value=decoder_pad_id)
        decoder_inputs.append(dec_padded)

        # Pad target positions with -100 (ignored in loss)
        targets_padded = F.pad(targets, (0, max_dec_len_batch - targets_len), value=-100)
        target_positions.append(targets_padded)

    return {
        "encoder_input_ids": torch.stack(encoder_inputs),
        "decoder_input_ids": torch.stack(decoder_inputs),
        "target_positions": torch.stack(target_positions),
        "encoder_padding_mask": torch.stack(encoder_masks),
    }


def build_pointer_dataloader(
    args: PointerDataArgs,
    rank: int,
    world_size: int,
    split: str = "train",
) -> DataLoader:
    """Build distributed dataloader for pointer QA."""

    tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
    dataset = PointerQADataset(args, tokenizer, split)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        seed=args.seed,
    )

    # Get pad token IDs
    decoder_pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    if args.encoder_tokenizer_name:
        from transformers import AutoTokenizer
        enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
        encoder_pad_id = enc_tokenizer.pad_token_id if enc_tokenizer.pad_token_id is not None else 0
    else:
        encoder_pad_id = decoder_pad_id

    collate_fn = partial(
        pointer_collate_fn,
        encoder_pad_id=encoder_pad_id,
        decoder_pad_id=decoder_pad_id,
        max_encoder_len=args.max_encoder_len,
        max_decoder_len=args.max_decoder_len,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


class InfinitePointerDataLoader:
    """Wrapper to make DataLoader infinite."""

    def __init__(self, dataloader: DataLoader, sampler: DistributedSampler):
        self.dataloader = dataloader
        self.sampler = sampler
        self.epoch = 0
        self.step_in_epoch = 0
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._iterator is None:
            self.sampler.set_epoch(self.epoch)
            self._iterator = iter(self.dataloader)

        try:
            batch = next(self._iterator)
            self.step_in_epoch += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.step_in_epoch = 0
            self.sampler.set_epoch(self.epoch)
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
            self.step_in_epoch += 1
            return batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def get_epoch_progress(self) -> float:
        total_batches = len(self.dataloader)
        if total_batches == 0:
            return 0.0
        return (self.step_in_epoch / total_batches) * 100.0

    def __len__(self) -> int:
        return len(self.dataloader)


def build_infinite_pointer_dataloader(
    args: PointerDataArgs,
    rank: int,
    world_size: int,
    split: str = "train",
) -> InfinitePointerDataLoader:
    """Build infinite distributed dataloader for pointer QA."""

    tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
    dataset = PointerQADataset(args, tokenizer, split)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        seed=args.seed,
    )

    decoder_pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    if args.encoder_tokenizer_name:
        from transformers import AutoTokenizer
        enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
        encoder_pad_id = enc_tokenizer.pad_token_id if enc_tokenizer.pad_token_id is not None else 0
    else:
        encoder_pad_id = decoder_pad_id

    collate_fn = partial(
        pointer_collate_fn,
        encoder_pad_id=encoder_pad_id,
        decoder_pad_id=decoder_pad_id,
        max_encoder_len=args.max_encoder_len,
        max_decoder_len=args.max_decoder_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    return InfinitePointerDataLoader(dataloader, sampler)


def build_pointer_train_val_dataloaders(
    args: PointerDataArgs,
    rank: int,
    world_size: int,
) -> Tuple[InfinitePointerDataLoader, Optional[DataLoader]]:
    """Build train and validation dataloaders with optional split."""

    from datasets import load_dataset

    tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)

    decoder_pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)
    if args.encoder_tokenizer_name:
        from transformers import AutoTokenizer
        enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
        encoder_pad_id = enc_tokenizer.pad_token_id if enc_tokenizer.pad_token_id is not None else 0
    else:
        encoder_pad_id = decoder_pad_id

    collate_fn = partial(
        pointer_collate_fn,
        encoder_pad_id=encoder_pad_id,
        decoder_pad_id=decoder_pad_id,
        max_encoder_len=args.max_encoder_len,
        max_decoder_len=args.max_decoder_len,
    )

    val_dataloader = None

    if args.val_split_ratio > 0:
        logger.info(f"Splitting data: {1 - args.val_split_ratio:.0%} train, {args.val_split_ratio:.0%} val")

        if args.dataset_config:
            full_dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
        else:
            full_dataset = load_dataset(args.dataset_name, split="train")

        if args.max_samples is not None and args.max_samples < len(full_dataset):
            full_dataset = full_dataset.select(range(args.max_samples))

        splits = full_dataset.train_test_split(test_size=args.val_split_ratio, seed=args.seed)
        train_hf = splits["train"]
        val_hf = splits["test"]

        logger.info(f"Train: {len(train_hf)}, Val: {len(val_hf)}")

        # Train
        train_dataset = PointerQADataset(args, tokenizer, split="train", hf_dataset=train_hf)
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        train_loader = InfinitePointerDataLoader(train_dataloader, train_sampler)

        # Val
        val_dataset = PointerQADataset(args, tokenizer, split="val", hf_dataset=val_hf)
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
    else:
        train_loader = build_infinite_pointer_dataloader(args, rank, world_size, split="train")

    return train_loader, val_dataloader
