# Copyright (c) Meta Platforms, Inc. and affiliates.

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
class EncDecDataArgs:
    """Data arguments for encoder-decoder QA training."""

    # HuggingFace dataset configuration
    dataset_name: str = ""
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # Limit number of samples (None = use all)

    # Column names in the dataset
    question_column: str = "question"
    answer_column: str = "answer"
    context_column: str = "gold_doc"

    # Sequence length limits
    max_encoder_len: int = 2048
    max_decoder_len: int = 512

    # Training parameters
    batch_size: int = 8
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True

    # Tokenizer configuration (used for decoder, and encoder if encoder_tokenizer not set)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)

    # Optional: separate encoder tokenizer (for pretrained encoders like ModernBERT)
    # If set, uses HuggingFace AutoTokenizer for encoder input
    encoder_tokenizer_name: Optional[str] = None  # e.g., "answerdotai/ModernBERT-base"

    # Train/validation split
    # If > 0, splits the training data into train/val (e.g., 0.1 = 10% for validation)
    val_split_ratio: float = 0.0

    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2


class QADataset(Dataset):
    """Dataset for QA with encoder-decoder architecture.

    Encoder input: gold_doc tokens
    Decoder input: question tokens + answer tokens (concatenated)
    Decoder labels: -100 for question positions, answer tokens for answer positions

    Supports separate tokenizers for encoder and decoder when using pretrained encoders.
    """

    def __init__(
        self,
        args: EncDecDataArgs,
        tokenizer,
        split: str = "train",
        hf_dataset=None,  # Optional: pass pre-loaded/split HF dataset
    ):
        self.args = args
        self.tokenizer = tokenizer  # Decoder tokenizer
        self.split = split

        # Setup encoder tokenizer (separate from decoder if specified)
        self.encoder_tokenizer = None
        self.use_hf_encoder_tokenizer = args.encoder_tokenizer_name is not None

        if self.use_hf_encoder_tokenizer:
            from transformers import AutoTokenizer
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
            logger.info(f"Using HuggingFace encoder tokenizer: {args.encoder_tokenizer_name}")

        # Use provided dataset or load from HuggingFace
        if hf_dataset is not None:
            self.dataset = hf_dataset
            logger.info(f"Using provided dataset with {len(self.dataset)} examples ({split})")
        else:
            # Load HuggingFace dataset
            from datasets import load_dataset

            if args.dataset_config:
                self.dataset = load_dataset(
                    args.dataset_name, args.dataset_config, split=split
                )
            else:
                self.dataset = load_dataset(args.dataset_name, split=split)

            # Limit number of samples if specified
            if args.max_samples is not None and args.max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(args.max_samples))
                logger.info(f"Limited to {args.max_samples} samples from {args.dataset_name} ({split})")
            else:
                logger.info(f"Loaded {len(self.dataset)} examples from {args.dataset_name} ({split})")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Get text fields
        context = item[self.args.context_column]
        question = item[self.args.question_column]
        answer = item[self.args.answer_column]['text'][0] #TODO fix for squad only

        # Handle different answer formats (some datasets have list of answers)
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if isinstance(answer, dict) and "text" in answer:
            # SQuAD-style answer format
            answer = answer["text"][0] if answer["text"] else ""

        # Tokenize encoder input (context/document)
        if self.use_hf_encoder_tokenizer:
            # Use HuggingFace tokenizer for encoder (e.g., ModernBERT)
            enc_output = self.encoder_tokenizer(
                context,
                truncation=True,
                max_length=self.args.max_encoder_len,
                return_tensors=None,
                add_special_tokens=True,
            )
            doc_tokens = enc_output["input_ids"]
        else:
            # Use lingua tokenizer
            doc_tokens = self.tokenizer.encode(
                context,
                add_bos=self.args.add_bos,
                add_eos=self.args.add_eos,
            )

        # Tokenize decoder input (question + separator + answer) - always use lingua tokenizer
        # Add separator between question and answer for clearer boundary
        question_with_sep = question + "\nAnswer:"
        question_tokens = self.tokenizer.encode(
            question_with_sep,
            add_bos=self.args.add_bos,  # BOS at start of decoder input
            add_eos=False,  # No EOS between question and answer
        )

        answer_tokens = self.tokenizer.encode(
            answer,
            add_bos=False,  # No BOS between question and answer
            add_eos=self.args.add_eos,
        )

        # Truncate encoder input
        encoder_input = doc_tokens[: self.args.max_encoder_len]

        # Build decoder input and labels
        # Decoder sees: [BOS] question answer [EOS]
        # Labels are shifted: question[1:] + answer + [EOS], with question part masked
        decoder_input = question_tokens + answer_tokens

        # Create labels: mask question tokens with -100 (ignored in loss)
        question_len = len(question_tokens)
        labels = [-100] * question_len + answer_tokens

        # Truncate decoder
        if len(decoder_input) > self.args.max_decoder_len:
            decoder_input = decoder_input[: self.args.max_decoder_len]
            labels = labels[: self.args.max_decoder_len]

        # Shift for next-token prediction
        # Input: tokens[:-1], Labels: tokens[1:]
        if len(decoder_input) > 1:
            decoder_input_shifted = decoder_input[:-1]
            labels_shifted = labels[1:]
        else:
            decoder_input_shifted = decoder_input
            labels_shifted = labels

        return {
            "encoder_input_ids": torch.tensor(encoder_input, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_shifted, dtype=torch.long),
            "labels": torch.tensor(labels_shifted, dtype=torch.long),
        }


def qa_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    encoder_pad_id: int,
    decoder_pad_id: int,
    max_encoder_len: int,
    max_decoder_len: int,
) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding.

    Pads sequences to the maximum length in the batch or the specified max lengths.
    """
    # Find max lengths in this batch
    max_enc_len_batch = min(
        max(item["encoder_input_ids"].shape[0] for item in batch),
        max_encoder_len,
    )
    max_dec_len_batch = min(
        max(item["decoder_input_ids"].shape[0] for item in batch),
        max_decoder_len - 1,  # -1 because we shifted
    )

    encoder_inputs = []
    decoder_inputs = []
    labels = []
    encoder_masks = []

    for item in batch:
        enc_len = item["encoder_input_ids"].shape[0]
        dec_len = item["decoder_input_ids"].shape[0]

        # Truncate if necessary
        enc_input = item["encoder_input_ids"][:max_enc_len_batch]
        dec_input = item["decoder_input_ids"][:max_dec_len_batch]
        lab = item["labels"][:max_dec_len_batch]

        enc_len = enc_input.shape[0]
        dec_len = dec_input.shape[0]

        # Pad encoder (use encoder's pad token to avoid vocab size mismatch)
        enc_padded = F.pad(
            enc_input, (0, max_enc_len_batch - enc_len), value=encoder_pad_id
        )
        encoder_inputs.append(enc_padded)

        # Create encoder mask (True for valid tokens)
        enc_mask = torch.zeros(max_enc_len_batch, dtype=torch.bool)
        enc_mask[:enc_len] = True
        encoder_masks.append(enc_mask)

        # Pad decoder (use decoder's pad token)
        dec_padded = F.pad(
            dec_input, (0, max_dec_len_batch - dec_len), value=decoder_pad_id
        )
        decoder_inputs.append(dec_padded)

        # Pad labels with -100 (ignored in loss)
        lab_padded = F.pad(lab, (0, max_dec_len_batch - dec_len), value=-100)
        labels.append(lab_padded)

    return {
        "encoder_input_ids": torch.stack(encoder_inputs),
        "decoder_input_ids": torch.stack(decoder_inputs),
        "labels": torch.stack(labels),
        "encoder_padding_mask": torch.stack(encoder_masks),
    }


def build_qa_dataloader(
    args: EncDecDataArgs,
    rank: int,
    world_size: int,
    split: str = "train",
) -> DataLoader:
    """Build distributed dataloader for QA dataset."""
    tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
    dataset = QADataset(args, tokenizer, split)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        seed=args.seed,
    )

    # Get decoder pad token ID (use EOS if no dedicated pad token)
    decoder_pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    # Get encoder pad token ID (from HF tokenizer if using pretrained encoder)
    if args.encoder_tokenizer_name:
        from transformers import AutoTokenizer
        enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
        encoder_pad_id = enc_tokenizer.pad_token_id if enc_tokenizer.pad_token_id is not None else 0
    else:
        # Same tokenizer for both encoder and decoder
        encoder_pad_id = decoder_pad_id

    collate_fn = partial(
        qa_collate_fn,
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


class InfiniteDataLoader:
    """Wrapper to make DataLoader infinite by resetting on exhaustion."""

    def __init__(self, dataloader: DataLoader, sampler: DistributedSampler):
        self.dataloader = dataloader
        self.sampler = sampler
        self.epoch = 0
        self.step_in_epoch = 0
        self._iterator = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
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
        """Set the epoch for the sampler."""
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def get_epoch_progress(self) -> float:
        """Return the percentage of the current epoch completed."""
        total_batches = len(self.dataloader)
        if total_batches == 0:
            return 0.0
        return (self.step_in_epoch / total_batches) * 100.0

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return len(self.dataloader)


def build_infinite_qa_dataloader(
    args: EncDecDataArgs,
    rank: int,
    world_size: int,
    split: str = "train",
) -> InfiniteDataLoader:
    """Build infinite distributed dataloader for QA dataset."""
    tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
    dataset = QADataset(args, tokenizer, split)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        seed=args.seed,
    )

    # Get decoder pad token ID (use EOS if no dedicated pad token)
    decoder_pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    # Get encoder pad token ID (from HF tokenizer if using pretrained encoder)
    if args.encoder_tokenizer_name:
        from transformers import AutoTokenizer
        enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
        encoder_pad_id = enc_tokenizer.pad_token_id if enc_tokenizer.pad_token_id is not None else 0
    else:
        # Same tokenizer for both encoder and decoder
        encoder_pad_id = decoder_pad_id

    collate_fn = partial(
        qa_collate_fn,
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

    return InfiniteDataLoader(dataloader, sampler)


@dataclass
class EncDecDataLoaderState:
    """State for resumable data loading."""
    epoch: int = 0
    step_in_epoch: int = 0


def init_dataloader_state() -> EncDecDataLoaderState:
    """Initialize data loader state."""
    return EncDecDataLoaderState(epoch=0, step_in_epoch=0)


def build_train_val_dataloaders(
    args: EncDecDataArgs,
    rank: int,
    world_size: int,
) -> Tuple[InfiniteDataLoader, Optional[DataLoader]]:
    """Build train and validation dataloaders with optional train/val split.

    If args.val_split_ratio > 0, splits the training data into train/val.
    Otherwise, returns only the training dataloader (val_dataloader=None).

    Args:
        args: Data configuration
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Tuple of (train_dataloader, val_dataloader)
        val_dataloader is None if val_split_ratio == 0
    """
    from datasets import load_dataset

    tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)

    # Get pad token IDs
    decoder_pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)
    if args.encoder_tokenizer_name:
        from transformers import AutoTokenizer
        enc_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer_name)
        encoder_pad_id = enc_tokenizer.pad_token_id if enc_tokenizer.pad_token_id is not None else 0
    else:
        encoder_pad_id = decoder_pad_id

    collate_fn = partial(
        qa_collate_fn,
        encoder_pad_id=encoder_pad_id,
        decoder_pad_id=decoder_pad_id,
        max_encoder_len=args.max_encoder_len,
        max_decoder_len=args.max_decoder_len,
    )

    val_dataloader = None

    if args.val_split_ratio > 0:
        # Load and split the training data
        logger.info(f"Splitting training data: {1 - args.val_split_ratio:.0%} train, {args.val_split_ratio:.0%} val")

        if args.dataset_config:
            full_dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")
        else:
            full_dataset = load_dataset(args.dataset_name, split="train")

        # Limit samples before splitting if specified
        if args.max_samples is not None and args.max_samples < len(full_dataset):
            full_dataset = full_dataset.select(range(args.max_samples))

        # Split into train/val
        splits = full_dataset.train_test_split(test_size=args.val_split_ratio, seed=args.seed)
        train_hf_dataset = splits["train"]
        val_hf_dataset = splits["test"]

        logger.info(f"Train size: {len(train_hf_dataset)}, Val size: {len(val_hf_dataset)}")

        # Create train dataset and dataloader
        train_dataset = QADataset(args, tokenizer, split="train", hf_dataset=train_hf_dataset)
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
        train_loader = InfiniteDataLoader(train_dataloader, train_sampler)

        # Create val dataset and dataloader
        val_dataset = QADataset(args, tokenizer, split="val", hf_dataset=val_hf_dataset)
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
        # No splitting, just build training dataloader
        train_loader = build_infinite_qa_dataloader(args, rank, world_size, split="train")

    return train_loader, val_dataloader
