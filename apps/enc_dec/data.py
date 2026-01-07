# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Any, Iterator
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

        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")

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

        # Tokenize decoder input (question + answer) - always use lingua tokenizer
        question_tokens = self.tokenizer.encode(
            question,
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
    pad_id: int,
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

        # Pad encoder
        enc_padded = F.pad(
            enc_input, (0, max_enc_len_batch - enc_len), value=pad_id
        )
        encoder_inputs.append(enc_padded)

        # Create encoder mask (True for valid tokens)
        enc_mask = torch.zeros(max_enc_len_batch, dtype=torch.bool)
        enc_mask[:enc_len] = True
        encoder_masks.append(enc_mask)

        # Pad decoder
        dec_padded = F.pad(
            dec_input, (0, max_dec_len_batch - dec_len), value=pad_id
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

    # Get pad token ID (use EOS if no dedicated pad token)
    pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    collate_fn = partial(
        qa_collate_fn,
        pad_id=pad_id,
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
        self._iterator = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._iterator is None:
            self.sampler.set_epoch(self.epoch)
            self._iterator = iter(self.dataloader)

        try:
            return next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self._iterator = iter(self.dataloader)
            return next(self._iterator)

    def set_epoch(self, epoch: int):
        """Set the epoch for the sampler."""
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

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

    pad_id = getattr(tokenizer, "pad_id", tokenizer.eos_id)

    collate_fn = partial(
        qa_collate_fn,
        pad_id=pad_id,
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
