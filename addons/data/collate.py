"""
Collation utilities for packed sequences.

Framework-agnostic - pure PyTorch, no distributed imports.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from addons.tasks.schema import BatchedContextBasedExamples


@dataclass
class PackedSequences:
    """
    Packed sequences with cumulative lengths for variable-length attention.

    No padding - variable lengths handled via cu_seqlens (xformers/flash-attn).
    """

    tokens: torch.Tensor          # [total_tokens] or [total_tokens, dim] for hidden states
    cu_seqlens: torch.Tensor      # [num_seqs + 1] cumulative lengths
    lengths: List[int]            # [num_seqs] length of each sequence

    @property
    def max_seqlen(self) -> int:
        return max(self.lengths) if self.lengths else 0

    @property
    def num_seqs(self) -> int:
        return len(self.lengths)

    @classmethod
    def from_tensors(
        cls,
        tensors: List[torch.Tensor],
        device: torch.device,
    ) -> "PackedSequences":
        """Pack list of tensors into single sequence with cu_seqlens."""
        if not tensors:
            return cls(
                tokens=torch.tensor([], device=device),
                cu_seqlens=torch.tensor([0], dtype=torch.int32, device=device),
                lengths=[],
            )

        lengths = [t.shape[0] for t in tensors]
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(lengths), dim=0)),
            dtype=torch.int32,
            device=device,
        )
        tokens = torch.cat(tensors, dim=0).to(device)

        return cls(tokens=tokens, cu_seqlens=cu_seqlens, lengths=lengths)


@dataclass
class TokenizedBatch:
    """
    Tokenized version of BatchedContextBasedExamples.

    Mirrors BatchedContextBasedExamples structure:
    - documents (text) -> encoder_tokens (packed)
    - queries + targets (text) -> decoder_tokens (packed) + labels
    - document_hashes -> doc_hash_to_idx + example_doc_indices
    """

    # Encoder: packed tokenized documents (deduplicated)
    encoder_tokens: PackedSequences
    doc_hash_to_idx: Dict[str, int]

    # Decoder: packed tokenized sequences (query + target)
    decoder_tokens: PackedSequences
    labels: torch.Tensor  # [total_dec_tokens]

    # Cross-attention mapping: which docs each example attends to
    example_doc_indices: List[List[int]]

    @property
    def batch_size(self) -> int:
        return self.decoder_tokens.num_seqs

    @classmethod
    def from_batched_examples(
        cls,
        batch: BatchedContextBasedExamples,
        encoder_tokenizer: Any,
        decoder_tokenizer: Any,
        encoder_max_len: int,
        decoder_max_len: int,
        device: torch.device,
    ) -> "TokenizedBatch":
        """
        Tokenize a BatchedContextBasedExamples.

        Steps:
        1. Tokenize deduplicated documents -> encoder_tokens
        2. Tokenize queries + targets -> decoder_tokens + labels
        3. Map document_hashes to indices
        """
        raise NotImplementedError
