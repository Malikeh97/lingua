"""
Collation utilities for packed sequences.

Framework-agnostic - pure PyTorch, no distributed imports.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

_DEBUG = os.environ.get("DEBUG") == "1"
_debug_printed = False

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
        # 1. Tokenize deduplicated documents
        # Assign a stable index to each unique doc hash
        doc_hash_to_idx: Dict[str, int] = {}
        doc_tensors: List[torch.Tensor] = []

        for doc_hash, doc_text in batch.documents.items():
            doc_hash_to_idx[doc_hash] = len(doc_tensors)
            enc = encoder_tokenizer(
                doc_text,
                max_length=encoder_max_len,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            doc_tensors.append(enc["input_ids"].squeeze(0))

        encoder_tokens = PackedSequences.from_tensors(doc_tensors, device)

        # 2. Tokenize queries + targets -> decoder tokens + labels
        dec_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []
        example_doc_indices: List[List[int]] = []

        for i in range(len(batch)):
            query = batch.queries[i]
            target = batch.target_texts[i]

            # Tokenize query and target without special tokens
            # We manage BOS/EOS explicitly: [BOS] query_tokens target_tokens [EOS]
            bos_id = decoder_tokenizer.cls_token_id or decoder_tokenizer.bos_token_id
            eos_id = decoder_tokenizer.sep_token_id or decoder_tokenizer.eos_token_id

            query_enc = decoder_tokenizer(
                query,
                add_special_tokens=False,
                return_tensors="pt",
            )
            target_enc = decoder_tokenizer(
                target,
                add_special_tokens=False,
                return_tensors="pt",
            )
            query_ids = query_enc["input_ids"].squeeze(0)
            target_ids = target_enc["input_ids"].squeeze(0)

            # Build: [BOS] query target [EOS]
            parts = []
            if bos_id is not None:
                parts.append(torch.tensor([bos_id]))
            parts.append(query_ids)
            prompt_len = sum(p.shape[0] for p in parts)
            parts.append(target_ids)
            if eos_id is not None:
                parts.append(torch.tensor([eos_id]))

            full_ids = torch.cat(parts, dim=0)[:decoder_max_len]

            # Labels: -100 for prompt (BOS + query), predict target + EOS
            query_len = min(prompt_len, decoder_max_len)
            labels = full_ids.clone()
            labels[:query_len] = -100

            dec_tensors.append(full_ids)
            label_tensors.append(labels)

            # Map this example's doc hashes to indices
            example_doc_indices.append(
                [doc_hash_to_idx[h] for h in batch.document_hashes[i]]
            )

        global _debug_printed
        if _DEBUG and not _debug_printed:
            _debug_printed = True
            print("[collate DEBUG] First batch tokenization:")
            print(f"  bos_id={bos_id}, eos_id={eos_id}")
            print(f"  num_examples={len(batch)}, num_unique_docs={len(batch.documents)}")
            print(f"  encoder: {len(doc_tensors)} docs, lengths={[t.shape[0] for t in doc_tensors]}")
            print(f"  decoder: {len(dec_tensors)} seqs, lengths={[t.shape[0] for t in dec_tensors]}")
            for j in range(min(2, len(dec_tensors))):
                print(f"  --- example[{j}] ---")
                print(f"    source: {batch.sources[j]}")
                print(f"    query: {batch.queries[j]!r}")
                print(f"    target: {batch.target_texts[j]!r}")
                print(f"    num_docs: {len(batch.document_hashes[j])}")
                doc_idx = example_doc_indices[j][0]
                print(f"    enc_ids[0]: {doc_tensors[doc_idx].tolist()[:20]}...")
                enc_text = decoder_tokenizer.decode(doc_tensors[doc_idx], skip_special_tokens=False)
                if len(enc_text) > 400:
                    enc_text = enc_text[:200] + " ... " + enc_text[-200:]
                print(f"    enc_decoded[0]: {enc_text!r}")
                print(f"    dec_ids: {dec_tensors[j].tolist()}")
                print(f"    dec_decoded: {decoder_tokenizer.decode(dec_tensors[j], skip_special_tokens=False)!r}")
                print(f"    labels:  {label_tensors[j].tolist()}")
                print(f"    doc_indices: {example_doc_indices[j]}")

        decoder_tokens = PackedSequences.from_tensors(dec_tensors, device)
        all_labels = torch.cat(label_tensors, dim=0).to(device)

        return cls(
            encoder_tokens=encoder_tokens,
            doc_hash_to_idx=doc_hash_to_idx,
            decoder_tokens=decoder_tokens,
            labels=all_labels,
            example_doc_indices=example_doc_indices,
        )

    def prompt_only(self) -> "TokenizedBatch":
        """
        Strip target tokens, keeping only the prompt (where labels == -100).

        Used at generation time: the model receives the prompt and generates the rest.
        """
        device = self.decoder_tokens.tokens.device
        cu = self.decoder_tokens.cu_seqlens
        prompt_tensors = []

        for i in range(self.decoder_tokens.num_seqs):
            s = cu[i].item()
            e = cu[i + 1].item()
            seq_labels = self.labels[s:e]
            prompt_len = (seq_labels == -100).sum().item()
            prompt_tensors.append(self.decoder_tokens.tokens[s:s + prompt_len])

        return TokenizedBatch(
            encoder_tokens=self.encoder_tokens,
            doc_hash_to_idx=self.doc_hash_to_idx,
            decoder_tokens=PackedSequences.from_tensors(prompt_tensors, device),
            labels=torch.tensor([], device=device),  # no labels at generation
            example_doc_indices=self.example_doc_indices,
        )
