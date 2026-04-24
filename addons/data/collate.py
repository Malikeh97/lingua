"""
Collation utilities for packed sequences.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

_DEBUG = os.environ.get("DEBUG") == "1"
_debug_printed = False

from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


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
        doc_max_tokens: int,
        target_max_tokens: int,
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
                max_length=doc_max_tokens,
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

            full_ids = torch.cat(parts, dim=0)[:target_max_tokens]

            # Labels: -100 for prompt (BOS + query), predict target + EOS
            query_len = min(prompt_len, target_max_tokens)
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

    @classmethod
    def from_tokenized_examples(
        cls,
        examples: List["ContextBasedExample"],
        device: torch.device = torch.device("cpu"),
    ) -> "TokenizedBatch":
        """
        Build TokenizedBatch from pre-tokenized ContextBasedExamples.

        Deduplicates documents by hash, packs encoder/decoder tokens.
        """
        # Deduplicate documents across all examples
        doc_hash_to_idx: Dict[str, int] = {}
        doc_tensors: List[torch.Tensor] = []

        for ex in examples:
            for doc_hash in ex.document_hashes:
                if doc_hash not in doc_hash_to_idx:
                    doc_hash_to_idx[doc_hash] = len(doc_tensors)
                    doc_tensors.append(ex.doc_token_ids[doc_hash])

        encoder_tokens = PackedSequences.from_tensors(doc_tensors, device)

        # Pack decoder tokens and labels
        dec_tensors = [ex.dec_token_ids for ex in examples]
        label_tensors = [ex.tok_labels for ex in examples]

        decoder_tokens = PackedSequences.from_tensors(dec_tensors, device)
        all_labels = torch.cat(label_tensors, dim=0).to(device)

        # Build doc index mapping per example
        example_doc_indices = [
            [doc_hash_to_idx[h] for h in ex.document_hashes]
            for ex in examples
        ]

        return cls(
            encoder_tokens=encoder_tokens,
            doc_hash_to_idx=doc_hash_to_idx,
            decoder_tokens=decoder_tokens,
            labels=all_labels,
            example_doc_indices=example_doc_indices,
        )

    def to(self, device: torch.device) -> "TokenizedBatch":
        """Move all tensors to device."""
        return TokenizedBatch(
            encoder_tokens=PackedSequences(
                tokens=self.encoder_tokens.tokens.to(device),
                cu_seqlens=self.encoder_tokens.cu_seqlens.to(device),
                lengths=self.encoder_tokens.lengths,
            ),
            doc_hash_to_idx=self.doc_hash_to_idx,
            decoder_tokens=PackedSequences(
                tokens=self.decoder_tokens.tokens.to(device),
                cu_seqlens=self.decoder_tokens.cu_seqlens.to(device),
                lengths=self.decoder_tokens.lengths,
            ),
            labels=self.labels.to(device),
            example_doc_indices=self.example_doc_indices,
        )


_DTYPE_TO_IDX = {torch.long: 0, torch.int32: 1, torch.float32: 2, torch.bfloat16: 3, torch.float16: 4}
_IDX_TO_DTYPE = {v: k for k, v in _DTYPE_TO_IDX.items()}


def broadcast_batch(
    batch: "TokenizedBatch",
    sp_group: "torch.distributed.ProcessGroup",
) -> "TokenizedBatch":
    """Broadcast a TokenizedBatch from sp_rank 0 to all ranks in sp_group.

    sp_rank 0 must have the full batch; other ranks receive it.
    """
    import torch.distributed as dist

    sp_rank = dist.get_rank(sp_group)
    device = torch.device("cuda")

    def _bcast_tensor(t: torch.Tensor, dtype: torch.dtype = torch.long) -> torch.Tensor:
        if sp_rank == 0:
            ndim = torch.tensor([t.ndim], dtype=torch.long, device=device)
            dtype_idx = torch.tensor([_DTYPE_TO_IDX.get(t.dtype, 0)], dtype=torch.long, device=device)
        else:
            ndim = torch.zeros(1, dtype=torch.long, device=device)
            dtype_idx = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(ndim, src=0, group=sp_group)
        dist.broadcast(dtype_idx, src=0, group=sp_group)
        actual_dtype = _IDX_TO_DTYPE[dtype_idx.item()]

        if sp_rank == 0:
            shape_tensor = torch.tensor(t.shape, dtype=torch.long, device=device)
        else:
            shape_tensor = torch.zeros(ndim.item(), dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0, group=sp_group)

        if sp_rank != 0:
            t = torch.empty(shape_tensor.tolist(), dtype=actual_dtype, device=device)
        else:
            t = t.to(device)
        dist.broadcast(t, src=0, group=sp_group)
        return t

    def _bcast_int_list(lst: list) -> list:
        t = torch.tensor(lst, dtype=torch.long, device=device) if sp_rank == 0 else None
        size = torch.tensor([len(lst)] if sp_rank == 0 else [0], dtype=torch.long, device=device)
        dist.broadcast(size, src=0, group=sp_group)
        if sp_rank != 0:
            t = torch.empty(size.item(), dtype=torch.long, device=device)
        dist.broadcast(t, src=0, group=sp_group)
        return t.tolist()

    # Broadcast encoder
    enc_tokens = _bcast_tensor(batch.encoder_tokens.tokens if sp_rank == 0 else torch.empty(0))
    enc_cu = _bcast_tensor(batch.encoder_tokens.cu_seqlens if sp_rank == 0 else torch.empty(0))
    enc_lengths = _bcast_int_list(batch.encoder_tokens.lengths if sp_rank == 0 else [])

    # Broadcast decoder
    dec_tokens = _bcast_tensor(batch.decoder_tokens.tokens if sp_rank == 0 else torch.empty(0))
    dec_cu = _bcast_tensor(batch.decoder_tokens.cu_seqlens if sp_rank == 0 else torch.empty(0))
    dec_lengths = _bcast_int_list(batch.decoder_tokens.lengths if sp_rank == 0 else [])

    # Broadcast labels
    labels = _bcast_tensor(batch.labels if sp_rank == 0 else torch.empty(0))

    # Broadcast example_doc_indices (list of lists of ints)
    # Flatten: [num_examples, max_docs_per_example] padded
    if sp_rank == 0:
        num_ex = len(batch.example_doc_indices)
        max_docs = max(len(d) for d in batch.example_doc_indices) if num_ex > 0 else 0
        flat = torch.full((num_ex, max_docs), -1, dtype=torch.long, device=device)
        for i, docs in enumerate(batch.example_doc_indices):
            flat[i, :len(docs)] = torch.tensor(docs, dtype=torch.long)
    else:
        num_ex = 0
        max_docs = 0

    meta = torch.tensor([num_ex, max_docs], dtype=torch.long, device=device)
    dist.broadcast(meta, src=0, group=sp_group)
    num_ex, max_docs = meta[0].item(), meta[1].item()

    if sp_rank != 0:
        flat = torch.empty(num_ex, max_docs, dtype=torch.long, device=device)
    dist.broadcast(flat, src=0, group=sp_group)

    example_doc_indices = []
    for i in range(num_ex):
        docs = flat[i][flat[i] >= 0].tolist()
        example_doc_indices.append(docs)

    # Broadcast doc_hash_to_idx (just indices, hashes not needed for forward)
    # We keep it empty on non-rank-0 since it's only used for dedup at packing time
    doc_hash_to_idx = batch.doc_hash_to_idx if sp_rank == 0 else {}

    return TokenizedBatch(
        encoder_tokens=PackedSequences(tokens=enc_tokens, cu_seqlens=enc_cu, lengths=enc_lengths),
        doc_hash_to_idx=doc_hash_to_idx,
        decoder_tokens=PackedSequences(tokens=dec_tokens, cu_seqlens=dec_cu, lengths=dec_lengths),
        labels=labels,
        example_doc_indices=example_doc_indices,
    )


def shard_batch(
    batch: "TokenizedBatch",
    sp_rank: int,
    sp_size: int,
) -> "TokenizedBatch":
    """Shard a TokenizedBatch by decoder examples for this sp_rank.

    Encoder tokens are kept in full (needed for cross-attention ring exchange).
    Decoder examples are split evenly across sp ranks.
    """
    num_examples = batch.batch_size
    # Divide examples across ranks
    examples_per_rank = (num_examples + sp_size - 1) // sp_size
    start = sp_rank * examples_per_rank
    end = min(start + examples_per_rank, num_examples)

    if start >= num_examples:
        # This rank has no examples (uneven split)
        device = batch.decoder_tokens.tokens.device
        return TokenizedBatch(
            encoder_tokens=batch.encoder_tokens,
            doc_hash_to_idx=batch.doc_hash_to_idx,
            decoder_tokens=PackedSequences(
                tokens=torch.empty(0, dtype=batch.decoder_tokens.tokens.dtype, device=device),
                cu_seqlens=torch.tensor([0], dtype=torch.int32, device=device),
                lengths=[],
            ),
            labels=torch.empty(0, dtype=batch.labels.dtype, device=device),
            example_doc_indices=[],
        )

    # Extract decoder shard
    dec_cu = batch.decoder_tokens.cu_seqlens
    dec_start = dec_cu[start].item()
    dec_end = dec_cu[end].item()

    shard_tokens = batch.decoder_tokens.tokens[dec_start:dec_end]
    shard_lengths = batch.decoder_tokens.lengths[start:end]
    shard_cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(shard_lengths), dim=0)),
        dtype=torch.int32,
        device=batch.decoder_tokens.tokens.device,
    )
    shard_labels = batch.labels[dec_start:dec_end]
    shard_doc_indices = batch.example_doc_indices[start:end]

    return TokenizedBatch(
        encoder_tokens=batch.encoder_tokens,  # full — needed for cross-attn ring
        doc_hash_to_idx=batch.doc_hash_to_idx,
        decoder_tokens=PackedSequences(tokens=shard_tokens, cu_seqlens=shard_cu, lengths=shard_lengths),
        labels=shard_labels,
        example_doc_indices=shard_doc_indices,
    )
