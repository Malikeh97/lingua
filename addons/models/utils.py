"""Model utilities."""

import torch
from transformers import AutoTokenizer

from .config import ModelArgs


def build_tokenizers(model_args: ModelArgs):
    """Build encoder and decoder tokenizers from model args.

    If decoder_name is empty, reuses the encoder tokenizer
    (from-scratch decoder shares encoder embeddings).
    """
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.encoder_name)

    if model_args.decoder_name:
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_args.decoder_name)
    else:
        decoder_tokenizer = encoder_tokenizer

    return encoder_tokenizer, decoder_tokenizer


def gather_encoder_states(
    encoder_output: torch.Tensor,
    enc_cu_seqlens: torch.Tensor,
    example_doc_indices: list,
) -> tuple:
    """Gather per-example encoder states from deduplicated encoder output.

    Args:
        encoder_output: [total_unique_enc_tokens, dim] deduplicated encoder states
        enc_cu_seqlens: [num_unique_docs + 1] cumulative lengths of unique docs
        example_doc_indices: list of list of doc indices per example

    Returns:
        (encoder_final_states, cu_seqlens_kv, max_seqlen_kv)
    """
    device = encoder_output.device
    enc_chunks = []
    enc_lengths = []
    for doc_indices in example_doc_indices:
        example_encs = []
        for idx in doc_indices:
            s = enc_cu_seqlens[idx].item()
            e = enc_cu_seqlens[idx + 1].item()
            example_encs.append(encoder_output[s:e])
        cat = torch.cat(example_encs, dim=0) if example_encs else encoder_output[:0]
        enc_chunks.append(cat)
        enc_lengths.append(cat.shape[0])

    encoder_final_states = torch.cat(enc_chunks, dim=0)
    cu_seqlens_kv = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(enc_lengths), dim=0)),
        dtype=torch.int32, device=device,
    )
    max_seqlen_kv = max(enc_lengths) if enc_lengths else 0
    return encoder_final_states, cu_seqlens_kv, max_seqlen_kv


def make_packed_causal_mask(
    cu_seqlens: torch.Tensor,
    total_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a block-diagonal causal mask from cu_seqlens.

    Returns [1, 1, total_len, total_len] additive mask for SDPA,
    where -inf blocks attention and 0 allows it. Each packed sequence
    can only attend causally within its own boundaries.
    """
    mask = torch.full(
        (total_len, total_len), float("-inf"), device=device, dtype=dtype,
    )
    num_seqs = len(cu_seqlens) - 1
    for i in range(num_seqs):
        s = cu_seqlens[i].item()
        e = cu_seqlens[i + 1].item()
        seq_len = e - s
        # Lower-triangular (causal) block for this sequence
        causal_block = torch.tril(
            torch.zeros(seq_len, seq_len, device=device, dtype=dtype),
        )
        # Upper triangle of block is already -inf from the full init
        mask[s:e, s:e] = causal_block
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
