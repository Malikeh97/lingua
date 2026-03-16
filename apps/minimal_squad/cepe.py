#!/usr/bin/env python3
"""
Unified script for generative QA on SQuAD with configurable data formats.
Initialization with pretrained models is implemented inspired by CEPE-style cross-attention adapters for efficient encoder-decoder adaptation: 
https://arxiv.org/pdf/2402.16617

Format Grammar:
- Q = Question, C = Context, A = Answer, S = Span
- / = concatenate in same sequence
- // = split (encoder // decoder)

Data formats (data_format):
- Q/C/A: Q+C+A in one sequence (decoder-only, LM loss on A)
- Q/C/S: Q+C in input, span extraction (decoder-only)
- C//Q/A: C in encoder, Q+A in decoder (enc-dec, LM loss on A)
- C/Q//S: C+Q in encoder, span extraction (enc-dec)

Span expression method (span_expr):
- none: No span extraction, generate answer text directly
- bertlike: Span head only, train extraction head, output "[SPAN]"
- first_last_hidden: Span head on encoder hidden states + generation anchors
- first_last_attn: Attention-based span extraction using decoder cross-attention

Model aliases (model_name)
- Encoder-decoder: deberta_v3_300m, deberta_v2_900m, modernbert_150m, modernbert_400m
- Decoder-only: tinyllama_1b, llama3.2_1b, llama3.2_3b, llama3.1_8b

Usage:
    python -m apps.encdec_for_squad --data_format "Q/C/A" --model_name tinyllama_1b
    python -m apps.encdec_for_squad --data_format "C//Q/A" --model_name deberta_v3_300m
    python -m apps.encdec_for_squad --data_format "C/Q//S" --span_expr bertlike --model_name modernbert_150m
    python -m apps.encdec_for_squad --data_format "C/Q//S" --span_expr first_last_hidden --model_name modernbert_150m
"""

import argparse
import re
import string
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import os as _os, sys as _sys
_fla_path = _os.path.join(_os.path.dirname(__file__), "flash-linear-attention")
if _fla_path not in _sys.path:
    _sys.path.insert(0, _fla_path)
try:
    from fla.ops.kda import chunk_kda as _chunk_kda
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False
print(f"[cepe] FLA available: {_FLA_AVAILABLE}", flush=True)


# ============== Data Preparation ==============


class DataFormat(Enum):
    """Data format options. Add new formats here as experiments progress."""

    DEC_QCA = "Q/C/A"  # Decoder-only: Q+C+A in one sequence
    DEC_QCS = "Q/C/S"  # Decoder-only: Q+C in input, span extraction
    DEC_CQA = "C/Q/A"  # Decoder-only: C+Q+A in one sequence
    ENCDEC_C_QA = "C//Q/A"  # Enc-dec: C in encoder, Q+A in decoder
    ENCDEC_C_QS = (
        "C//Q/S"  # Enc-dec: C in encoder, Q in encoder + span extraction in decoder
    )
    ENCDEC_CQ_A = "C/Q//A"  # Enc-dec: C+Q in encoder, answer in decoder
    ENCDEC_QC_A = "Q/C//A"  # Enc-dec: Q+C in encoder, answer in decoder
    ENCDEC_CQ_S = "C/Q//S"  # Enc-dec: C+Q in encoder, span extraction
    ENCDEC_QC_S = "Q/C//S"  # Enc-dec: Q+C in encoder, span extraction


DATA_FORMAT_CHOICES = [f.value for f in DataFormat]


@dataclass
class ParsedFormat:
    """Result of parsing a format string."""

    encoder_parts: List[str]  # Parts before // (or empty if no //)
    decoder_parts: List[str]  # Parts after // (or all parts if no //)
    has_split: bool  # Whether // exists
    is_span_extraction: bool  # Whether S is in the format, indicating span extraction


def parse_format(format_str: str) -> ParsedFormat:
    """
    Parse format string based on Q/C/A grammar.

    / = concatenate in same sequence
    // = split between encoder/decoder

    Examples:
        "Q/C/A" -> encoder=[], decoder=[Q,C,A], no split
        "C//Q/A" -> encoder=[C], decoder=[Q,A], has split
        "C/Q//A" -> encoder=[C,Q], decoder=[A], has split (generation)
        "C/Q//S" -> encoder=[C,Q], decoder=[S], span extraction
    """
    is_span = "S" in format_str
    has_split = "//" in format_str
    if has_split:
        enc_str, dec_str = format_str.split("//", 1)
        encoder_parts = [p.strip() for p in enc_str.split("/") if p.strip()]
        decoder_parts = [p.strip() for p in dec_str.split("/") if p.strip()]
    else:
        encoder_parts = []
        decoder_parts = [p.strip() for p in format_str.split("/") if p.strip()]

    return ParsedFormat(
        encoder_parts=encoder_parts,
        decoder_parts=decoder_parts,
        has_split=has_split,
        is_span_extraction=is_span,
    )


def get_part_text(part: str, sample: Dict) -> str:
    """Get text for a format part (Q, C, or A)."""
    if part == "Q":
        return sample["question"].strip()
    elif part == "C":
        return sample["context"]
    elif part == "A":
        answers = sample["answers"]
        return answers["text"][0] if answers["text"] else ""
    elif part == "S":
        return "[SPAN]"
    else:
        raise ValueError(f"Unknown part: {part}")


def build_sequence(parts: List[str], sample: Dict, separator: str = " ") -> str:
    """Build a text sequence from format parts."""
    texts = [get_part_text(p, sample) for p in parts]
    return separator.join(texts)


class DataPreparer:
    """
    Generic data preparation based on parsed format.

    Architecture:
    1. Tokenize Q, C, A segments separately (no special tokens)
    2. Find span positions within C (if span extraction)
    3. Assemble according to format, adding special tokens
    4. Update span positions with assembly offset
    5. Create labels based on which segments should have loss
    """

    def __init__(
        self,
        tokenizer,
        parsed_format: ParsedFormat,
        max_length: int = 512,
        dec_max_length: int = 128,
        span_expr: str = "none",
        decoder_tokenizer=None,
    ):
        self.tokenizer = tokenizer  # encoder tokenizer (or shared)
        self.decoder_tokenizer = decoder_tokenizer or tokenizer
        self.has_dual_tokenizer = decoder_tokenizer is not None
        self.fmt = parsed_format
        self.max_length = max_length
        self.dec_max_length = dec_max_length
        self.span_expr = span_expr

        # Cache special token IDs for encoder tokenizer
        self.bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
        self.eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

        # Cache special token IDs for decoder tokenizer
        self.dec_bos_id = self.decoder_tokenizer.bos_token_id or self.decoder_tokenizer.cls_token_id
        self.dec_eos_id = self.decoder_tokenizer.eos_token_id or self.decoder_tokenizer.sep_token_id

        # Part prefixes (boilerplate) - these never have loss
        # Note: Q/C/A are just format notation, prefixes are human-readable
        self.part_prefixes = {
            "Q": "Question: ",
            "C": " Context: ",
            "A": " Answer: ",
            "S": "",  # [SPAN] is handled specially
        }

    def _tokenize_prefix(self, prefix: str) -> List[int]:
        """Tokenize a prefix string without special tokens."""
        if not prefix:
            return []
        return self.tokenizer(prefix, add_special_tokens=False, return_tensors=None)[
            "input_ids"
        ]

    def _tokenize_segments(self, sample: Dict) -> Dict[str, Dict]:
        """Tokenize Q, C, A segments separately without special tokens.

        When a separate decoder_tokenizer exists, also tokenizes with it
        and stores results as Q_dec, C_dec, A_dec.
        """
        segments = {}

        # Question
        segments["Q"] = self.tokenizer(
            sample["question"].strip(),
            add_special_tokens=False,
            return_tensors=None,
        )

        # Context (with offset mapping for span extraction)
        segments["C"] = self.tokenizer(
            sample["context"],
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors=None,
        )

        # Answer
        answers = sample["answers"]
        a_text = answers["text"][0] if answers["text"] else ""
        segments["A"] = self.tokenizer(
            a_text,
            add_special_tokens=False,
            return_tensors=None,
        )

        # Dual tokenizer: also tokenize with decoder tokenizer
        if self.has_dual_tokenizer:
            segments["Q_dec"] = self.decoder_tokenizer(
                sample["question"].strip(),
                add_special_tokens=False,
                return_tensors=None,
            )
            segments["C_dec"] = self.decoder_tokenizer(
                sample["context"],
                add_special_tokens=False,
                return_tensors=None,
            )
            segments["A_dec"] = self.decoder_tokenizer(
                a_text,
                add_special_tokens=False,
                return_tensors=None,
            )

        return segments

    def _find_span_in_context(
        self, context_segment: Dict, answers: Dict
    ) -> Tuple[int, int]:
        """Find answer span positions within context tokens (0-indexed within C)."""
        answer_text = answers["text"][0] if answers["text"] else ""
        answer_start = answers["answer_start"][0] if answers["answer_start"] else 0
        answer_end = answer_start + len(answer_text)

        offsets = context_segment["offset_mapping"]
        start_tok, end_tok = 0, 0

        for idx, (s, e) in enumerate(offsets):
            if s <= answer_start < e:
                start_tok = idx
            if s < answer_end <= e:
                end_tok = idx
                break

        return start_tok, end_tok

    def _assemble_sequence(
        self,
        segments: Dict[str, Dict],
        parts: List[str],
        loss_parts: List[str],
        max_length: int,
        span_in_context: Optional[Tuple[int, int]] = None,
        side: str = "encoder",
    ) -> Dict:
        """
        Assemble segments into a sequence according to parts order.

        Args:
            segments: Tokenized segments {Q, C, A} (and Q_dec, C_dec, A_dec if dual tokenizer)
            parts: Order of parts to assemble (e.g., ["C", "Q"] or ["Q", "A"])
            loss_parts: Which parts should have loss (e.g., ["A"])
            max_length: Maximum sequence length
            span_in_context: (start, end) positions within C segment (0-indexed)
            side: "encoder" or "decoder" - selects which tokenizer's tokens to use

        Returns:
            Dict with input_ids, attention_mask, labels, and optionally
            start_positions, end_positions (adjusted for assembly offset)
        """
        # Select tokenizer, BOS/EOS based on side
        use_dec = side == "decoder" and self.has_dual_tokenizer
        tok = self.decoder_tokenizer if use_dec else self.tokenizer
        bos = self.dec_bos_id if use_dec else self.bos_id
        eos = self.dec_eos_id if use_dec else self.eos_id
        seg_suffix = "_dec" if use_dec else ""

        input_ids = [bos]
        labels = [-100]
        context_offset = None  # Track where C starts in assembled sequence

        for part in parts:
            if part == "S":
                # Special [SPAN] token for span extraction output
                span_id = tok.convert_tokens_to_ids("[SPAN]")
                input_ids.append(span_id)
                labels.append(span_id if "S" in loss_parts else -100)

                # For first_last modes, add anchor tokens after [SPAN]
                # Use actual token IDs from context to avoid tokenization mismatch
                # Format: first_token ... last_token (always use ellipsis for consistent parsing)
                if (
                    self.span_expr.startswith("first_last")
                    and span_in_context is not None
                ):
                    c_ids = segments["C"]["input_ids"]
                    start_pos, end_pos = span_in_context
                    first_token_id = c_ids[start_pos]
                    last_token_id = c_ids[end_pos]

                    ellipsis_ids = tok.encode(" ...", add_special_tokens=False)
                    anchor_ids = [first_token_id] + ellipsis_ids + [last_token_id]

                    input_ids.extend(anchor_ids)
                    labels.extend(
                        anchor_ids if "S" in loss_parts else [-100] * len(anchor_ids)
                    )
            else:
                # Add prefix (boilerplate) - never has loss
                prefix = self.part_prefixes.get(part, "")
                if prefix:
                    prefix_ids = tok(
                        prefix, add_special_tokens=False, return_tensors=None
                    )["input_ids"]
                else:
                    prefix_ids = []
                if prefix_ids:
                    input_ids.extend(prefix_ids)
                    labels.extend([-100] * len(prefix_ids))

                # Track context offset AFTER prefix, before content
                if part == "C":
                    context_offset = len(input_ids)

                # Add content (use decoder-tokenized version if on decoder side)
                seg_key = part + seg_suffix
                part_ids = segments.get(seg_key, segments[part])["input_ids"]
                input_ids.extend(part_ids)

                if part in loss_parts:
                    labels.extend(part_ids)
                else:
                    labels.extend([-100] * len(part_ids))

        # Add EOS
        input_ids.append(eos)
        labels.append(eos if loss_parts else -100)

        # Truncate
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

        # Adjust span positions if C was included
        if span_in_context is not None and context_offset is not None:
            result["start_positions"] = span_in_context[0] + context_offset
            result["end_positions"] = span_in_context[1] + context_offset

        return result

    def prepare_features(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Prepare features for all samples.

        Unified pipeline:
        1. Tokenize segments (Q, C, A)
        2. Find span positions if needed
        3. Assemble encoder sequence (if enc-dec)
        4. Assemble decoder sequence (or single sequence for dec-only)
        """
        num_samples = len(examples["question"])

        # Determine loss parts based on format
        if self.fmt.is_span_extraction:
            loss_parts = ["S"]  # Loss on span output
        else:
            loss_parts = ["A"]  # Loss on answer

        # Initialize result containers
        if self.fmt.has_split:
            results = {
                "encoder_input_ids": [],
                "encoder_attention_mask": [],
                "decoder_input_ids": [],
                "labels": [],
            }
        else:
            results = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        if self.fmt.is_span_extraction:
            results["start_positions"] = []
            results["end_positions"] = []

        for i in range(num_samples):
            sample = {k: examples[k][i] for k in examples}

            # Step 1: Tokenize all segments
            segments = self._tokenize_segments(sample)

            # Step 2: Find span positions within C (if span extraction)
            span_in_context = None
            if self.fmt.is_span_extraction:
                span_in_context = self._find_span_in_context(
                    segments["C"], sample["answers"]
                )

            # Step 3: Assemble according to format
            if self.fmt.has_split:
                # Encoder-decoder: assemble encoder and decoder separately
                enc_result = self._assemble_sequence(
                    segments,
                    self.fmt.encoder_parts,
                    loss_parts=[],  # No loss on encoder
                    max_length=self.max_length,
                    span_in_context=span_in_context,
                    side="encoder",
                )
                dec_result = self._assemble_sequence(
                    segments,
                    self.fmt.decoder_parts,
                    loss_parts=loss_parts,
                    max_length=self.dec_max_length,
                    span_in_context=span_in_context,
                    side="decoder",
                )

                results["encoder_input_ids"].append(enc_result["input_ids"])
                results["encoder_attention_mask"].append(enc_result["attention_mask"])
                results["decoder_input_ids"].append(dec_result["input_ids"])
                results["labels"].append(dec_result["labels"])

                # Span positions are relative to encoder (where C is)
                if self.fmt.is_span_extraction:
                    results["start_positions"].append(
                        enc_result.get("start_positions", 0)
                    )
                    results["end_positions"].append(enc_result.get("end_positions", 0))
            else:
                # Decoder-only: single sequence
                result = self._assemble_sequence(
                    segments,
                    self.fmt.decoder_parts,
                    loss_parts=loss_parts,
                    max_length=self.max_length,
                    span_in_context=span_in_context,
                )

                results["input_ids"].append(result["input_ids"])
                results["attention_mask"].append(result["attention_mask"])
                results["labels"].append(result["labels"])

                if self.fmt.is_span_extraction:
                    results["start_positions"].append(result.get("start_positions", 0))
                    results["end_positions"].append(result.get("end_positions", 0))

        return results


def get_collate_fn(
    pad_token_id: int, padding_side: str = "right", dec_pad_token_id: int = None
):
    """Collate function with dynamic padding, determines handling based on batch keys.

    Args:
        pad_token_id: Token ID to use for padding (encoder side)
        padding_side: "left" or "right" - where to add padding
            - "right": Standard for encoder models and span extraction
            - "left": Preferred for decoder-only causal LM batch generation
        dec_pad_token_id: Token ID for decoder padding (if different from encoder)
    """
    dec_pad = dec_pad_token_id if dec_pad_token_id is not None else pad_token_id

    def pad_seq(seqs, pad_value):
        max_len = max(len(s) for s in seqs)
        if padding_side == "left":
            return [[pad_value] * (max_len - len(s)) + s for s in seqs]
        else:  # right
            return [s + [pad_value] * (max_len - len(s)) for s in seqs]

    def collate(batch):
        keys = batch[0].keys()
        result = {}

        # Encoder-decoder mode
        if "encoder_input_ids" in keys:
            result["encoder_input_ids"] = torch.tensor(
                pad_seq([x["encoder_input_ids"] for x in batch], pad_token_id)
            )
            result["encoder_attention_mask"] = torch.tensor(
                pad_seq([x["encoder_attention_mask"] for x in batch], 0)
            )
        if "decoder_input_ids" in keys:
            result["decoder_input_ids"] = torch.tensor(
                pad_seq([x["decoder_input_ids"] for x in batch], dec_pad)
            )
        if "labels" in keys:
            result["labels"] = torch.tensor(pad_seq([x["labels"] for x in batch], -100))
        # Span extraction or decoder-only mode
        if "input_ids" in keys:
            result["input_ids"] = torch.tensor(
                pad_seq([x["input_ids"] for x in batch], pad_token_id)
            )
            result["attention_mask"] = torch.tensor(
                pad_seq([x["attention_mask"] for x in batch], 0)
            )

        # Span extraction
        if "start_positions" in keys:
            result["start_positions"] = torch.tensor(
                [x["start_positions"] for x in batch]
            )
            result["end_positions"] = torch.tensor([x["end_positions"] for x in batch])

        return result

    return collate


# ============== Model Aliases ==============


ENCDEC_ALIASES = {
    "deberta_v3_300m": "microsoft/deberta-v3-large",
    "deberta_v2_900m": "microsoft/deberta-v2-xlarge",
    "deberta_v2_1.5b": "microsoft/deberta-v2-xxlarge",
    "modernbert_150m": "answerdotai/ModernBERT-base",
    "modernbert_400m": "answerdotai/ModernBERT-large",
}

DEC_ALIASES = {
    "tinyllama_1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B",
    "llama3.1_8b": "meta-llama/Llama-3.1-8B",
}

MODEL_ALIASES = {**ENCDEC_ALIASES, **DEC_ALIASES}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def get_encoder_embeddings(encoder, model_name: str) -> torch.Tensor:
    model_name_lower = model_name.lower()
    if "deberta" in model_name_lower:
        return encoder.embeddings.word_embeddings.weight
    elif "modernbert" in model_name_lower:
        return encoder.embeddings.tok_embeddings.weight
    elif "bert" in model_name_lower or "roberta" in model_name_lower:
        return encoder.embeddings.word_embeddings.weight
    else:
        if hasattr(encoder, "embeddings"):
            if hasattr(encoder.embeddings, "word_embeddings"):
                return encoder.embeddings.word_embeddings.weight
            elif hasattr(encoder.embeddings, "tok_embeddings"):
                return encoder.embeddings.tok_embeddings.weight
        raise ValueError(f"Cannot find embeddings for model: {model_name}")


def infer_special_tokens(tokenizer):
    """
    Infer bos_token and eos_token from tokenizer's behavior if they are missing.
    Some tokenizers (like ModernBERT) don't explicitly set eos_token but add a separator.
    """
    # Only infer if missing
    if tokenizer.bos_token is not None and tokenizer.eos_token is not None:
        return

    # Check behavior with add_special_tokens=True
    ids_with_special = tokenizer.encode("a", add_special_tokens=True)
    ids_without_special = tokenizer.encode("a", add_special_tokens=False)

    # Find the sequence "a" inside the special sequence
    seq = ids_with_special
    sub = ids_without_special
    n = len(seq)
    m = len(sub)

    match_index = -1
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            match_index = i
            break

    if match_index != -1:
        # Check for Prefix (BOS/CLS)
        if match_index > 0:
            bos_id = seq[0]
            token_str = tokenizer.decode([bos_id], skip_special_tokens=False)
            if tokenizer.bos_token is None:
                tokenizer.bos_token = token_str
                print(f"Inferred BOS token: {tokenizer.bos_token} (id: {bos_id})")
            if tokenizer.cls_token is None:
                tokenizer.cls_token = token_str
                print(f"Inferred CLS token: {tokenizer.cls_token} (id: {bos_id})")

        # Check for Suffix (EOS/SEP)
        if match_index + m < n:
            eos_id = seq[-1]
            token_str = tokenizer.decode([eos_id], skip_special_tokens=False)
            if tokenizer.eos_token is None:
                tokenizer.eos_token = token_str
                print(f"Inferred EOS token: {tokenizer.eos_token} (id: {eos_id})")
            if tokenizer.sep_token is None:
                tokenizer.sep_token = token_str
                print(f"Inferred SEP token: {tokenizer.sep_token} (id: {eos_id})")

    # Infer PAD from EOS if missing
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Inferred PAD token from EOS: {tokenizer.pad_token}")
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
            print(f"Inferred PAD token from SEP: {tokenizer.pad_token}")


# ============== Models ==============


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x,
        encoder_output,
        causal_mask=None,
        encoder_padding_mask=None,
        return_attn=False,
    ):
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x, cross_attn_weights = self.cross_attn(
            x,
            encoder_output,
            encoder_output,
            key_padding_mask=(
                ~encoder_padding_mask if encoder_padding_mask is not None else None
            ),
            need_weights=return_attn,
            average_attn_weights=False,  # Return per-head weights: (batch, num_heads, tgt_len, src_len)
        )
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        if return_attn:
            return x, cross_attn_weights
        return x

    def init_weights(self, init_std: float, factor: float):
        """Lingua-style init: truncated normal, output projections scaled by factor."""
        out_std = init_std / factor
        ffn_out_std = (self.ffn[0].out_features ** -0.5) / factor

        def tn(w, std):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

        # Attention Q/K/V and output projections
        for attn in [self.self_attn, self.cross_attn]:
            if attn.in_proj_weight is not None:
                tn(attn.in_proj_weight, init_std)
            tn(attn.out_proj.weight, out_std)

        # FFN
        tn(self.ffn[0].weight, init_std)
        tn(self.ffn[3].weight, ffn_out_std)

        # LayerNorms to ones
        for norm in [self.self_attn_norm, self.cross_attn_norm, self.ffn_norm]:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)


class CrossAttentionAdapter(nn.Module):
    """Cross-attention adapter injected between self-attention and FFN of a frozen decoder layer.

    Architecture: LayerNorm -> MultiheadAttention(Q=decoder, K/V=encoder) -> residual.
    Output projection zero-initialized so the adapter starts as a no-op, preserving
    the pre-trained decoder's behavior. The model gradually learns to use encoder info.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        # Zero-init output projection so adapter starts as identity
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, x, encoder_output, encoder_padding_mask=None, return_attn=False):
        residual = x
        x = self.cross_attn_norm(x)
        x, attn_weights = self.cross_attn(
            x,
            encoder_output,
            encoder_output,
            key_padding_mask=(
                ~encoder_padding_mask if encoder_padding_mask is not None else None
            ),
            need_weights=return_attn,
            average_attn_weights=False,
        )
        output = residual + x
        if return_attn:
            return output, attn_weights
        return output

    def init_weights(self, init_std: float, factor: float):
        """Init Q/K/V with truncated normal, output proj stays zero."""

        def tn(w, std):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

        if self.cross_attn.in_proj_weight is not None:
            tn(self.cross_attn.in_proj_weight, init_std)
        # Output proj: zero (CEPE-style identity start)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        # LayerNorm
        nn.init.ones_(self.cross_attn_norm.weight)
        nn.init.zeros_(self.cross_attn_norm.bias)


class LinearCrossAttentionAdapter(nn.Module):
    """Linear-complexity cross-attention adapter for CEPE.

    Supports two variants:
      - "linear": Parallel S = K^T V encoding (O(enc_len) memory state).
      - "linear_kda": Sequential KDA-style delta-rule recurrence over encoder tokens.

    Both variants zero-init out_proj.weight so the adapter starts as a no-op,
    preserving the pre-trained decoder's behavior at initialization.
    Returns the same API as CrossAttentionAdapter.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        variant: str = "linear",
    ):
        super().__init__()
        assert variant in ("linear", "linear_kda"), f"Unknown variant: {variant!r}"
        self.variant = variant
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        if variant == "linear_kda":
            # FLA-style: direct D → H*d_k projection for per-element log-space decay gates.
            # logsigmoid output is always ≤ 0, so exp(g) ≤ 1 (guaranteed decay).
            # Replaces the old rank-bottleneck alpha_down/alpha_up pair.
            self.g_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
            self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)
            # Zero-init: logsigmoid(0) = -log2 ≈ -0.693 → moderate initial decay rate
            nn.init.zeros_(self.g_proj.weight)
            nn.init.zeros_(self.g_proj.bias)

        # Zero-init output projection: adapter starts as identity (no-op)
        nn.init.zeros_(self.out_proj.weight)

    def _feature_map(self, x):
        """Swish activation followed by L2 normalization along head_dim."""
        x = torch.nn.functional.silu(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def forward(self, x, encoder_output, encoder_padding_mask=None, return_attn=False):
        """
        Args:
            x: decoder hidden states [B, T_dec, D]
            encoder_output: encoder hidden states [B, T_enc, D]
            encoder_padding_mask: bool tensor [B, T_enc], True = valid token
            return_attn: if True, returns (output, None) — no explicit weights for linear
        """
        residual = x
        x = self.cross_attn_norm(x)

        B, T_dec, D = x.shape
        T_enc = encoder_output.shape[1]
        H = self.num_heads
        d_k = self.head_dim

        # Project Q, K, V and reshape to (B, H, T, d_k)
        Q = self.q_proj(x).view(B, T_dec, H, d_k).transpose(1, 2)           # (B, H, T_dec, d_k)
        K = self.k_proj(encoder_output).view(B, T_enc, H, d_k).transpose(1, 2)  # (B, H, T_enc, d_k)
        V = self.v_proj(encoder_output).view(B, T_enc, H, d_k).transpose(1, 2)  # (B, H, T_enc, d_k)

        # Apply feature maps
        Q = self._feature_map(Q)   # (B, H, T_dec, d_k)
        K = self._feature_map(K)   # (B, H, T_enc, d_k)
        V = torch.nn.functional.silu(V)  # (B, H, T_enc, d_k)

        if self.variant == "linear":
            # Zero-out padded encoder positions before accumulation
            if encoder_padding_mask is not None:
                # mask: (B, 1, T_enc, 1) — True=valid
                mask = encoder_padding_mask[:, None, :, None].to(K.dtype)
                K = K * mask
                V = V * mask

            # S = K^T @ V  (B, H, d_k, d_k)
            S = torch.matmul(K.transpose(-2, -1), V)
            # z = sum of K over enc positions (B, H, d_k)
            z = K.sum(dim=2)

            # O = Q @ S  (B, H, T_dec, d_k)
            O = torch.matmul(Q, S)
            # Normalize
            denom = torch.matmul(Q, z.unsqueeze(-1)).clamp(min=1e-6)  # (B, H, T_dec, 1)
            O = O / denom

        else:  # linear_kda: chunked delta-rule recurrence (inspired by FLA naive_chunk_kda)
            #
            # Gate design (FLA convention):
            #   g_log = logsigmoid(g_proj(enc))  → per-element, log-space, always ≤ 0
            #   exp(g_log) ≤ 1 everywhere → guaranteed exponential decay
            #   This replaces the old sigmoid-bottleneck alpha_down/alpha_up.
            #
            # Recurrence design (replaces TBPTT):
            #   Within each chunk of BT=64 tokens: exact delta-rule via an intra-chunk
            #   interaction matrix A and forward substitution (O(BT^2) ops, no Python loop
            #   over tokens).  Across NT = T_enc/64 chunks: sequential S propagation.
            #   Autograd graph depth = O(NT) ≈ 8 instead of O(T_enc) ≈ 512.
            #   Memory: O(num_adapters * BT^2) instead of O(num_adapters * T_enc * d_k^2).
            #   Gradients are EXACT (no truncation).

            # Log-space per-element decay gates: [B, T_enc, H*d_k] → [B, H, T_enc, d_k]
            g_log = F.logsigmoid(self.g_proj(encoder_output))
            g_log = g_log.view(B, T_enc, H, d_k).permute(0, 2, 1, 3)   # [B, H, T_enc, d_k]

            # Per-head write gate: [B, T_enc, H] → [B, H, T_enc]
            beta_val = torch.sigmoid(self.beta_proj(encoder_output))
            beta_val = beta_val.permute(0, 2, 1)                         # [B, H, T_enc]

            # Zero-out padded encoder positions before any accumulation
            if encoder_padding_mask is not None:
                pad_m = encoder_padding_mask[:, None, :, None].to(K.dtype)  # [B, 1, T_enc, 1]
                K = K * pad_m
                V = V * pad_m

            if _FLA_AVAILABLE:
                # ── Fast path: single fused CUDA kernel via flash-linear-attention ──
                # Convert [B, H, T, d_k] → [B, T, H, d_k] as expected by chunk_kda.
                K_enc = K.permute(0, 2, 1, 3).contiguous()       # [B, T_enc, H, d_k]
                V_enc = V.permute(0, 2, 1, 3).contiguous()       # [B, T_enc, H, d_k]
                g_enc = g_log.permute(0, 2, 1, 3).contiguous()   # [B, T_enc, H, d_k]
                b_enc = beta_val.permute(0, 2, 1).contiguous()   # [B, T_enc, H]

                # chunk_kda requires float32 initial state
                h0 = K.new_zeros(B, H, d_k, d_k, dtype=torch.float32)

                # Q is only used to produce the intra-encoder output (which we discard);
                # pass K_enc as a shape-compatible placeholder.
                # scale=1.0: Q and K are already L2-normalised by _feature_map.
                _, final_state = _chunk_kda(
                    q=K_enc,
                    k=K_enc,
                    v=V_enc,
                    g=g_enc,
                    beta=b_enc,
                    scale=1.0,
                    initial_state=h0,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    disable_recompute=True,   # save intermediates → faster backward
                    safe_gate=True,           # enable M=16 TensorCore path
                    lower_bound=-5.0,         # logsigmoid gates are always < 0, -5 is safe
                )
                # final_state: [B, H, d_k, d_k] (float32) — cast to match Q dtype
                S = final_state.to(Q.dtype)

            else:
                # ── Fallback path: Python outer loop + batched triangular solve ──
                # Inner forward-substitution loop replaced by a single
                # torch.linalg.solve_triangular call (one batched CUDA LAPACK kernel).
                CHUNK = 64
                S = K.new_zeros(B, H, d_k, d_k)

                for t_start in range(0, T_enc, CHUNK):
                    t_end = min(t_start + CHUNK, T_enc)
                    BT = t_end - t_start

                    K_c = K[:, :, t_start:t_end, :]        # [B, H, BT, d_k]
                    V_c = V[:, :, t_start:t_end, :]        # [B, H, BT, d_k]
                    g_c = g_log[:, :, t_start:t_end, :]    # [B, H, BT, d_k]
                    b_c = beta_val[:, :, t_start:t_end]    # [B, H, BT]

                    # Intra-chunk cumulative gate
                    g_cum = g_c.cumsum(dim=2)              # [B, H, BT, d_k]
                    K_g = K_c * g_cum.exp()                # [B, H, BT, d_k]

                    # Intra-chunk delta-rule interaction matrix (lower triangular)
                    A = torch.matmul(K_c, K_g.transpose(-2, -1))   # [B, H, BT, BT]
                    A = A * b_c.unsqueeze(-1)

                    upper = torch.triu(
                        torch.ones(BT, BT, dtype=torch.bool, device=K.device), diagonal=0
                    )
                    A = (-A).masked_fill(upper[None, None], 0.0)

                    # Exact (I - A)^{-1} via batched triangular solve — replaces the
                    # 63-iteration Python forward-substitution loop.
                    eye_BT = torch.eye(BT, device=K.device, dtype=K.dtype)
                    # solve_triangular doesn't support bfloat16; cast to float32 and back
                    _A = A.float()
                    _eye = eye_BT.float()
                    A = torch.linalg.solve_triangular(
                        _eye[None, None] - _A,
                        _eye[None, None].expand(B, H, -1, -1),
                        upper=False,
                    ).to(K.dtype) * b_c.unsqueeze(-1)

                    # Inter-chunk aggregates
                    w = torch.matmul(A, K_g)
                    u = torch.matmul(A, V_c)
                    v_new = u - torch.matmul(w, S)

                    g_last = g_cum[:, :, -1, :]
                    S = S * g_last.exp().unsqueeze(-1)
                    K_w = K_c * (g_last[:, :, None, :] - g_cum).exp()
                    S = S + torch.matmul(K_w.transpose(-2, -1), v_new)

            # Decode: all decoder queries read from the final encoder state.
            # Q and K are L2-normalised by _feature_map, providing implicit normalisation.
            O = torch.matmul(Q, S)                      # [B, H, T_dec, d_k]

        # Merge heads and project
        O = O.transpose(1, 2).contiguous().view(B, T_dec, D)  # (B, T_dec, D)
        O = self.dropout(O)
        O = self.out_proj(O)

        output = residual + O
        if return_attn:
            return output, None
        return output

    def init_weights(self, init_std: float, factor: float):
        """Init Q/K/V with trunc_normal; out_proj zero; KDA gates zero; LayerNorm ones/zeros."""

        def tn(w, std):
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

        tn(self.q_proj.weight, init_std)
        tn(self.k_proj.weight, init_std)
        tn(self.v_proj.weight, init_std)
        # Output proj: zero (CEPE-style identity start)
        nn.init.zeros_(self.out_proj.weight)

        if self.variant == "linear_kda":
            # g_proj already zero-inited in __init__; reinforce here for robustness
            nn.init.zeros_(self.g_proj.weight)
            nn.init.zeros_(self.g_proj.bias)
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

        nn.init.ones_(self.cross_attn_norm.weight)
        nn.init.zeros_(self.cross_attn_norm.bias)


class AttentionSpanHead(nn.Module):
    """Learns to extract span positions from cross-attention weights.

    Combines attention scores across layers and heads when generating
    "first" and "last" anchor tokens to predict span start/end positions.
    """

    def __init__(self, num_layers: int, num_heads: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        # Learned weights to combine attention from different layers/heads
        # Shape: (num_layers, num_heads) for start and end separately
        self.start_layer_weights = nn.Parameter(torch.ones(num_layers, num_heads))
        self.end_layer_weights = nn.Parameter(torch.ones(num_layers, num_heads))

    def forward(
        self,
        cross_attn_weights: List[torch.Tensor],
        first_token_pos: int,
        last_token_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute span logits from cross-attention weights.

        Args:
            cross_attn_weights: List of (batch, num_heads, tgt_len, src_len) tensors,
                one per decoder layer.
            first_token_pos: Position in decoder sequence where "first" anchor is generated.
            last_token_pos: Position in decoder sequence where "last" anchor is generated.

        Returns:
            start_logits: (batch, src_len) - logits for span start position
            end_logits: (batch, src_len) - logits for span end position
        """
        # Stack attention weights: (num_layers, batch, num_heads, tgt_len, src_len)
        stacked = torch.stack(cross_attn_weights, dim=0)

        # Extract attention at first_token_pos for start prediction
        # Shape: (num_layers, batch, num_heads, src_len)
        first_attn = stacked[:, :, :, first_token_pos, :]

        # Extract attention at last_token_pos for end prediction
        last_attn = stacked[:, :, :, last_token_pos, :]

        # Apply learned weights and sum across layers/heads
        # Use raw weights (no softmax) to allow sharp distributions in output logits
        start_weights = self.start_layer_weights.view(
            self.num_layers, self.num_heads, 1
        )
        end_weights = self.end_layer_weights.view(self.num_layers, self.num_heads, 1)

        # Weighted sum: (batch, src_len)
        start_logits = (first_attn * start_weights).sum(dim=(0, 2))
        end_logits = (last_attn * end_weights).sum(dim=(0, 2))

        return start_logits, end_logits


class UnifiedModel(nn.Module):
    """Unified model for Causal LM, Enc-Dec Generation, and Span Extraction.

    Compositional design:
    - model_type: "dec" (decoder-only) or "encdec" (encoder-decoder)
    - span_expr:
        - "none": generation only
        - "bertlike": span head only
        - "first_last_hidden": span head on hidden states + generation
        - "first_last_attn": attention-based span + generation (encdec only)
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        span_expr: str = "none",
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        span_loss_weight: float = 1.0,
        vocab_size: int = None,
        enc_local_layer_ratio: float = 0.0,
        sep_token_id: int = None,
        decoder_model_name: str = None,
        cross_attn_num_heads: int = 16,
        cross_attn_type: str = "softmax",
    ):
        """
        Args:
            enc_local_layer_ratio: Fraction of bottom encoder layers that use local attention
                (no attending across SEP boundaries). 0.0 = all global (default), 1.0 = all local.
                E.g., 0.5 with 22 layers means bottom 11 layers are local.
            sep_token_id: Token ID for [SEP]. Required if enc_local_layer_ratio > 0.
            decoder_model_name: Pre-trained model to use as decoder (e.g., 'tinyllama_1b').
                When set with model_type='encdec', uses pre-trained decoder with cross-attention adapters.
            cross_attn_num_heads: Number of attention heads for cross-attention adapters.
            cross_attn_type: Cross-attention adapter type: 'softmax' (default), 'linear', or 'linear_kda'.
        """
        super().__init__()
        model_name = resolve_model_name(model_name)
        self.model_name = model_name
        self.model_type = model_type
        self.span_expr = span_expr
        self.span_loss_weight = span_loss_weight
        self.use_pretrained_decoder = bool(decoder_model_name)
        self.enc_local_layer_ratio = enc_local_layer_ratio
        self.sep_token_id = sep_token_id
        self.cross_attn_type = cross_attn_type

        if cross_attn_type != "softmax" and span_expr == "first_last_attn":
            raise AssertionError(
                f"cross_attn_type={cross_attn_type!r} is incompatible with "
                "span_expr='first_last_attn' (linear attention produces no explicit "
                "attention weight matrices). Use span_expr='none', 'bertlike', or "
                "'first_last_hidden'."
            )

        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        num_heads = config.num_attention_heads

        # Base model: CausalLM for decoder-only, AutoModel for encoder-decoder
        if model_type == "dec":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, torch_dtype=torch.bfloat16
            )
        else:
            self.base_model = AutoModel.from_pretrained(
                model_name, config=config, torch_dtype=torch.bfloat16
            )

        # Calculate local attention layer threshold for encoder
        self.num_encoder_layers = config.num_hidden_layers
        self.enc_local_layers = int(enc_local_layer_ratio * self.num_encoder_layers)
        if self.enc_local_layers > 0 and model_type == "encdec":
            print(
                f"Encoder local attention: bottom {self.enc_local_layers}/{self.num_encoder_layers} layers"
            )

        # Custom decoder for enc-dec generation (not needed for bertlike-only)
        if model_type == "encdec" and span_expr != "bertlike":
            if decoder_model_name:
                self._init_pretrained_decoder(
                    decoder_model_name, cross_attn_num_heads, dropout, cross_attn_type
                )
            else:
                self._init_decoder(num_decoder_layers, num_heads, dropout, max_seq_len)

        # Resize embeddings if vocab_size differs (e.g., [SPAN] token added)
        if vocab_size is not None and vocab_size != self.vocab_size:
            self.resize_embeddings(vocab_size)

        # Span head based on span_expr mode
        if span_expr in ("bertlike", "first_last_hidden"):
            self.qa_outputs = nn.Linear(self.hidden_size, 2)
        elif span_expr == "first_last_attn":
            if model_type != "encdec":
                raise ValueError("first_last_attn requires model_type='encdec'")
            self.attn_span_head = AttentionSpanHead(num_decoder_layers, num_heads)

        # Mark pretrained params (encoder)
        for p in self.base_model.parameters():
            p.is_pretrained = True
            p.is_pretrained_encoder = True

        # Mark pretrained decoder params (if using pretrained decoder)
        if self.use_pretrained_decoder:
            for module in [
                self.decoder_tok_embeddings,
                self.pretrained_decoder_layers,
                self.decoder_final_norm,
                self.decoder_lm_head,
                self.decoder_rotary_emb,
            ]:
                for p in module.parameters():
                    p.is_pretrained = True
                    p.is_pretrained_decoder = True

    def _init_decoder(self, num_decoder_layers, num_heads, dropout, max_seq_len):
        """Initialize decoder components for enc-dec generation."""
        self.decoder_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        with torch.no_grad():
            encoder_embeds = get_encoder_embeddings(self.base_model, self.model_name)
            self.decoder_embed.weight.copy_(encoder_embeds.float())
        self.decoder_embed.weight.is_pretrained = True
        self.decoder_embed.weight.is_pretrained_decoder = True

        self.pos_embed = nn.Embedding(max_seq_len, self.hidden_size)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(self.hidden_size, num_heads, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.output_proj.weight = self.decoder_embed.weight

    def _init_pretrained_decoder(
        self, decoder_model_name, cross_attn_num_heads, dropout, cross_attn_type: str = "softmax"
    ):
        """Initialize decoder from a pre-trained causal LM with cross-attention adapters.

        Loads a pre-trained model (e.g., TinyLlama) and extracts its components.
        Cross-attention adapters are injected between self-attention and FFN of each layer.
        """
        resolved_name = resolve_model_name(decoder_model_name)
        dec_config = AutoConfig.from_pretrained(resolved_name)
        if hasattr(dec_config, "attn_implementation"):
            dec_config.attn_implementation = "sdpa"

        self.decoder_hidden_size = dec_config.hidden_size
        self.decoder_num_layers = dec_config.num_hidden_layers
        self.decoder_vocab_size = dec_config.vocab_size
        self.decoder_bos_id = dec_config.bos_token_id or 1
        self.decoder_eos_id = dec_config.eos_token_id or 2

        print(f"Loading pretrained decoder: {resolved_name}")
        print(
            f"  hidden_size={self.decoder_hidden_size}, "
            f"layers={self.decoder_num_layers}, "
            f"vocab={self.decoder_vocab_size}"
        )

        full_decoder = AutoModelForCausalLM.from_pretrained(
            resolved_name, config=dec_config, torch_dtype=torch.bfloat16
        )

        # Extract components from the HF model
        self.decoder_tok_embeddings = full_decoder.model.embed_tokens
        self.pretrained_decoder_layers = full_decoder.model.layers
        self.decoder_final_norm = full_decoder.model.norm
        self.decoder_lm_head = full_decoder.lm_head
        self.decoder_rotary_emb = full_decoder.model.rotary_emb

        # Encoder-to-decoder projection (enc_hidden -> dec_hidden)
        self.encoder_projection = nn.Linear(
            self.hidden_size, self.decoder_hidden_size, bias=False
        )

        # Cross-attention adapters: one per decoder layer
        if cross_attn_type == "softmax":
            adapter_cls = CrossAttentionAdapter
            adapter_kwargs = {}
        else:
            adapter_cls = LinearCrossAttentionAdapter
            adapter_kwargs = {"variant": cross_attn_type}

        self.cross_attn_adapters = nn.ModuleList(
            [
                adapter_cls(self.decoder_hidden_size, cross_attn_num_heads, dropout, **adapter_kwargs)
                for _ in range(self.decoder_num_layers)
            ]
        )

        trainable = sum(
            p.numel()
            for p in list(self.cross_attn_adapters.parameters())
            + list(self.encoder_projection.parameters())
        )
        frozen = sum(
            p.numel()
            for m in [
                self.decoder_tok_embeddings,
                self.pretrained_decoder_layers,
                self.decoder_final_norm,
                self.decoder_lm_head,
            ]
            for p in m.parameters()
        )
        print(
            f"  Adapter params: {trainable:,} trainable, {frozen:,} pretrained (frozen by default)"
        )

        # Match adapter dtype to pretrained decoder (bfloat16)
        self.cross_attn_adapters = self.cross_attn_adapters.to(torch.bfloat16)

        # Clean up the shell model
        del full_decoder

    def _run_pretrained_decoder(
        self,
        decoder_input_ids,
        encoder_output,
        encoder_attention_mask,
        return_attn=False,
    ):
        """Run pre-trained decoder with cross-attention adapters.

        Manually loops through decoder layer components:
          1. Self-attention (layer.input_layernorm + layer.self_attn) — frozen, uses RoPE
          2. Cross-attention adapter — trained
          3. FFN (layer.post_attention_layernorm + layer.mlp) — frozen, SwiGLU
        """
        # Project encoder output to decoder hidden size (float32 -> bfloat16 to match decoder)
        projected_encoder = self.encoder_projection(encoder_output).to(
            torch.bfloat16
        )  # [B, enc_len, dec_hidden]

        # Embed decoder tokens
        x = self.decoder_tok_embeddings(decoder_input_ids)  # [B, dec_len, dec_hidden]

        dec_seq_len = decoder_input_ids.shape[1]
        position_ids = torch.arange(
            dec_seq_len, device=decoder_input_ids.device
        ).unsqueeze(0)

        # Compute rotary position embeddings (cos, sin) for self-attention
        position_embeddings = self.decoder_rotary_emb(x, position_ids=position_ids)

        cross_attn_weights = [] if return_attn else None

        for layer, adapter in zip(
            self.pretrained_decoder_layers, self.cross_attn_adapters
        ):
            # Step 1: Self-attention (frozen)
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, position_embeddings=position_embeddings, use_cache=False)[0]
            x = residual + x

            # Step 2: Cross-attention adapter (trained)
            if return_attn:
                x, attn = adapter(
                    x,
                    projected_encoder,
                    encoder_padding_mask=encoder_attention_mask.bool(),
                    return_attn=True,
                )
                cross_attn_weights.append(attn)
            else:
                x = adapter(
                    x,
                    projected_encoder,
                    encoder_padding_mask=encoder_attention_mask.bool(),
                )

            # Step 3: FFN (frozen)
            residual = x
            x = layer.post_attention_layernorm(x)
            x = layer.mlp(x)
            x = residual + x

        # Final norm and LM head
        x = self.decoder_final_norm(x)
        logits = self.decoder_lm_head(x)

        if return_attn:
            return logits, cross_attn_weights
        return logits

    def resize_embeddings(self, new_vocab_size: int):
        """Resize token embeddings for both base model and custom decoder.

        Call this after adding special tokens to the tokenizer.
        """
        if new_vocab_size == self.vocab_size:
            return

        # Resize base model embeddings
        self.base_model.resize_token_embeddings(new_vocab_size)

        # Resize custom decoder embedding if it exists
        if hasattr(self, "decoder_embed"):
            old_embed = self.decoder_embed
            new_embed = nn.Embedding(new_vocab_size, self.hidden_size)
            with torch.no_grad():
                # Copy min(old, new) embeddings
                copy_size = min(old_embed.num_embeddings, new_vocab_size)
                new_embed.weight[:copy_size] = old_embed.weight[:copy_size]
            self.decoder_embed = new_embed
            self.output_proj = nn.Linear(self.hidden_size, new_vocab_size, bias=False)
            self.output_proj.weight = self.decoder_embed.weight

        self.vocab_size = new_vocab_size

    def init_weights(self):
        """Lingua-style init: truncated normal ±3σ, output scaling by sqrt(3*n_layers)."""
        # From-scratch decoder init
        if hasattr(self, "decoder_layers"):
            std = self.hidden_size ** -0.5
            factor = (3 * len(self.decoder_layers)) ** 0.5

            def tn(weight):
                nn.init.trunc_normal_(
                    weight, mean=0.0, std=std, a=-3 * std, b=3 * std
                )

            # Decoder layers
            for layer in self.decoder_layers:
                layer.init_weights(std, factor)

            # Embeddings (skip if inherited from encoder)
            for name in ["decoder_embed", "pos_embed"]:
                if hasattr(self, name):
                    weight = getattr(self, name).weight
                    if not getattr(weight, "is_pretrained", False):
                        tn(weight)

            # LayerNorm
            if hasattr(self, "output_norm"):
                nn.init.ones_(self.output_norm.weight)
                nn.init.zeros_(self.output_norm.bias)

        # Pretrained decoder: init cross-attention adapters + encoder projection
        if hasattr(self, "cross_attn_adapters"):
            dec_std = self.decoder_hidden_size ** -0.5
            dec_factor = (3 * self.decoder_num_layers) ** 0.5

            for adapter in self.cross_attn_adapters:
                adapter.init_weights(dec_std, dec_factor)

            # Encoder projection
            nn.init.trunc_normal_(
                self.encoder_projection.weight,
                mean=0.0,
                std=dec_std,
                a=-3 * dec_std,
                b=3 * dec_std,
            )

        # Span heads
        if hasattr(self, "qa_outputs"):
            std = self.hidden_size ** -0.5
            nn.init.trunc_normal_(
                self.qa_outputs.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if self.qa_outputs.bias is not None:
                nn.init.zeros_(self.qa_outputs.bias)

        if hasattr(self, "attn_span_head"):
            nn.init.ones_(self.attn_span_head.start_layer_weights)
            nn.init.ones_(self.attn_span_head.end_layer_weights)

    def _create_segment_mask(self, input_ids, attention_mask):
        """Create attention mask that blocks attention across SEP boundaries.

        Returns a 4D mask of shape (batch, 1, seq_len, seq_len) where True = blocked.
        Tokens can only attend within their segment (between SEP tokens).
        """
        # Find SEP positions: (batch, seq_len)
        is_sep = input_ids == self.sep_token_id

        # Create segment IDs by cumsum of SEP positions
        # Each segment gets a unique ID: tokens before first SEP = 0, between first and second = 1, etc.
        segment_ids = is_sep.cumsum(dim=1)  # (batch, seq_len)

        # Create mask: can only attend if in same segment
        # (batch, seq_len, 1) != (batch, 1, seq_len) -> (batch, seq_len, seq_len)
        segment_mask = segment_ids.unsqueeze(2) != segment_ids.unsqueeze(1)

        # Combine with padding mask: also block attention to padding
        # attention_mask is (batch, seq_len), 1 = attend, 0 = block
        padding_mask = (attention_mask == 0).unsqueeze(1)  # (batch, 1, seq_len)
        combined_mask = segment_mask | padding_mask  # (batch, seq_len, seq_len)

        # Add head dimension: (batch, 1, seq_len, seq_len)
        return combined_mask.unsqueeze(1)

    def _get_encoder_layers(self):
        """Get the list of encoder layers from the base model."""
        # ModernBERT / BERT-style models
        if hasattr(self.base_model, "encoder") and hasattr(
            self.base_model.encoder, "layers"
        ):
            return self.base_model.encoder.layers
        # Some models use 'layer' instead of 'layers'
        if hasattr(self.base_model, "encoder") and hasattr(
            self.base_model.encoder, "layer"
        ):
            return self.base_model.encoder.layer
        # Fallback for other architectures
        raise ValueError(
            f"Cannot find encoder layers in model: {type(self.base_model)}"
        )

    def _get_encoder_embeddings_module(self):
        """Get the embeddings module from the base model."""
        if hasattr(self.base_model, "embeddings"):
            return self.base_model.embeddings
        if hasattr(self.base_model, "model") and hasattr(
            self.base_model.model, "embeddings"
        ):
            return self.base_model.model.embeddings
        raise ValueError(f"Cannot find embeddings in model: {type(self.base_model)}")

    def _run_encoder(self, input_ids, attention_mask):
        """Run encoder and return hidden states.

        If enc_local_layers > 0, applies segment-local attention to bottom layers.
        """
        # Fast path: no local attention restriction
        if self.enc_local_layers == 0:
            return self.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.float()

        # Slow path: per-layer attention control
        # Create masks
        # Global mask: standard padding mask, shape (batch, 1, 1, seq_len) for broadcasting
        global_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e9

        # Local mask: segment-isolated + padding, shape (batch, 1, seq_len, seq_len)
        segment_blocked = self._create_segment_mask(input_ids, attention_mask)
        local_mask = segment_blocked.float() * -1e9

        # Get embeddings
        embeddings_module = self._get_encoder_embeddings_module()
        hidden_states = embeddings_module(input_ids)

        # Run through layers with appropriate masks
        layers = self._get_encoder_layers()
        for i, layer in enumerate(layers):
            mask = local_mask if i < self.enc_local_layers else global_mask
            # Most HF layers return tuple (hidden_states, ...) or object with last_hidden_state
            layer_output = layer(hidden_states, attention_mask=mask)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # Apply final layer norm if present
        if hasattr(self.base_model, "encoder") and hasattr(
            self.base_model.encoder, "final_layer_norm"
        ):
            hidden_states = self.base_model.encoder.final_layer_norm(hidden_states)

        return hidden_states.float()

    def _run_custom_decoder(
        self,
        decoder_input_ids,
        encoder_output,
        encoder_attention_mask,
        return_attn=False,
    ):
        """Run custom decoder layers with cross-attention.

        Args:
            return_attn: If True, also returns list of cross-attention weights per layer.
        """
        dec_seq_len = decoder_input_ids.shape[1]
        positions = torch.arange(dec_seq_len, device=decoder_input_ids.device)
        x = self.decoder_embed(decoder_input_ids) + self.pos_embed(positions)

        causal_mask = torch.triu(
            torch.ones(dec_seq_len, dec_seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        cross_attn_weights = [] if return_attn else None
        for layer in self.decoder_layers:
            if return_attn:
                x, attn = layer(
                    x,
                    encoder_output,
                    causal_mask=causal_mask,
                    encoder_padding_mask=encoder_attention_mask.bool(),
                    return_attn=True,
                )
                cross_attn_weights.append(attn)
            else:
                x = layer(
                    x,
                    encoder_output,
                    causal_mask=causal_mask,
                    encoder_padding_mask=encoder_attention_mask.bool(),
                )

        x = self.output_norm(x)
        logits = self.output_proj(x)

        if return_attn:
            return logits, cross_attn_weights
        return logits

    def _compute_span_loss(
        self, start_logits, end_logits, start_positions, end_positions
    ):
        """Compute span extraction loss."""
        if start_positions is None or end_positions is None:
            return None
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index - 1)
        end_positions = end_positions.clamp(0, ignored_index - 1)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        return (
            loss_fct(start_logits, start_positions)
            + loss_fct(end_logits, end_positions)
        ) / 2

    def _compute_gen_loss(self, logits, labels):
        """Compute generation loss with label shifting."""
        if labels is None:
            return None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        start_positions=None,
        end_positions=None,
    ):
        result = {}
        needs_gen = self.span_expr != "bertlike"
        needs_hidden_span = self.span_expr in ("bertlike", "first_last_hidden")
        needs_attn_span = self.span_expr == "first_last_attn"

        # Step 1: Get hidden states and run generation based on model_type
        if self.model_type == "dec":
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels if needs_gen else None,
                output_hidden_states=needs_hidden_span,
            )
            hidden_states = outputs.hidden_states[-1] if needs_hidden_span else None
            gen_loss = outputs.loss if needs_gen else None
            result["logits"] = outputs.logits
            cross_attn_weights = None
        else:  # encdec
            hidden_states = self._run_encoder(encoder_input_ids, encoder_attention_mask)
            run_decoder = (
                self._run_pretrained_decoder
                if self.use_pretrained_decoder
                else self._run_custom_decoder
            )
            if needs_gen:
                if needs_attn_span:
                    # Get cross-attention weights for attention-based span extraction
                    logits, cross_attn_weights = run_decoder(
                        decoder_input_ids,
                        hidden_states,
                        encoder_attention_mask,
                        return_attn=True,
                    )
                else:
                    logits = run_decoder(
                        decoder_input_ids, hidden_states, encoder_attention_mask
                    )
                    cross_attn_weights = None
                result["logits"] = logits
                gen_loss = self._compute_gen_loss(logits, labels)
            else:
                gen_loss = None
                cross_attn_weights = None

        # Step 2: Span extraction based on span_expr mode
        span_loss = None
        if needs_hidden_span:
            # Hidden-state based span extraction (bertlike, first_last_hidden)
            span_logits = self.qa_outputs(hidden_states.float())
            start_logits, end_logits = span_logits.squeeze(-1).split(1, dim=-1)
            result["start_logits"] = start_logits.squeeze(-1)
            result["end_logits"] = end_logits.squeeze(-1)
            span_loss = self._compute_span_loss(
                result["start_logits"],
                result["end_logits"],
                start_positions,
                end_positions,
            )
        elif needs_attn_span:
            # Attention-based span extraction (first_last_attn)
            # Decoder sequence: [BOS] [SPAN] first_anchor ... last_anchor [EOS]
            # first_pos=1: attention from [SPAN] position when predicting first anchor
            # last_pos=-2: attention from second-to-last position when predicting last anchor
            dec_len = decoder_input_ids.shape[1]
            start_logits, end_logits = self.attn_span_head(
                cross_attn_weights, first_token_pos=1, last_token_pos=dec_len - 2
            )
            result["start_logits"] = start_logits
            result["end_logits"] = end_logits
            span_loss = self._compute_span_loss(
                start_logits, end_logits, start_positions, end_positions
            )

        # Step 3: Combine losses
        if self.span_expr == "bertlike":
            result["loss"] = span_loss
        elif self.span_expr.startswith("first_last"):
            if span_loss is not None and gen_loss is not None:
                result["loss"] = self.span_loss_weight * span_loss + gen_loss
            else:
                result["loss"] = span_loss or gen_loss
        else:  # none
            result["loss"] = gen_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        tokenizer=None,
        max_new_tokens=64,
        decoder_prefix_ids=None,
        **kwargs,
    ):
        """Generate answer: first generate tokens, then optionally resolve span.

        Structure:
        1. Generate phase: produce decoder tokens (skip for bertlike)
        2. Resolve phase: if span extraction, locate span in encoder using anchors + logits

        Returns:
            For span_expr != "none": extracted span token IDs from encoder
            For span_expr == "none": generated decoder token IDs
        """
        # Step 1: Generate decoder tokens
        if self.span_expr == "bertlike":
            # No generation needed, skip to span resolution
            generated_ids = None
            encoder_output = None
            cross_attn_weights = None
        elif self.model_type == "dec":
            # Decoder-only generation
            generated_ids = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            encoder_output = None
            cross_attn_weights = None
        else:
            # Encoder-decoder generation
            generated_ids, encoder_output, cross_attn_weights = self._generate_encdec(
                encoder_input_ids,
                encoder_attention_mask,
                tokenizer,
                max_new_tokens,
                decoder_prefix_ids,
                return_attn=(self.span_expr == "first_last_attn"),
            )

        # Step 2: Resolve span (if span extraction mode)
        if self.span_expr == "none":
            return generated_ids

        if self.model_type == "dec":
            return self._resolve_span_dec(
                input_ids, attention_mask, generated_ids, tokenizer
            )
        else:
            return self._resolve_span_encdec(
                encoder_input_ids,
                encoder_attention_mask,
                encoder_output,
                generated_ids,
                cross_attn_weights,
                tokenizer,
            )

    @torch.no_grad()
    def _generate_encdec(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        tokenizer,
        max_new_tokens,
        decoder_prefix_ids=None,
        return_attn=False,
    ):
        """Encoder-decoder autoregressive generation.

        Returns:
            (generated_ids, encoder_output, cross_attn_weights)
        """
        device = encoder_input_ids.device
        batch_size = encoder_input_ids.shape[0]
        encoder_output = self._run_encoder(encoder_input_ids, encoder_attention_mask)

        run_decoder = (
            self._run_pretrained_decoder
            if self.use_pretrained_decoder
            else self._run_custom_decoder
        )

        # Initialize decoder with appropriate BOS/EOS tokens
        if decoder_prefix_ids is not None:
            decoder_ids = decoder_prefix_ids
        elif self.use_pretrained_decoder:
            start_token = self.decoder_bos_id
            decoder_ids = torch.full(
                (batch_size, 1), start_token, device=device, dtype=torch.long
            )
        else:
            start_token = tokenizer.cls_token_id or tokenizer.bos_token_id or 0
            decoder_ids = torch.full(
                (batch_size, 1), start_token, device=device, dtype=torch.long
            )

        if self.use_pretrained_decoder:
            eos_id = self.decoder_eos_id
        else:
            eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        cross_attn_weights = None

        for _ in range(max_new_tokens):
            if return_attn:
                logits, cross_attn_weights = run_decoder(
                    decoder_ids,
                    encoder_output,
                    encoder_attention_mask,
                    return_attn=True,
                )
            else:
                logits = run_decoder(
                    decoder_ids, encoder_output, encoder_attention_mask
                )

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_ids = torch.cat([decoder_ids, next_token], dim=1)

            if eos_id and (next_token == eos_id).all():
                break

        return decoder_ids, encoder_output, cross_attn_weights

    @torch.no_grad()
    def _resolve_span_dec(self, input_ids, attention_mask, generated_ids, tokenizer):
        """Resolve span for decoder-only models."""
        # Get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Get span logits
        span_logits = self.qa_outputs(hidden_states.float())
        start_logits, end_logits = span_logits.squeeze(-1).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # For first_last modes, parse anchors from generated_ids
        if self.span_expr.startswith("first_last") and generated_ids is not None:
            first_anchors, last_anchors = self._parse_anchors(generated_ids, tokenizer)
            return self._resolve_with_anchors(
                input_ids, start_logits, end_logits, first_anchors, last_anchors
            )
        else:
            # bertlike: pure logits
            return self._logits_to_spans(input_ids, start_logits, end_logits)

    @torch.no_grad()
    def _resolve_span_encdec(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        encoder_output,
        generated_ids,
        cross_attn_weights,
        tokenizer,
    ):
        """Resolve span for encoder-decoder models."""
        # Get encoder output if not provided (bertlike case)
        if encoder_output is None:
            encoder_output = self._run_encoder(
                encoder_input_ids, encoder_attention_mask
            )

        # Get span logits based on mode
        if self.span_expr in ("bertlike", "first_last_hidden"):
            span_logits = self.qa_outputs(encoder_output.float())
            start_logits, end_logits = span_logits.squeeze(-1).split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
        else:  # first_last_attn
            dec_len = generated_ids.shape[1]
            start_logits, end_logits = self.attn_span_head(
                cross_attn_weights,
                first_token_pos=1,
                last_token_pos=max(1, dec_len - 2),
            )

        # For first_last modes, parse anchors and filter candidates
        if self.span_expr.startswith("first_last") and generated_ids is not None:
            first_anchors, last_anchors = self._parse_anchors(generated_ids, tokenizer)
            return self._resolve_with_anchors(
                encoder_input_ids, start_logits, end_logits, first_anchors, last_anchors
            )
        else:
            # bertlike: pure logits
            return self._logits_to_spans(encoder_input_ids, start_logits, end_logits)

    def _parse_anchors(self, generated_ids, tokenizer):
        """Parse first/last anchor tokens from generated sequence.

        Expected structure: [CLS] [SPAN] first_anchor(s) ... last_anchor(s) [EOS]
        """
        batch_size = generated_ids.shape[0]
        eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        ellipsis_ids = tokenizer.encode(" ...", add_special_tokens=False)

        first_anchors = []
        last_anchors = []

        for i in range(batch_size):
            seq = generated_ids[i].tolist()
            ellipsis_start = self._find_subsequence(seq, ellipsis_ids)

            if ellipsis_start is not None:
                first_anchors.append(seq[2:ellipsis_start])
                end_slice = seq[ellipsis_start + len(ellipsis_ids) :]
                if end_slice and end_slice[-1] == eos_id:
                    end_slice = end_slice[:-1]
                last_anchors.append(end_slice)
            else:
                # Single token span
                anchor = seq[2:3] if len(seq) > 2 else []
                first_anchors.append(anchor)
                last_anchors.append(anchor)

        return first_anchors, last_anchors

    def _resolve_with_anchors(
        self, input_ids, start_logits, end_logits, first_anchors, last_anchors
    ):
        """Resolve span using anchor filtering + logit disambiguation."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        spans = []

        for i in range(batch_size):
            enc_ids = input_ids[i].tolist()
            first_anchor = first_anchors[i]
            last_anchor = last_anchors[i]

            # Find candidate positions matching anchor tokens
            if first_anchor:
                start_candidates = [
                    j for j, tok in enumerate(enc_ids) if tok == first_anchor[0]
                ]
            else:
                start_candidates = list(range(len(enc_ids)))

            if last_anchor:
                end_candidates = [
                    j for j, tok in enumerate(enc_ids) if tok == last_anchor[-1]
                ]
            else:
                end_candidates = list(range(len(enc_ids)))

            # Use logits to select best among candidates
            if start_candidates:
                start_scores = start_logits[i][start_candidates]
                start_idx = start_candidates[start_scores.argmax().item()]
            else:
                start_idx = start_logits[i].argmax().item()

            if end_candidates:
                valid_end = [j for j in end_candidates if j >= start_idx]
                if valid_end:
                    end_scores = end_logits[i][valid_end]
                    end_idx = valid_end[end_scores.argmax().item()]
                else:
                    end_idx = start_idx
            else:
                end_idx = end_logits[i].argmax().item()
                if end_idx < start_idx:
                    end_idx = start_idx

            spans.append(input_ids[i, start_idx : end_idx + 1])

        return self._pad_spans(spans, device)

    def _find_subsequence(self, seq: List[int], subseq: List[int]) -> Optional[int]:
        """Find starting index of subsequence in sequence."""
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i : i + len(subseq)] == subseq:
                return i
        return None

    def _logits_to_spans(self, input_ids, start_logits, end_logits):
        """Convert logits to extracted spans (no anchor filtering)."""
        batch_size = input_ids.shape[0]
        spans = []
        for i in range(batch_size):
            start_idx = start_logits[i].argmax().item()
            end_idx = end_logits[i].argmax().item()
            if end_idx < start_idx:
                end_idx = start_idx
            spans.append(input_ids[i, start_idx : end_idx + 1])
        return self._pad_spans(spans, input_ids.device)

    def _pad_spans(self, spans: List[torch.Tensor], device) -> torch.Tensor:
        """Pad list of span tensors to same length."""
        batch_size = len(spans)
        max_span_len = max(s.shape[0] for s in spans)
        padded = torch.zeros(batch_size, max_span_len, dtype=torch.long, device=device)
        for i, span in enumerate(spans):
            padded[i, : span.shape[0]] = span
        return padded


# ============== Evaluation Helpers ==============


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(
        remove_articles(
            s.lower().translate(str.maketrans("", "", string.punctuation))
        ).split()
    )


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision, recall = num_same / len(pred_tokens), num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def evaluate_predictions(predictions, ground_truths):
    exact_matches, f1_scores = [], []
    for pred, golds in zip(predictions, ground_truths):
        golds = [golds] if isinstance(golds, str) else golds
        em = max(int(normalize_answer(pred) == normalize_answer(g)) for g in golds)
        f1 = max(compute_f1(pred, g) for g in golds)
        exact_matches.append(em)
        f1_scores.append(f1)
    return {
        "exact_match": sum(exact_matches) / len(exact_matches) * 100,
        "f1": sum(f1_scores) / len(f1_scores) * 100,
    }


# ============== Training Functions ==============


def train_step(model, batch):
    """Generic train step based on model config."""
    # Build kwargs based on model type
    kwargs = {"labels": batch.get("labels")}

    if model.model_type == "encdec":
        kwargs["encoder_input_ids"] = batch["encoder_input_ids"]
        kwargs["encoder_attention_mask"] = batch["encoder_attention_mask"]
        kwargs["decoder_input_ids"] = batch.get("decoder_input_ids")
    else:
        kwargs["input_ids"] = batch["input_ids"]
        kwargs["attention_mask"] = batch["attention_mask"]

    # Span extraction fields
    if model.span_expr != "none":
        kwargs["start_positions"] = batch.get("start_positions")
        kwargs["end_positions"] = batch.get("end_positions")

    outputs = model(**kwargs)

    if isinstance(outputs, dict):
        return outputs["loss"]
    return outputs.loss


def _get_device(model):
    return next(model.parameters()).device


def train_epoch(model, dataloader, optimizer, epoch, use_wandb, eval_callback=None):
    model.train()
    device = _get_device(model)
    total_loss = 0
    total_steps = len(dataloader)
    eval_interval = max(1, total_steps // 10)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = train_step(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if use_wandb and step % 10 == 0:
            wandb.log({"train/loss": loss.item(), "train/step": step})

        # Evaluate every 10% of the epoch steps (skip step 0)
        if eval_callback is not None and step > 0 and step % eval_interval == 0:
            pct = int(round(step / total_steps * 100))
            pbar.write(f"\n[Epoch {epoch} | {pct}% ({step}/{total_steps})] Running validation...")
            eval_callback(epoch=epoch, step=step, total_steps=total_steps)
            model.train()

    return total_loss / len(dataloader)


def evaluate_loss(model, dataloader):
    model.eval()
    device = _get_device(model)
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_step(model, batch)
            total_loss += loss.item()
    return {"loss": total_loss / len(dataloader)}


def evaluate_generation(
    model,
    dataset,
    tokenizer,
    preparer: DataPreparer,
    max_samples=None,
    decoder_tokenizer=None,
):
    """Evaluate generation using the same data pipeline as training."""
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    features = dataset.map(
        preparer.prepare_features, batched=True, remove_columns=dataset.column_names
    )

    model.eval()
    device = _get_device(model)
    predictions = []
    ground_truths = [sample["answers"]["text"] for sample in dataset]

    for feature in tqdm(features, desc="Generating"):
        # Determine input tensors based on model type
        if model.model_type == "encdec":
            encoder_input_ids = torch.tensor([feature["encoder_input_ids"]]).to(device)
            encoder_attention_mask = torch.tensor(
                [feature["encoder_attention_mask"]]
            ).to(device)
            input_ids = None
            attention_mask = None
        else:
            input_ids = torch.tensor([feature["input_ids"]]).to(device)
            attention_mask = torch.tensor([feature["attention_mask"]]).to(device)
            encoder_input_ids = None
            encoder_attention_mask = None

        # For generation modes, determine prefix/prompt handling
        decoder_prefix_ids = None
        prompt_len = 0

        if model.span_expr == "none":
            labels = feature["labels"]
            prompt_len = sum(1 for l in labels if l == -100)

            if model.model_type == "encdec":
                # Enc-dec: use decoder prefix
                decoder_prefix = feature["decoder_input_ids"][:prompt_len]
                decoder_prefix_ids = torch.tensor([decoder_prefix]).to(device)
            else:
                # Dec-only: truncate input to prompt only
                input_ids = input_ids[:, :prompt_len]
                attention_mask = attention_mask[:, :prompt_len]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_prefix_ids=decoder_prefix_ids,
                tokenizer=tokenizer,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode output (use decoder tokenizer if pretrained decoder is used)
        dec_tok = decoder_tokenizer if decoder_tokenizer else tokenizer
        if model.span_expr != "none":
            # Span extraction: output is the extracted span (always encoder tokenizer)
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        elif getattr(model, "use_pretrained_decoder", False):
            # Pretrained decoder: decode with decoder tokenizer
            pred = dec_tok.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()
        else:
            # Generation: skip prefix tokens
            pred = tokenizer.decode(
                output_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()

        predictions.append(pred)

    return evaluate_predictions(predictions, ground_truths), predictions, ground_truths


# ============== Main ==============


def main():
    parser = argparse.ArgumentParser(
        description="Unified QA with configurable data formats"
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="Q/C/A",
        choices=DATA_FORMAT_CHOICES,
        help=f"Data format: {DATA_FORMAT_CHOICES}",
    )
    parser.add_argument(
        "--span_expr",
        type=str,
        default="none",
        choices=["bertlike", "first_last_hidden", "first_last_attn", "none"],
        help="Span expression: none=generate, bertlike=span head only, "
        "first_last_hidden=span head + generation, first_last_attn=attention-based span + generation",
    )
    parser.add_argument(
        "--span_loss_weight",
        type=float,
        default=1.0,
        help="Weight for span extraction loss (default: 1.0)",
    )
    parser.add_argument(
        "--enc_local_layer_ratio",
        type=float,
        default=0.0,
        help="Fraction of bottom encoder layers with local attention (no cross-SEP). "
        "0.0=all global (default), 1.0=all local. E.g., 0.5 with 22 layers means bottom 11 are local.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="encdec",
        choices=["encdec", "dec"],
        help="Model type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tinyllama_1b",
        help=f"Model alias or HF name. Aliases: {', '.join(MODEL_ALIASES.keys())}",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained_weight_updating",
        type=float,
        default=None,
        help="Pretrained weight updating mode: 0.0/None to freeze, >0.0 to scale LR",
    )
    parser.add_argument(
        "--encoder_weight_updating",
        type=float,
        default=None,
        help="Override pretrained_weight_updating for encoder only. "
        "E.g. --pretrained_weight_updating 0.0 --encoder_weight_updating 0.333 "
        "freezes decoder but trains encoder at 0.333x LR (CEPE-style).",
    )
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument(
        "--decoder_model_name",
        type=str,
        default=None,
        help="Pre-trained decoder model (e.g., tinyllama_1b). When set with model_type=encdec, "
        "uses pre-trained decoder with cross-attention adapters instead of from-scratch decoder.",
    )
    parser.add_argument(
        "--cross_attn_num_heads",
        type=int,
        default=16,
        help="Number of attention heads for cross-attention adapters (default: 16)",
    )
    parser.add_argument(
        "--cross_attn_type",
        type=str,
        default="softmax",
        choices=["softmax", "linear", "linear_kda"],
        help=(
            "Cross-attention adapter type: 'softmax' (default, nn.MHA), "
            "'linear' (parallel S=K^TV), 'linear_kda' (KDA delta-rule recurrence)."
        ),
    )
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--dec_max_length", type=int, default=8192)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=500)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--wandb_project", type=str, default="minimal_squad")
    parser.add_argument("--wandb_run_name", type=str, default="debug")
    args = parser.parse_args()

    # Parse format
    parsed_format = parse_format(args.data_format)
    resolved_model_name = resolve_model_name(args.model_name)

    # Validate config consistency
    if args.span_expr != "none" and not parsed_format.is_span_extraction:
        raise ValueError(
            f"--span_expr={args.span_expr} requires a span extraction data format (e.g., C/Q//S), "
            f"but got --data_format={args.data_format}. Use 'S' after '//' for span extraction."
        )
    if parsed_format.is_span_extraction and args.span_expr == "none":
        raise ValueError(
            f"Data format {args.data_format} indicates span extraction, "
            f"but --span_expr is 'none'. Use --span_expr=bertlike, first_last_hidden, or first_last_attn."
        )

    print(
        f"Parsed format: encoder={parsed_format.encoder_parts}, decoder={parsed_format.decoder_parts}"
    )
    print(f"Model type: {args.model_type}, Span extraction: {args.span_expr}")

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and args.wandb_run_name is not None
    if use_wandb:
        config = vars(args).copy()
        config["resolved_model_name"] = resolved_model_name
        config["model_type"] = args.model_type
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    print(f"Device: {args.device}")
    print(f"Data format: {args.data_format}")
    print(f"Model: {resolved_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    infer_special_tokens(tokenizer)

    # Load decoder tokenizer if using pretrained decoder
    decoder_tokenizer = None
    if args.decoder_model_name:
        resolved_decoder = resolve_model_name(args.decoder_model_name)
        decoder_tokenizer = AutoTokenizer.from_pretrained(resolved_decoder)
        infer_special_tokens(decoder_tokenizer)
        print(f"Decoder tokenizer: {resolved_decoder} (vocab_size={len(decoder_tokenizer)})")

    # Add [SPAN] special token for span extraction modes
    if parsed_format.is_span_extraction:
        special_tokens = getattr(tokenizer, "additional_special_tokens", None) or []
        if "[SPAN]" not in special_tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": special_tokens + ["[SPAN]"]}
            )
            print(
                f"Added [SPAN] special token (id: {tokenizer.convert_tokens_to_ids('[SPAN]')})"
            )

    # Load dataset
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad")

    # Filter out examples where total length exceeds 2000 characters
    def filter_by_length(example):
        q_len = len(example["question"])
        c_len = len(example["context"])
        a_len = max((len(ans) for ans in example["answers"]["text"]), default=0)
        return q_len + c_len + a_len <= 2000

    original_train_size = len(dataset["train"])
    original_val_size = len(dataset["validation"])
    dataset = dataset.filter(filter_by_length)
    print(f"Filtered train: {original_train_size} -> {len(dataset['train'])}")
    print(f"Filtered val: {original_val_size} -> {len(dataset['validation'])}")

    # Prepare data
    preparer = DataPreparer(
        tokenizer,
        parsed_format,
        args.max_length,
        args.dec_max_length,
        args.span_expr,
        decoder_tokenizer=decoder_tokenizer,
    )

    train_dataset = dataset["train"]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    val_dataset = dataset["validation"]
    if args.max_val_samples:
        val_dataset = val_dataset.select(
            range(min(args.max_val_samples, len(val_dataset)))
        )

    print("Preparing features...")
    train_features = train_dataset.map(
        preparer.prepare_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    # Debug: print one example with token_ids converted to token strings
    print("\n" + "=" * 60)
    print("Inspect: Example from train_features (token_ids -> tokens)")
    print("=" * 60)
    example = train_features[0]
    for key, value in example.items():
        if "input_ids" in key:
            # Convert token ids to token strings
            tokens = tokenizer.convert_ids_to_tokens(value)
            print(f"\n{key}:")
            print(f"  IDs: {value[:50]}{'...' if len(value) > 50 else ''}")
            print(f"  Tokens: {tokens[:50]}{'...' if len(tokens) > 50 else ''}")
            print(
                f"  Decoded: {tokenizer.decode(value, skip_special_tokens=False)[:200]}..."
            )
        elif key == "labels":
            # Labels contain -100 for ignored positions, filter those for decoding
            valid_ids = [v for v in value if v != -100]
            tokens = tokenizer.convert_ids_to_tokens(valid_ids) if valid_ids else []
            print(f"\n{key}:")
            print(f"  IDs: {value[:50]}{'...' if len(value) > 50 else ''}")
            print(f"  Valid tokens: {tokens[:50]}{'...' if len(tokens) > 50 else ''}")
            print(
                f"  Decoded (valid only): {tokenizer.decode(valid_ids, skip_special_tokens=False)[:200] if valid_ids else '(empty)'}..."
            )
        elif "mask" in key:
            print(f"\n{key}: len={len(value)}, sum={sum(value)}")

    # Show span tokens if positions are present
    if "start_positions" in example and "end_positions" in example:
        start_pos = example["start_positions"]
        end_pos = example["end_positions"]
        # Span positions refer to encoder_input_ids (enc-dec) or input_ids (dec-only)
        span_source = example.get("encoder_input_ids", example.get("input_ids", []))
        if span_source and start_pos < len(span_source) and end_pos < len(span_source):
            span_ids = span_source[start_pos : end_pos + 1]
            span_tokens = tokenizer.convert_ids_to_tokens(span_ids)
            span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
            print(f"\nspan [start_positions, end_positions]: [{start_pos}:{end_pos}]:")
            print(f"  tokens: {span_tokens}")
            print(f"  decoded: {span_text}")

    # Also show ground truth answer for comparison
    sample = train_dataset[0]
    print(
        f"\nGround truth answer: {sample['answers']['text'][0] if sample['answers']['text'] else 'N/A'}"
    )
    print("=" * 60 + "\n")

    val_features = val_dataset.map(
        preparer.prepare_features, batched=True, remove_columns=val_dataset.column_names
    )

    # Use right-padding for encoder models and span extraction (span positions assume left-aligned content)
    # Use left-padding for decoder-only causal LM (preferred for batch generation)
    if args.model_type == "dec" and not parsed_format.is_span_extraction:
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer.padding_side = padding_side
    dec_pad_id = decoder_tokenizer.pad_token_id if decoder_tokenizer else None
    collate_fn = get_collate_fn(
        tokenizer.pad_token_id, padding_side=padding_side, dec_pad_token_id=dec_pad_id
    )
    print(f"Using {padding_side}-padding")
    train_loader = DataLoader(
        train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_features, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Train: {len(train_features)}, Val: {len(val_features)}")

    # Create model
    print("Loading model...")
    model = UnifiedModel(
        model_name=resolved_model_name,
        model_type=args.model_type,
        span_expr=args.span_expr,
        num_decoder_layers=args.num_decoder_layers,
        max_seq_len=args.dec_max_length,
        span_loss_weight=args.span_loss_weight,
        # Only resize embeddings if we added [SPAN] token
        vocab_size=len(tokenizer) if parsed_format.is_span_extraction else None,
        enc_local_layer_ratio=args.enc_local_layer_ratio,
        sep_token_id=tokenizer.sep_token_id,
        decoder_model_name=args.decoder_model_name,
        cross_attn_num_heads=args.cross_attn_num_heads,
        cross_attn_type=args.cross_attn_type,
    )

    # Initialize custom decoder weights with lingua-style initialization
    model.init_weights()

    model = model.to(args.device)

    # Parameter handling for pretrained_weight_updating

    pretrained_encoder_ids = set()
    pretrained_decoder_ids = set()
    other_ids = set()

    for name, param in model.named_parameters():
        # If [SPAN] was added, don't count embedding as pretrained
        is_embedding = "embed" in name.lower() or "word_embedding" in name.lower()
        if getattr(param, "is_pretrained", False) and not (
            parsed_format.is_span_extraction and is_embedding
        ):
            if getattr(param, "is_pretrained_encoder", False):
                pretrained_encoder_ids.add(id(param))
            else:
                pretrained_decoder_ids.add(id(param))
        else:
            other_ids.add(id(param))

    pretrained_encoder_params = [
        p for p in model.parameters() if id(p) in pretrained_encoder_ids
    ]
    pretrained_decoder_params = [
        p for p in model.parameters() if id(p) in pretrained_decoder_ids
    ]
    other_params = [p for p in model.parameters() if id(p) in other_ids]

    # Determine effective LR multipliers
    encoder_updating = (
        args.encoder_weight_updating
        if args.encoder_weight_updating is not None
        else args.pretrained_weight_updating
    )
    decoder_updating = args.pretrained_weight_updating

    optimizer_grouped_parameters = []

    # Encoder pretrained params
    if encoder_updating is None or encoder_updating == 0.0:
        print("Freezing pretrained encoder weights...")
        for p in pretrained_encoder_params:
            p.requires_grad = False
    else:
        print(f"Scaling pretrained encoder LR by {encoder_updating}...")
        optimizer_grouped_parameters.append(
            {"params": pretrained_encoder_params, "lr": args.lr * encoder_updating}
        )

    # Decoder pretrained params
    if decoder_updating is None or decoder_updating == 0.0:
        print("Freezing pretrained decoder weights...")
        for p in pretrained_decoder_params:
            p.requires_grad = False
    else:
        print(f"Scaling pretrained decoder LR by {decoder_updating}...")
        optimizer_grouped_parameters.append(
            {"params": pretrained_decoder_params, "lr": args.lr * decoder_updating}
        )

    # Non-pretrained params (adapters, projection, heads) — always at full LR
    optimizer_grouped_parameters.append({"params": other_params, "lr": args.lr})

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    if use_wandb:
        wandb.log(
            {
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, fused=True
    )

    def run_squad_eval(epoch, step, total_steps):
        """Run SQuAD validation and log results."""
        val_metrics = evaluate_loss(model, val_loader)
        gen_metrics, predictions, ground_truths = evaluate_generation(
            model,
            dataset["validation"],
            tokenizer,
            preparer,
            args.eval_samples,
            decoder_tokenizer=decoder_tokenizer,
        )
        pct = int(round(step / total_steps * 100)) if total_steps > 0 else 100
        print(
            f"[Epoch {epoch} | {pct}%] Val loss: {val_metrics['loss']:.4f} | "
            f"EM: {gen_metrics['exact_match']:.2f}% | F1: {gen_metrics['f1']:.2f}%"
        )
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "epoch_pct": pct,
                    "val/loss": val_metrics["loss"],
                    "val/exact_match": gen_metrics["exact_match"],
                    "val/f1": gen_metrics["f1"],
                }
            )
        print("\n  Samples:")
        for i in range(min(3, len(predictions))):
            sample = dataset["validation"][i]
            gold = ground_truths[i][0] if ground_truths[i] else "N/A"
            print(f"    Q: {sample['question'][:60]}...")
            print(f"    Gold: {gold}")
            print(f"    Pred: {predictions[i]}")
            print()
        return val_metrics, gen_metrics

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, use_wandb, eval_callback=run_squad_eval
        )
        print(f"Epoch {epoch} - Train loss: {train_loss:.4f}")

        # End-of-epoch evaluation (100%)
        val_metrics, gen_metrics = run_squad_eval(
            epoch=epoch, step=len(train_loader), total_steps=len(train_loader)
        )

        if use_wandb:
            wandb.log({"epoch": epoch, "train/epoch_loss": train_loss})

    print("\nTraining complete!")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
