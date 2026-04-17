"""
Model configuration for FineSearch encoder-decoder.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelArgs:
    """Model configuration. Supports encoder-decoder or decoder-only."""

    # Model type
    model_type: Literal["encdec", "dec"] = "encdec"

    # Encoder (used when model_type="encdec")
    encoder_name: str = "answerdotai/ModernBERT-base"
    freeze_encoder: bool = False

    # Encoder local attention (hybrid local/global)
    # See addons/models/attention.py for mask implementations
    enc_local_layer_ratio: float = 0.0  # Fraction of bottom layers with local attention
    enc_local_mask_type: Literal["segment", "block_causal"] = "segment"

    # Decoder
    decoder_name: str = ""  # Pretrained decoder, empty = from-scratch
    num_decoder_layers: int = 6  # Only used when decoder_name is empty
    decoder_max_position: int = 8192  # RoPE max sequence length

    # Activation checkpointing
    activation_checkpointing: bool = False

    # Cross-attention (ignored in decoder-only mode)
    # See addons/models/attention.py for kernel implementations
    cross_attn_layers: str = "all"  # "all" or comma-separated layer indices
    cross_attn_heads: int = 8
    cross_attn_type: Literal[
        "softmax", "linear", "gla", "delta_rule", "gated_delta_rule",
        "kda", "retention", "based", "simple_gla", "mamba2", "rwkv6",
        "hgrn2", "abc",
    ] = "softmax"


@dataclass
class GenerationArgs:
    """Generation configuration."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
