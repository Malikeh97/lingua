# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Embedding Pointer Architecture for Extractive QA.

This module implements an encoder-decoder model where the decoder points directly
to encoder positions instead of generating tokens from vocabulary. This is a pure
extractive approach where outputs are guaranteed to come from the input context.

The decoder uses dot-product attention between its hidden states and encoder
hidden states (contextualized embeddings) to select positions.

Key components:
- PointerDecoder: Decoder that outputs pointer logits over encoder positions
- EncDecPointerTransformer: Full encoder-decoder model with pointer mechanism
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import BlockMask

from lingua.transformer import (
    RMSNorm,
    FeedForward,
    RotaryEmbedding,
    Attention,
    repeat_kv,
    InitStdFactor,
)

# Import shared components from enc_dec
from apps.enc_dec.enc_dec import (
    EncDecTransformerArgs,
    EncoderType,
    PretrainedEncoder,
    Encoder,
    EmbeddingOnlyEncoder,
    DecoderBlock,
    CrossAttention,
    create_causal_mask,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token_enc_dec,
    load_pretrained_decoder_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class PointerMechanismArgs:
    """Arguments for pointer mechanism configuration."""
    pointer_temperature: float = 1.0  # Temperature for pointer logits (lower = sharper)
    use_pointer_projection: bool = True  # Use learned projection for pointer or direct dot product
    pointer_dim: Optional[int] = None  # Dimension for pointer projection (None = use model dim)


class PointerDecoder(nn.Module):
    """Decoder that points to encoder positions for extractive QA.

    Instead of generating tokens from vocabulary, the decoder outputs
    pointer logits over encoder positions. The final output is extracted
    by indexing into the encoder input tokens.
    """

    def __init__(
        self,
        args: EncDecTransformerArgs,
        pointer_args: Optional[PointerMechanismArgs] = None,
        shared_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()

        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.max_seqlen = args.max_decoder_seqlen

        dec_args = args.decoder
        self.n_layers = dec_args.n_layers
        self.init_base_std = dec_args.init_base_std
        self.init_std_factor = InitStdFactor(dec_args.init_std_factor)

        # Pointer mechanism args
        self.pointer_args = pointer_args or PointerMechanismArgs()
        self.pointer_dim = self.pointer_args.pointer_dim or args.dim

        # Compute head_dim and n_heads
        head_dim = dec_args.head_dim or args.dim // dec_args.n_heads
        n_heads = dec_args.n_heads or args.dim // head_dim
        n_kv_heads = dec_args.n_kv_heads or n_heads

        # Use shared embeddings if provided
        if shared_embeddings is not None:
            self.tok_embeddings = shared_embeddings
        else:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.rope_embeddings = RotaryEmbedding(
            theta=dec_args.rope_theta,
            head_dim=head_dim,
            max_seqlen=args.max_decoder_seqlen,
        )

        # Standard decoder layers (from enc_dec)
        self.layers = nn.ModuleList()
        for _ in range(dec_args.n_layers):
            self.layers.append(
                DecoderBlock(
                    dim=args.dim,
                    head_dim=head_dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_dim_multiplier=dec_args.ffn_dim_multiplier,
                    multiple_of=dec_args.multiple_of,
                    norm_eps=dec_args.norm_eps,
                    rope_theta=dec_args.rope_theta,
                )
            )

        self.norm = RMSNorm(args.dim, eps=dec_args.norm_eps)

        # Pointer projection heads
        if self.pointer_args.use_pointer_projection:
            # Project decoder hidden states for pointer computation
            self.decoder_pointer_proj = nn.Linear(args.dim, self.pointer_dim, bias=False)
            # Project encoder hidden states for pointer computation
            self.encoder_pointer_proj = nn.Linear(args.dim, self.pointer_dim, bias=False)
        else:
            self.decoder_pointer_proj = None
            self.encoder_pointer_proj = None

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        target_positions: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            input_ids: [B, dec_seq] decoder token IDs
            encoder_output: [B, enc_seq, D] encoder hidden states
            encoder_input_ids: [B, enc_seq] encoder token IDs (for extracting output)
            encoder_mask: [B, enc_seq] padding mask for encoder (True = valid)
            target_positions: [B, dec_seq] target encoder positions (for training)
            tok_idx: Token indices for incremental decoding
            attn_impl: attention implementation

        Returns:
            If target_positions is provided: (loss, aux_dict)
            Otherwise: pointer_logits [B, dec_seq, enc_seq]
        """
        bsz, seqlen = input_ids.shape
        _, enc_seqlen = encoder_input_ids.shape

        h = self.tok_embeddings(input_ids)
        freq_cis = self.rope_embeddings(seqlen=seqlen, tok_idx=tok_idx)

        # Create causal mask for self-attention
        causal_mask = create_causal_mask(seqlen, attn_impl)

        # Run decoder layers
        for layer in self.layers:
            h = layer(
                h,
                encoder_output,
                freq_cis,
                self_attn_mask=causal_mask,
                encoder_mask=encoder_mask,
                attn_impl=attn_impl,
            )

        h = self.norm(h)  # [B, dec_seq, D]

        # Compute pointer logits via dot product with encoder
        if self.pointer_args.use_pointer_projection:
            h_proj = self.decoder_pointer_proj(h)  # [B, dec_seq, pointer_dim]
            enc_proj = self.encoder_pointer_proj(encoder_output)  # [B, enc_seq, pointer_dim]
        else:
            h_proj = h
            enc_proj = encoder_output

        # Pointer logits: [B, dec_seq, enc_seq]
        pointer_logits = torch.bmm(h_proj, enc_proj.transpose(1, 2))

        # Scale by sqrt(dim) for stability
        pointer_logits = pointer_logits / (self.pointer_dim ** 0.5)

        # Apply temperature
        if self.pointer_args.pointer_temperature != 1.0:
            pointer_logits = pointer_logits / self.pointer_args.pointer_temperature

        # Mask padding positions in encoder
        if encoder_mask is not None:
            # encoder_mask: [B, enc_seq], True for valid
            pointer_logits = pointer_logits.masked_fill(
                ~encoder_mask.unsqueeze(1),  # [B, 1, enc_seq]
                float('-inf')
            )

        if target_positions is not None:
            # Compute cross-entropy loss over encoder positions
            # target_positions: [B, dec_seq] with values in [0, enc_seq) or -100 for padding

            # Flatten for cross-entropy
            pointer_logits_flat = pointer_logits.view(-1, enc_seqlen)  # [B*dec_seq, enc_seq]
            target_flat = target_positions.view(-1)  # [B*dec_seq]

            loss = F.cross_entropy(
                pointer_logits_flat,
                target_flat,
                ignore_index=-100,
                reduction='mean',
            )

            # Compute accuracy for logging
            with torch.no_grad():
                valid_mask = (target_positions != -100)
                predictions = pointer_logits.argmax(dim=-1)  # [B, dec_seq]
                correct = (predictions == target_positions) & valid_mask
                accuracy = correct.sum().float() / valid_mask.sum().float() if valid_mask.sum() > 0 else 0.0

            aux = {
                "pointer_accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
            }

            return loss, aux

        return pointer_logits

    def reset_parameters(self, shared_embeddings: bool = False):
        self.rope_embeddings.reset_parameters()
        self.norm.reset_parameters()

        init_std = self.init_base_std or (self.dim ** (-0.5))

        # Only initialize embeddings if not shared
        if not shared_embeddings:
            nn.init.trunc_normal_(
                self.tok_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        # Initialize pointer projections
        if self.pointer_args.use_pointer_projection:
            nn.init.trunc_normal_(
                self.decoder_pointer_proj.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )
            nn.init.trunc_normal_(
                self.encoder_pointer_proj.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    def init_weights(self, shared_embeddings: bool = False):
        self.reset_parameters(shared_embeddings)
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (self.n_layers + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]
            layer.init_weights(self.init_base_std, factor)


class EncDecPointerTransformer(nn.Module):
    """Full encoder-decoder transformer with pointer mechanism.

    The decoder points to encoder positions instead of generating vocabulary tokens.
    This guarantees extractive behavior - outputs always come from the input.

    Supports three encoder types:
    - "trainable": Full trainable transformer encoder
    - "pretrained": Frozen pretrained encoder (e.g., ModernBERT)
    - "embedding_only": Just embeddings, no encoder layers
    """

    def __init__(
        self,
        args: EncDecTransformerArgs,
        pointer_args: Optional[PointerMechanismArgs] = None,
    ):
        super().__init__()

        self.args = args
        self.dim = args.dim
        self.encoder_type = EncoderType(args.encoder_type)
        self.share_embeddings = args.share_embeddings
        self.pointer_args = pointer_args or PointerMechanismArgs()

        # Build encoder based on type (reuse from enc_dec)
        if self.encoder_type == EncoderType.PRETRAINED:
            self.encoder = PretrainedEncoder(args)
            self.decoder = PointerDecoder(args, self.pointer_args, shared_embeddings=None)
        elif self.encoder_type == EncoderType.EMBEDDING_ONLY:
            self.decoder = PointerDecoder(args, self.pointer_args, shared_embeddings=None)
            self.encoder = EmbeddingOnlyEncoder(
                args,
                shared_embeddings=self.decoder.tok_embeddings if args.share_embeddings else None
            )
        else:  # TRAINABLE
            self.encoder = Encoder(args)
            shared_emb = self.encoder.tok_embeddings if args.share_embeddings else None
            self.decoder = PointerDecoder(args, self.pointer_args, shared_embeddings=shared_emb)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ):
        """
        Args:
            encoder_input_ids: [B, enc_seq] - document/context tokens
            decoder_input_ids: [B, dec_seq] - question tokens (or previous answer tokens)
            target_positions: [B, dec_seq] - target encoder positions (for training)
            encoder_padding_mask: [B, enc_seq] - True for valid encoder positions
            attn_impl: attention implementation

        Returns:
            If target_positions is provided: (loss, aux_dict)
            Otherwise: pointer_logits [B, dec_seq, enc_seq]
        """
        # Encode the document
        encoder_output = self.encoder(
            encoder_input_ids,
            padding_mask=encoder_padding_mask,
            attn_impl=attn_impl,
        )

        # Decode with pointer mechanism
        return self.decoder(
            decoder_input_ids,
            encoder_output,
            encoder_input_ids=encoder_input_ids,
            encoder_mask=encoder_padding_mask,
            target_positions=target_positions,
            attn_impl=attn_impl,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.encoder_type == EncoderType.PRETRAINED:
            self.decoder.reset_parameters(shared_embeddings=False)
        elif self.encoder_type == EncoderType.EMBEDDING_ONLY:
            self.decoder.reset_parameters(shared_embeddings=False)
        else:
            self.decoder.reset_parameters(shared_embeddings=self.share_embeddings)

    def init_weights(self):
        self.encoder.init_weights()
        if self.encoder_type == EncoderType.PRETRAINED:
            self.decoder.init_weights(shared_embeddings=False)
        elif self.encoder_type == EncoderType.EMBEDDING_ONLY:
            self.decoder.init_weights(shared_embeddings=False)
        else:
            self.decoder.init_weights(shared_embeddings=self.share_embeddings)

    def decode_pointer_output(
        self,
        pointer_logits: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_tokenizer,
    ) -> List[str]:
        """Convert pointer logits to text strings.

        Args:
            pointer_logits: [B, dec_seq, enc_seq] logits from forward pass
            encoder_input_ids: [B, enc_seq] encoder token IDs
            encoder_tokenizer: Tokenizer for encoder (e.g., ModernBERT tokenizer)

        Returns:
            List of decoded strings, one per batch item
        """
        # Get predicted positions
        predicted_positions = pointer_logits.argmax(dim=-1)  # [B, dec_seq]

        batch_size = predicted_positions.shape[0]
        decoded_texts = []

        for b in range(batch_size):
            # Gather tokens at predicted positions
            positions = predicted_positions[b]  # [dec_seq]
            token_ids = encoder_input_ids[b].gather(0, positions)  # [dec_seq]

            # Decode using encoder tokenizer
            text = encoder_tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
            decoded_texts.append(text)

        return decoded_texts


def build_pointer_fsdp_grouping_plan(
    model_args: EncDecTransformerArgs,
    pointer_args: Optional[PointerMechanismArgs] = None,
) -> List[Tuple[str, bool]]:
    """Define FSDP sharding groups for pointer encoder-decoder model.

    Unlike the standard encoder-decoder, PointerDecoder doesn't have an 'output' layer.
    Instead it has pointer projection layers (if use_pointer_projection=True).
    """
    group_plan = []
    encoder_type = EncoderType(model_args.encoder_type)
    pointer_args = pointer_args or PointerMechanismArgs()

    # Encoder grouping (same as base)
    if encoder_type == EncoderType.PRETRAINED:
        group_plan.append(("encoder.encoder", False))
        if model_args.pretrained_encoder.encoder_dim != model_args.dim:
            group_plan.append(("encoder.projection", False))
    elif encoder_type == EncoderType.EMBEDDING_ONLY:
        group_plan.append(("encoder.tok_embeddings", False))
        group_plan.append(("encoder.norm", False))
    else:  # TRAINABLE
        group_plan.append(("encoder.tok_embeddings", False))
        for i in range(model_args.encoder.n_layers):
            group_plan.append((f"encoder.layers.{i}", False))
        group_plan.append(("encoder.norm", False))

    # Decoder embeddings
    if encoder_type != EncoderType.TRAINABLE or not model_args.share_embeddings:
        group_plan.append(("decoder.tok_embeddings", False))

    # Decoder layers
    for i in range(model_args.decoder.n_layers):
        group_plan.append((f"decoder.layers.{i}", False))

    # Decoder norm (this is the last layer if no pointer projections)
    group_plan.append(("decoder.norm", not pointer_args.use_pointer_projection))

    # Pointer projections (instead of decoder.output) - only if enabled
    if pointer_args.use_pointer_projection:
        group_plan.append(("decoder.decoder_pointer_proj", True))
        group_plan.append(("decoder.encoder_pointer_proj", True))

    return group_plan
