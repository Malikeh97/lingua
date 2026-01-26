# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Copy Mechanism Architecture for Extractive QA.

This module implements an encoder-decoder model with a copy mechanism that allows
the decoder to either generate tokens from vocabulary or copy tokens from the
encoder input. The copy distribution is derived from cross-attention weights.

Key components:
- CopyCrossAttention: Cross-attention that returns attention weights
- CopyDecoderBlock: Decoder block using CopyCrossAttention
- CopyDecoder: Decoder with copy gate and blending logic
- EncDecCopyTransformer: Full encoder-decoder model with copy mechanism
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
    cross_entropy,
    repeat_kv,
    InitStdFactor,
    TiedLinear,
)

# Import shared components from enc_dec
from apps.enc_dec.enc_dec import (
    EncDecTransformerArgs,
    EncoderType,
    PretrainedEncoder,
    Encoder,
    EmbeddingOnlyEncoder,
    create_causal_mask,
    create_encoder_padding_mask,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token_enc_dec,
    load_pretrained_decoder_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class CopyMechanismArgs:
    """Arguments for copy mechanism configuration."""
    copy_gate_init_bias: float = 0.0  # Initial bias for copy gate (positive = favor copying)
    use_last_layer_attn: bool = True  # Use attention from last layer only vs average all layers
    copy_attn_temperature: float = 1.0  # Temperature for copy attention (lower = sharper)


class CopyCrossAttention(nn.Module):
    """Cross-attention that returns attention weights for copy mechanism.

    Similar to CrossAttention in enc_dec.py but returns attention weights
    in addition to the output.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        # Query projection (from decoder)
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        # Key projection (from encoder)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        # Value projection (from encoder)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        # Output projection
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Decoder hidden states [B, dec_seq, D]
            encoder_output: Encoder outputs [B, enc_seq, D]
            encoder_mask: Padding mask for encoder [B, enc_seq] or attention mask
            attn_impl: Attention implementation
            return_attn_weights: Whether to return attention weights

        Returns:
            Output tensor [B, dec_seq, D]
            If return_attn_weights: Also returns attention weights [B, dec_seq, enc_seq]
        """
        bsz, dec_seq_len, _ = x.shape
        _, enc_seq_len, _ = encoder_output.shape

        # Compute Q from decoder, K/V from encoder
        xq = self.wq(x)
        xk = self.wk(encoder_output)
        xv = self.wv(encoder_output)

        output_shape = xq.shape

        # Reshape: B S D -> B S H head_dim
        xq = xq.view(bsz, dec_seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, enc_seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, enc_seq_len, self.n_kv_heads, self.head_dim)

        # Repeat KV for grouped query attention
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # Transpose for attention: B S H D -> B H S D
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Create attention mask from encoder padding mask
        attn_mask = None
        if encoder_mask is not None:
            if encoder_mask.dim() == 2:
                # Boolean padding mask [B, enc_seq]
                attn_mask = create_encoder_padding_mask(
                    encoder_mask, dec_seq_len, self.n_heads, "sdpa"
                )
            else:
                attn_mask = encoder_mask

        # Cast mask to match query dtype
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=xq.dtype)

        if return_attn_weights:
            # Manual attention computation to get weights
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(xq, xk.transpose(-2, -1)) * scale

            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, H, dec_seq, enc_seq]
            output = torch.matmul(attn_weights, xv)

            # Average attention weights across heads for copy mechanism
            attn_weights_avg = attn_weights.mean(dim=1)  # [B, dec_seq, enc_seq]

            output = output.transpose(1, 2).contiguous()
            output = self.wo(output.reshape(output_shape))

            return output, attn_weights_avg
        else:
            # Use efficient SDPA
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                is_causal=False,
            )
            output = output.transpose(1, 2).contiguous()
            return self.wo(output.reshape(output_shape))

    def reset_parameters(self, init_std: Optional[float] = None, factor: float = 1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class CopyDecoderBlock(nn.Module):
    """Decoder block with copy-enabled cross-attention."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
        norm_eps: float,
        rope_theta: float,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        # Causal self-attention
        self.self_attention = Attention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
        )

        # Cross-attention with attention weights output
        self.cross_attention = CopyCrossAttention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.self_attention_norm = RMSNorm(dim, eps=norm_eps)
        self.cross_attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        freq_cis: torch.Tensor,
        self_attn_mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
        return_cross_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            If return_cross_attn_weights: (output, cross_attn_weights)
            Otherwise: output
        """
        # 1. Causal self-attention
        h = x + self.self_attention(
            self.self_attention_norm(x),
            freq_cis,
            mask=self_attn_mask,
            attn_impl=attn_impl,
        )

        # 2. Cross-attention to encoder
        if return_cross_attn_weights:
            cross_out, cross_attn_weights = self.cross_attention(
                self.cross_attention_norm(h),
                encoder_output,
                encoder_mask=encoder_mask,
                attn_impl=attn_impl,
                return_attn_weights=True,
            )
            h = h + cross_out
        else:
            h = h + self.cross_attention(
                self.cross_attention_norm(h),
                encoder_output,
                encoder_mask=encoder_mask,
                attn_impl=attn_impl,
                return_attn_weights=False,
            )

        # 3. Feed-forward
        out = h + self.feed_forward(self.ffn_norm(h))

        if return_cross_attn_weights:
            return out, cross_attn_weights
        return out

    def init_weights(self, init_std: Optional[float] = None, factor: float = 1.0):
        self.self_attention.reset_parameters(init_std, factor)
        self.self_attention_norm.reset_parameters()
        self.cross_attention.reset_parameters(init_std, factor)
        self.cross_attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class CopyDecoder(nn.Module):
    """Decoder with copy mechanism for extractive QA.

    Combines generation distribution (over vocabulary) with copy distribution
    (over encoder positions) using a learned copy gate.
    """

    def __init__(
        self,
        args: EncDecTransformerArgs,
        copy_args: Optional[CopyMechanismArgs] = None,
        shared_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()

        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.max_seqlen = args.max_decoder_seqlen
        self.weight_tying = args.weight_tying

        dec_args = args.decoder
        self.n_layers = dec_args.n_layers
        self.init_base_std = dec_args.init_base_std
        self.init_std_factor = InitStdFactor(dec_args.init_std_factor)

        # Copy mechanism args
        self.copy_args = copy_args or CopyMechanismArgs()

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

        # Decoder layers with copy-enabled cross-attention
        self.layers = nn.ModuleList()
        for _ in range(dec_args.n_layers):
            self.layers.append(
                CopyDecoderBlock(
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

        # Generation head (vocabulary projection)
        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Copy gate: decides between generation and copying
        # Output is sigmoid, so positive bias = favor copying
        self.copy_gate = nn.Linear(args.dim, 1, bias=True)
        nn.init.constant_(self.copy_gate.bias, self.copy_args.copy_gate_init_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            input_ids: [B, dec_seq] decoder token IDs
            encoder_output: [B, enc_seq, D] encoder hidden states
            encoder_input_ids: [B, enc_seq] encoder token IDs (for copy mechanism)
            encoder_mask: [B, enc_seq] padding mask for encoder
            target: [B, dec_seq] target labels (optional, for training)
            tok_idx: Token indices for incremental decoding
            attn_impl: attention implementation

        Returns:
            If target is provided: (loss, aux_dict with copy stats)
            Otherwise: final probabilities [B, dec_seq, vocab_size]
        """
        bsz, seqlen = input_ids.shape
        _, enc_seqlen = encoder_input_ids.shape

        h = self.tok_embeddings(input_ids)
        freq_cis = self.rope_embeddings(seqlen=seqlen, tok_idx=tok_idx)

        # Create causal mask for self-attention
        causal_mask = create_causal_mask(seqlen, attn_impl)

        # Run decoder layers, collecting cross-attention weights from last layer
        cross_attn_weights = None
        for i, layer in enumerate(self.layers):
            is_last_layer = (i == len(self.layers) - 1)
            if is_last_layer and self.copy_args.use_last_layer_attn:
                h, cross_attn_weights = layer(
                    h,
                    encoder_output,
                    freq_cis,
                    self_attn_mask=causal_mask,
                    encoder_mask=encoder_mask,
                    attn_impl=attn_impl,
                    return_cross_attn_weights=True,
                )
            else:
                h = layer(
                    h,
                    encoder_output,
                    freq_cis,
                    self_attn_mask=causal_mask,
                    encoder_mask=encoder_mask,
                    attn_impl=attn_impl,
                    return_cross_attn_weights=False,
                )

        h = self.norm(h)

        # 1. Generation distribution
        gen_logits = self.output(h)  # [B, dec_seq, vocab_size]

        # 2. Copy gate
        copy_gate = torch.sigmoid(self.copy_gate(h))  # [B, dec_seq, 1]

        # 3. Copy distribution from cross-attention weights
        # cross_attn_weights: [B, dec_seq, enc_seq]
        if cross_attn_weights is None:
            # Fallback: use uniform distribution if weights not available
            cross_attn_weights = torch.ones(bsz, seqlen, enc_seqlen, device=h.device) / enc_seqlen

        # Apply temperature to copy attention
        if self.copy_args.copy_attn_temperature != 1.0:
            cross_attn_weights = F.softmax(
                torch.log(cross_attn_weights + 1e-10) / self.copy_args.copy_attn_temperature,
                dim=-1
            )

        # Mask padding in copy distribution
        if encoder_mask is not None:
            # encoder_mask: [B, enc_seq], True for valid positions
            cross_attn_weights = cross_attn_weights.masked_fill(
                ~encoder_mask.unsqueeze(1), 0.0
            )
            # Renormalize
            cross_attn_weights = cross_attn_weights / (cross_attn_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # 4. Convert to probabilities
        gen_probs = F.softmax(gen_logits, dim=-1)  # [B, dec_seq, vocab_size]

        # 5. Scatter copy distribution to vocabulary space
        # copy_probs[b, t, v] = sum over positions j where encoder_input_ids[b, j] == v
        copy_probs = torch.zeros_like(gen_probs)  # [B, dec_seq, vocab_size]

        # Expand encoder_input_ids for scatter: [B, enc_seq] -> [B, dec_seq, enc_seq]
        encoder_ids_expanded = encoder_input_ids.unsqueeze(1).expand(-1, seqlen, -1)

        # Scatter add: for each decoder position, add attention weight to corresponding vocab token
        copy_probs.scatter_add_(
            dim=2,
            index=encoder_ids_expanded,
            src=cross_attn_weights,
        )

        # 6. Blend generation and copy distributions
        final_probs = (1 - copy_gate) * gen_probs + copy_gate * copy_probs  # [B, dec_seq, vocab_size]

        if target is not None:
            # Compute negative log likelihood
            # Handle padding (-100 in target) - must replace before gather to avoid index out of bounds
            valid_mask = (target != -100)
            safe_target = target.clone()
            safe_target[~valid_mask] = 0  # Replace -100 with valid index

            # Gather probabilities for target tokens
            target_probs = final_probs.gather(dim=2, index=safe_target.unsqueeze(-1)).squeeze(-1)  # [B, dec_seq]
            target_probs = target_probs.masked_fill(~valid_mask, 1.0)  # Set padding to 1 to avoid log(0)

            # NLL loss
            loss = -torch.log(target_probs + 1e-10)
            loss = loss.masked_fill(~valid_mask, 0.0)
            loss = loss.sum() / valid_mask.sum()

            # Auxiliary info for logging
            aux = {
                "copy_gate_mean": copy_gate.mean().item(),
                "copy_gate_std": copy_gate.std().item(),
            }

            return loss, aux

        return final_probs

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

        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        # Initialize copy gate
        nn.init.trunc_normal_(
            self.copy_gate.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        nn.init.constant_(self.copy_gate.bias, self.copy_args.copy_gate_init_bias)

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


class EncDecCopyTransformer(nn.Module):
    """Full encoder-decoder transformer with copy mechanism.

    Supports three encoder types:
    - "trainable": Full trainable transformer encoder
    - "pretrained": Frozen pretrained encoder (e.g., ModernBERT)
    - "embedding_only": Just embeddings, no encoder layers
    """

    def __init__(
        self,
        args: EncDecTransformerArgs,
        copy_args: Optional[CopyMechanismArgs] = None,
    ):
        super().__init__()

        self.args = args
        self.dim = args.dim
        self.encoder_type = EncoderType(args.encoder_type)
        self.share_embeddings = args.share_embeddings
        self.copy_args = copy_args or CopyMechanismArgs()

        # Build encoder based on type (reuse from enc_dec)
        if self.encoder_type == EncoderType.PRETRAINED:
            self.encoder = PretrainedEncoder(args)
            self.decoder = CopyDecoder(args, self.copy_args, shared_embeddings=None)
        elif self.encoder_type == EncoderType.EMBEDDING_ONLY:
            self.decoder = CopyDecoder(args, self.copy_args, shared_embeddings=None)
            self.encoder = EmbeddingOnlyEncoder(
                args,
                shared_embeddings=self.decoder.tok_embeddings if args.share_embeddings else None
            )
        else:  # TRAINABLE
            self.encoder = Encoder(args)
            shared_emb = self.encoder.tok_embeddings if args.share_embeddings else None
            self.decoder = CopyDecoder(args, self.copy_args, shared_embeddings=shared_emb)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_target: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ):
        """
        Args:
            encoder_input_ids: [B, enc_seq] - document/context tokens
            decoder_input_ids: [B, dec_seq] - question + answer tokens
            decoder_target: [B, dec_seq] - target labels (answer only, question masked)
            encoder_padding_mask: [B, enc_seq] - True for valid encoder positions
            attn_impl: attention implementation

        Returns:
            If decoder_target is provided: (loss, aux_dict)
            Otherwise: probabilities [B, dec_seq, vocab_size]
        """
        # Encode the document
        encoder_output = self.encoder(
            encoder_input_ids,
            padding_mask=encoder_padding_mask,
            attn_impl=attn_impl,
        )

        # Decode with copy mechanism
        return self.decoder(
            decoder_input_ids,
            encoder_output,
            encoder_input_ids=encoder_input_ids,
            encoder_mask=encoder_padding_mask,
            target=decoder_target,
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


def build_copy_fsdp_grouping_plan(model_args: EncDecTransformerArgs) -> List[Tuple[str, bool]]:
    """Define FSDP sharding groups for copy encoder-decoder model.

    Same as enc_dec but with copy_gate added.
    """
    # Reuse the base plan
    return build_fsdp_grouping_plan(model_args)
