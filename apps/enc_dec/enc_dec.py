# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    create_block_mask,
)

from lingua.transformer import (
    RMSNorm,
    FeedForward,
    RotaryEmbedding,
    Attention,
    cross_entropy,
    apply_rotary_emb,
    repeat_kv,
    InitStdFactor,
    TiedLinear,
)
from lingua import probe

flex_attention_comp = torch.compile(flex_attention)


class AttentionType(Enum):
    """Types of attention mechanisms available."""
    SELF_ATTENTION = "self"
    CROSS_ATTENTION = "cross"


class EncoderType(Enum):
    """Types of encoder architectures available."""
    TRAINABLE = "trainable"  # Full trainable encoder (current implementation)
    PRETRAINED = "pretrained"  # Frozen pretrained encoder (e.g., ModernBERT)
    EMBEDDING_ONLY = "embedding_only"  # Just embeddings, no encoder layers


@dataclass
class EncoderArgs:
    """Encoder-specific transformer arguments."""
    n_layers: int = 6
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


@dataclass
class DecoderArgs:
    """Decoder-specific transformer arguments."""
    n_layers: int = 6
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"


@dataclass
class PretrainedEncoderArgs:
    """Arguments for pretrained encoder (e.g., ModernBERT).

    The encoder is loaded from HuggingFace and kept frozen.
    A projection layer maps from encoder_dim to decoder dim if needed.
    """
    model_name: str = "answerdotai/ModernBERT-base"  # HuggingFace model name
    encoder_dim: int = 768  # Hidden dim of the pretrained model
    pooling: str = "none"  # "none" (use all tokens), "mean", "cls"
    use_flash_attention: bool = True  # Use flash attention in ModernBERT if available


@dataclass
class PretrainedDecoderArgs:
    """Arguments for pretrained decoder initialization from HuggingFace.

    Loads weights from a HuggingFace causal LM model (e.g., LLaMA-style).
    Cross-attention layers are randomly initialized since they don't exist
    in standard causal LM models.
    """
    model_name: str = ""  # HuggingFace model name/path (empty = no pretrained decoder)
    freeze_pretrained: bool = False  # Whether to freeze loaded weights (cross-attention always trainable)


@dataclass
class EncDecTransformerArgs:
    """Combined encoder-decoder configuration.

    The `dim` parameter is shared between encoder and decoder
    to ensure cross-attention compatibility.
    """
    dim: int = 512
    max_encoder_seqlen: int = 2048
    max_decoder_seqlen: int = 512

    # Encoder type: "trainable", "pretrained", or "embedding_only"
    encoder_type: str = "trainable"

    encoder: EncoderArgs = field(default_factory=EncoderArgs)
    decoder: DecoderArgs = field(default_factory=DecoderArgs)
    pretrained_encoder: PretrainedEncoderArgs = field(default_factory=PretrainedEncoderArgs)
    pretrained_decoder: PretrainedDecoderArgs = field(default_factory=PretrainedDecoderArgs)

    seed: int = 42
    vocab_size: int = -1
    weight_tying: bool = False
    share_embeddings: bool = True  # Only used when encoder_type="trainable"


def create_causal_mask(seqlen: int, attn_impl: str) -> Union[BlockMask, AttentionBias, str]:
    """Create a causal mask for decoder self-attention."""
    if attn_impl == "fmha":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(f"Attention implementation {attn_impl} not supported")


def create_encoder_padding_mask(
    padding_mask: torch.Tensor,
    dec_seq_len: int,
    n_heads: int,
    attn_impl: str = "sdpa",
) -> Optional[torch.Tensor]:
    """Create attention mask from encoder padding mask for cross-attention.

    Args:
        padding_mask: [B, enc_seq] boolean tensor, True for valid positions
        dec_seq_len: decoder sequence length
        n_heads: number of attention heads
        attn_impl: attention implementation type

    Returns:
        Attention mask suitable for the specified attention implementation
    """
    if padding_mask is None:
        return None

    # Expand padding mask for cross-attention: [B, 1, 1, enc_seq]
    # This allows each decoder position to attend to all valid encoder positions
    attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)

    if attn_impl == "sdpa":
        # For SDPA, we need [B, n_heads, dec_seq, enc_seq]
        attn_mask = attn_mask.expand(-1, n_heads, dec_seq_len, -1)
        # Convert to float mask: 0 for valid, -inf for invalid
        attn_mask = attn_mask.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        return attn_mask
    elif attn_impl == "fmha":
        # xformers uses AttentionBias
        # For simplicity, return None and handle padding via sequence lengths
        return None
    else:
        return None


class CrossAttention(nn.Module):
    """Cross-attention mechanism where queries come from decoder,
    keys and values come from encoder outputs.

    Key differences from self-attention:
    - No RoPE applied (encoder positions are independent of decoder positions)
    - K/V from encoder memory, Q from decoder hidden states
    - No causal masking (full attention over encoder sequence)
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
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder hidden states [B, dec_seq, D]
            encoder_output: Encoder outputs [B, enc_seq, D]
            encoder_mask: Padding mask for encoder [B, enc_seq] or attention mask
            attn_impl: Attention implementation ("sdpa", "fmha", "flex_attention")

        Returns:
            Output tensor [B, dec_seq, D]
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

        # No RoPE for cross-attention - positions are independent

        # Handle KV cache for generation if present
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv)

        # Repeat KV for grouped query attention
        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "sdpa":
            # Transpose for SDPA: B S H D -> B H S D
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # Create attention mask from encoder padding mask
            attn_mask = None
            if encoder_mask is not None:
                if encoder_mask.dim() == 2:
                    # Boolean padding mask [B, enc_seq]
                    attn_mask = create_encoder_padding_mask(
                        encoder_mask, dec_seq_len, self.n_heads, attn_impl
                    )
                else:
                    # Already an attention mask
                    attn_mask = encoder_mask

            # Cast mask to match query dtype for mixed precision training
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=xq.dtype)

            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                is_causal=False,  # Cross-attention is never causal
            )
            output = output.transpose(1, 2).contiguous()

        elif attn_impl == "fmha":
            # xformers memory efficient attention
            # Note: For cross-attention, we don't use causal mask
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=None)

        elif attn_impl == "flex_attention":
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            # No block mask for cross-attention (full attention)
            output = flex_attention_comp(xq, xk, xv, block_mask=None)
            output = output.transpose(1, 2).contiguous()
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported")

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


class EncoderBlock(nn.Module):
    """Encoder block: bidirectional self-attention + FFN."""

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

        self.attention = Attention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # Bidirectional self-attention (no causal mask)
        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            mask=mask,  # None for bidirectional
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std: Optional[float] = None, factor: float = 1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention + FFN."""

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

        # Cross-attention to encoder
        self.cross_attention = CrossAttention(
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
    ) -> torch.Tensor:
        # 1. Causal self-attention
        h = x + self.self_attention(
            self.self_attention_norm(x),
            freq_cis,
            mask=self_attn_mask,
            attn_impl=attn_impl,
        )

        # 2. Cross-attention to encoder
        h = h + self.cross_attention(
            self.cross_attention_norm(h),
            encoder_output,
            encoder_mask=encoder_mask,
            attn_impl=attn_impl,
        )

        # 3. Feed-forward
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std: Optional[float] = None, factor: float = 1.0):
        self.self_attention.reset_parameters(init_std, factor)
        self.self_attention_norm.reset_parameters()
        self.cross_attention.reset_parameters(init_std, factor)
        self.cross_attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class Encoder(nn.Module):
    """Full encoder stack with embeddings."""

    def __init__(self, args: EncDecTransformerArgs):
        super().__init__()

        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.max_seqlen = args.max_encoder_seqlen

        enc_args = args.encoder
        self.n_layers = enc_args.n_layers
        self.init_base_std = enc_args.init_base_std
        self.init_std_factor = InitStdFactor(enc_args.init_std_factor)

        # Compute head_dim and n_heads
        head_dim = enc_args.head_dim or args.dim // enc_args.n_heads
        n_heads = enc_args.n_heads or args.dim // head_dim
        n_kv_heads = enc_args.n_kv_heads or n_heads

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.rope_embeddings = RotaryEmbedding(
            theta=enc_args.rope_theta,
            head_dim=head_dim,
            max_seqlen=args.max_encoder_seqlen,
        )

        self.layers = nn.ModuleList()
        for _ in range(enc_args.n_layers):
            self.layers.append(
                EncoderBlock(
                    dim=args.dim,
                    head_dim=head_dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_dim_multiplier=enc_args.ffn_dim_multiplier,
                    multiple_of=enc_args.multiple_of,
                    norm_eps=enc_args.norm_eps,
                    rope_theta=enc_args.rope_theta,
                )
            )

        self.norm = RMSNorm(args.dim, eps=enc_args.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, enc_seq] token IDs
            padding_mask: [B, enc_seq] boolean mask (True for valid tokens)
            attn_impl: attention implementation

        Returns:
            Encoder output [B, enc_seq, D]
        """
        bsz, seqlen = input_ids.shape

        h = self.tok_embeddings(input_ids)
        freq_cis = self.rope_embeddings(seqlen=seqlen)

        # Bidirectional attention - no mask needed
        for layer in self.layers:
            h = layer(h, freq_cis, mask=None, attn_impl=attn_impl)

        return self.norm(h)

    def reset_parameters(self):
        self.rope_embeddings.reset_parameters()
        self.norm.reset_parameters()

        init_std = self.init_base_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (self.n_layers + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]
            layer.init_weights(self.init_base_std, factor)


class PretrainedEncoder(nn.Module):
    """Frozen pretrained encoder (e.g., ModernBERT) with projection layer.

    Loads a HuggingFace model and keeps it frozen during training.
    Adds a trainable projection layer if encoder_dim != decoder_dim.
    """

    def __init__(self, args: EncDecTransformerArgs):
        super().__init__()

        self.dim = args.dim  # Target dimension (decoder's dim)
        self.max_seqlen = args.max_encoder_seqlen

        pretrained_args = args.pretrained_encoder
        self.model_name = pretrained_args.model_name
        self.encoder_dim = pretrained_args.encoder_dim
        self.pooling = pretrained_args.pooling
        self.use_flash_attention = pretrained_args.use_flash_attention

        # Load pretrained model from HuggingFace
        # Defer import to avoid dependency if not using pretrained encoder
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(self.model_name)
        # Enable flash attention if requested and supported
        if self.use_flash_attention:
            config.attn_implementation = "flash_attention_2"

        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Projection layer if dimensions don't match
        self.projection = None
        if self.encoder_dim != self.dim:
            self.projection = nn.Linear(self.encoder_dim, self.dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",  # Ignored, HF model handles its own attention
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, enc_seq] token IDs
            padding_mask: [B, enc_seq] boolean mask (True for valid tokens)
            attn_impl: Ignored for pretrained encoder

        Returns:
            Encoder output [B, enc_seq, D] where D is args.dim
        """
        # Convert padding mask to attention mask format expected by HuggingFace
        # HF expects: 1 for tokens to attend to, 0 for tokens to ignore
        attention_mask = None
        if padding_mask is not None:
            attention_mask = padding_mask.long()

        # Run through frozen encoder
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs.last_hidden_state  # [B, seq, encoder_dim]

        # Apply pooling if specified
        if self.pooling == "mean":
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).float()
                hidden_states = (hidden_states * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            else:
                hidden_states = hidden_states.mean(dim=1, keepdim=True)
        elif self.pooling == "cls":
            hidden_states = hidden_states[:, :1, :]  # Take first token

        # Project to decoder dimension if needed (this is trainable)
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)

        return hidden_states

    def reset_parameters(self):
        """Only reset the projection layer (encoder stays frozen)."""
        if self.projection is not None:
            init_std = self.dim ** (-0.5)
            nn.init.trunc_normal_(
                self.projection.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    def init_weights(self):
        self.reset_parameters()


class EmbeddingOnlyEncoder(nn.Module):
    """Simple encoder that only applies embeddings (no transformer layers).

    Uses the decoder's embedding layer to embed documents.
    Optionally adds a learned projection.
    """

    def __init__(
        self,
        args: EncDecTransformerArgs,
        shared_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()

        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.max_seqlen = args.max_encoder_seqlen

        # Use shared embeddings if provided, otherwise create new ones
        if shared_embeddings is not None:
            self.tok_embeddings = shared_embeddings
            self._shared = True
        else:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
            self._shared = False

        # Optional: Add layer norm for stability
        self.norm = RMSNorm(args.dim, eps=1e-5)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, enc_seq] token IDs
            padding_mask: Unused for embedding-only encoder
            attn_impl: Unused

        Returns:
            Embeddings [B, enc_seq, D]
        """
        h = self.tok_embeddings(input_ids)
        return self.norm(h)

    def reset_parameters(self):
        self.norm.reset_parameters()
        if not self._shared:
            init_std = self.dim ** (-0.5)
            nn.init.trunc_normal_(
                self.tok_embeddings.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

    def init_weights(self):
        self.reset_parameters()


class Decoder(nn.Module):
    """Full decoder stack with embeddings and output projection."""

    def __init__(
        self,
        args: EncDecTransformerArgs,
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

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, dec_seq] token IDs
            encoder_output: [B, enc_seq, D] encoder hidden states
            encoder_mask: [B, enc_seq] padding mask for encoder
            target: [B, dec_seq] target labels (optional, for training)
            tok_idx: Token indices for incremental decoding
            attn_impl: attention implementation

        Returns:
            If target is provided: cross-entropy loss
            Otherwise: logits [B, dec_seq, vocab_size]
        """
        bsz, seqlen = input_ids.shape

        h = self.tok_embeddings(input_ids)
        freq_cis = self.rope_embeddings(seqlen=seqlen, tok_idx=tok_idx)

        # Create causal mask for self-attention
        causal_mask = create_causal_mask(seqlen, attn_impl)

        for layer in self.layers:
            h = layer(
                h,
                encoder_output,
                freq_cis,
                self_attn_mask=causal_mask,
                encoder_mask=encoder_mask,
                attn_impl=attn_impl,
            )

        logits = self.output(self.norm(h))

        if target is not None:
            return cross_entropy(logits, target, ignore_index=-100)
        return logits

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


def load_pretrained_decoder_weights(
    decoder: Decoder,
    model_name: str,
    freeze_pretrained: bool = False,
    init_base_std: Optional[float] = None,
    seed: int = 42,
) -> Tuple[int, int]:
    """Load weights from a HuggingFace causal LM model into the decoder.

    Maps weights from LLaMA-style HF models to the enc_dec decoder.
    Cross-attention layers are randomly initialized since they don't exist in HF models.

    Args:
        decoder: The Decoder module to load weights into
        model_name: HuggingFace model name/path
        freeze_pretrained: Whether to freeze loaded weights (cross-attention always trainable)
        init_base_std: Base std for random initialization of cross-attention
        seed: Random seed for cross-attention initialization

    Returns:
        Tuple of (loaded_params, initialized_params) counts
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    import logging

    logger = logging.getLogger()
    logger.info(f"Loading pretrained decoder weights from: {model_name}")

    # Load HuggingFace model
    config = AutoConfig.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    hf_state_dict = hf_model.state_dict()
    decoder_state_dict = decoder.state_dict()

    # Weight mapping: HF -> enc_dec decoder
    # LLaMA-style naming convention
    weight_mapping = {}

    # Embeddings
    if "model.embed_tokens.weight" in hf_state_dict:
        weight_mapping["tok_embeddings.weight"] = "model.embed_tokens.weight"

    # Final norm
    if "model.norm.weight" in hf_state_dict:
        weight_mapping["norm.weight"] = "model.norm.weight"

    # LM head (output projection)
    if "lm_head.weight" in hf_state_dict:
        weight_mapping["output.weight"] = "lm_head.weight"

    # Layer-wise mappings
    n_layers = decoder.n_layers
    for i in range(n_layers):
        # Self-attention projections
        weight_mapping[f"layers.{i}.self_attention.wq.weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
        weight_mapping[f"layers.{i}.self_attention.wk.weight"] = f"model.layers.{i}.self_attn.k_proj.weight"
        weight_mapping[f"layers.{i}.self_attention.wv.weight"] = f"model.layers.{i}.self_attn.v_proj.weight"
        weight_mapping[f"layers.{i}.self_attention.wo.weight"] = f"model.layers.{i}.self_attn.o_proj.weight"

        # FFN (SwiGLU style)
        weight_mapping[f"layers.{i}.feed_forward.w1.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
        weight_mapping[f"layers.{i}.feed_forward.w2.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
        weight_mapping[f"layers.{i}.feed_forward.w3.weight"] = f"model.layers.{i}.mlp.up_proj.weight"

        # Layer norms
        weight_mapping[f"layers.{i}.self_attention_norm.weight"] = f"model.layers.{i}.input_layernorm.weight"
        weight_mapping[f"layers.{i}.ffn_norm.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"

    # Load matched weights
    loaded_params = 0
    mismatched_params = []

    for dec_key, hf_key in weight_mapping.items():
        if hf_key in hf_state_dict and dec_key in decoder_state_dict:
            hf_weight = hf_state_dict[hf_key]
            dec_weight = decoder_state_dict[dec_key]

            # Check shape compatibility
            if hf_weight.shape == dec_weight.shape:
                decoder_state_dict[dec_key] = hf_weight.clone()
                loaded_params += hf_weight.numel()
            else:
                mismatched_params.append((dec_key, dec_weight.shape, hf_weight.shape))
                logger.warning(
                    f"Shape mismatch for {dec_key}: decoder {dec_weight.shape} vs HF {hf_weight.shape}"
                )

    # Load state dict with matched weights
    decoder.load_state_dict(decoder_state_dict, strict=False)

    # Initialize cross-attention layers randomly
    initialized_params = 0
    init_std = init_base_std or (decoder.dim ** (-0.5))

    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(seed)
        for i, layer in enumerate(decoder.layers):
            # Initialize cross-attention weights
            layer.cross_attention.reset_parameters(init_std)
            layer.cross_attention_norm.reset_parameters()

            # Count cross-attention params
            for param in layer.cross_attention.parameters():
                initialized_params += param.numel()
            for param in layer.cross_attention_norm.parameters():
                initialized_params += param.numel()

    # Initialize rope embeddings (not in HF checkpoint)
    decoder.rope_embeddings.reset_parameters()

    # Optionally freeze pretrained weights
    if freeze_pretrained:
        logger.info("Freezing pretrained decoder weights (cross-attention remains trainable)")
        for name, param in decoder.named_parameters():
            # Keep cross-attention trainable
            if "cross_attention" not in name:
                param.requires_grad = False

    logger.info(f"Loaded {loaded_params:,} parameters from pretrained model")
    logger.info(f"Randomly initialized {initialized_params:,} parameters (cross-attention)")
    if mismatched_params:
        logger.warning(f"Skipped {len(mismatched_params)} parameters due to shape mismatch")

    # Clean up HF model
    del hf_model
    del hf_state_dict

    return loaded_params, initialized_params


class EncDecTransformer(nn.Module):
    """Full encoder-decoder transformer model.

    Supports three encoder types:
    - "trainable": Full trainable transformer encoder (original implementation)
    - "pretrained": Frozen pretrained encoder (e.g., ModernBERT)
    - "embedding_only": Just embeddings, no transformer layers
    """

    def __init__(self, args: EncDecTransformerArgs):
        super().__init__()

        self.args = args
        self.dim = args.dim
        self.encoder_type = EncoderType(args.encoder_type)
        self.share_embeddings = args.share_embeddings

        # Build encoder based on type
        if self.encoder_type == EncoderType.PRETRAINED:
            self.encoder = PretrainedEncoder(args)
            # For pretrained encoder, decoder always has its own embeddings
            self.decoder = Decoder(args, shared_embeddings=None)
        elif self.encoder_type == EncoderType.EMBEDDING_ONLY:
            # Build decoder first, then share embeddings with encoder
            self.decoder = Decoder(args, shared_embeddings=None)
            self.encoder = EmbeddingOnlyEncoder(
                args,
                shared_embeddings=self.decoder.tok_embeddings if args.share_embeddings else None
            )
        else:  # TRAINABLE (default)
            self.encoder = Encoder(args)
            shared_emb = self.encoder.tok_embeddings if args.share_embeddings else None
            self.decoder = Decoder(args, shared_embeddings=shared_emb)

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
            If decoder_target is provided: cross-entropy loss
            Otherwise: logits [B, dec_seq, vocab_size]
        """
        # Encode the document
        encoder_output = self.encoder(
            encoder_input_ids,
            padding_mask=encoder_padding_mask,
            attn_impl=attn_impl,
        )

        # Decode with cross-attention
        return self.decoder(
            decoder_input_ids,
            encoder_output,
            encoder_mask=encoder_padding_mask,
            target=decoder_target,
            attn_impl=attn_impl,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # For pretrained encoder, decoder embeddings are never shared
        # For embedding_only, share_embeddings controls whether encoder uses decoder's embeddings
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


def get_num_flop_per_token_enc_dec(
    encoder_params: int,
    decoder_params: int,
    encoder_layers: int,
    decoder_layers: int,
    dim: int,
    encoder_seq_len: int,
    decoder_seq_len: int,
) -> int:
    """Calculate FLOPs per token for encoder-decoder model."""
    # Encoder FLOPs (bidirectional attention)
    encoder_attn_flops = 3.5 * (4 * encoder_layers * encoder_seq_len * dim)
    encoder_flops = 6 * encoder_params + encoder_attn_flops

    # Decoder FLOPs (causal self-attention + cross-attention)
    decoder_self_attn_flops = 3.5 * (4 * decoder_layers * decoder_seq_len * dim // 2)  # Causal
    decoder_cross_attn_flops = 3.5 * (4 * decoder_layers * encoder_seq_len * dim)  # Full attention
    decoder_flops = 6 * decoder_params + decoder_self_attn_flops + decoder_cross_attn_flops

    return encoder_flops + decoder_flops


def build_fsdp_grouping_plan(model_args: EncDecTransformerArgs) -> List[Tuple[str, bool]]:
    """Define FSDP sharding groups for encoder-decoder model."""
    group_plan = []
    encoder_type = EncoderType(model_args.encoder_type)

    if encoder_type == EncoderType.PRETRAINED:
        # For pretrained encoder, put entire frozen encoder in one group
        # Only the projection layer (if any) is trainable
        group_plan.append(("encoder.encoder", False))  # Frozen HF model
        if model_args.pretrained_encoder.encoder_dim != model_args.dim:
            group_plan.append(("encoder.projection", False))

    elif encoder_type == EncoderType.EMBEDDING_ONLY:
        # Simple encoder: just embeddings and norm
        group_plan.append(("encoder.tok_embeddings", False))
        group_plan.append(("encoder.norm", False))

    else:  # TRAINABLE
        # Encoder embeddings
        group_plan.append(("encoder.tok_embeddings", False))

        # Encoder layers
        for i in range(model_args.encoder.n_layers):
            group_plan.append((f"encoder.layers.{i}", False))

        # Encoder norm
        group_plan.append(("encoder.norm", False))

    # Decoder embeddings (always separate for pretrained/embedding_only)
    if encoder_type != EncoderType.TRAINABLE or not model_args.share_embeddings:
        group_plan.append(("decoder.tok_embeddings", False))

    # Decoder layers
    for i in range(model_args.decoder.n_layers):
        group_plan.append((f"decoder.layers.{i}", False))

    # Decoder norm and output
    group_plan.append(("decoder.norm", False))
    group_plan.append(("decoder.output", True))

    return group_plan


def get_no_recompute_ops():
    """Optional policy for activation checkpointing."""
    return None
