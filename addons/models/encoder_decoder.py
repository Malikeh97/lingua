"""
Encoder-decoder transformer model.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

from lingua.transformer import (
    RMSNorm,
    Attention,
    FeedForward,
    RotaryEmbedding,
)

from .config import ModelArgs, GenerationArgs
from .attention import SoftmaxAttention, LinearAttention, get_available_variants
from addons.data.collate import TokenizedBatch


class CrossAttentionAdapter(nn.Module):
    """
    Cross-attention adapter injected between self-attention and FFN.

    Output projection is zero-initialized (CEPE-style) so the adapter
    starts as a no-op, preserving pretrained decoder behavior initially.

    Supported attn_type values:
        - "softmax": Standard scaled dot-product attention
        - Any FLA variant: "linear", "gla", "delta_rule", "gated_delta_rule",
          "kda", "retention", "based", "simple_gla", "mamba2", "rwkv6",
          "hgrn2", "abc"
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        attn_type: str = "softmax",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.cross_attn_norm = RMSNorm(dim)

        if attn_type == "softmax":
            self.cross_attn = SoftmaxAttention(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
        else:
            # All other types use LinearAttention with the specified variant
            self.cross_attn = LinearAttention(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                variant=attn_type,
            )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        """
        Args:
            x: decoder hidden states [total_dec, dim]
            encoder_output: encoder hidden states [total_enc, dim]
            cu_seqlens_q: cumulative decoder lengths
            cu_seqlens_kv: cumulative encoder lengths
            max_seqlen_q: max decoder length
            max_seqlen_kv: max encoder length
        """
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(
            x,
            cu_seqlens_q,
            max_seqlen_q,
            kv=encoder_output,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_kv=max_seqlen_kv,
            causal=False,  # Cross-attention is never causal
        )
        return residual + x


class EncoderDecoderBlock(nn.Module):
    """
    Decoder block with self-attention, cross-attention, and FFN.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        cross_attn_type: str = "softmax",
        ffn_dim_multiplier: float = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        # Self-attention
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = Attention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
        )

        # Cross-attention adapter
        self.cross_attn_adapter = CrossAttentionAdapter(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            attn_type=cross_attn_type,
        )

        # FFN
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        freq_cis: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        h = x + self.attention(self.attention_norm(x), freq_cis, mask="causal")

        # Cross-attention
        h = self.cross_attn_adapter(
            h, encoder_output,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
        )

        # FFN
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EncoderDecoder(nn.Module):
    """
    Encoder-decoder transformer with cross-attention.

    Supports:
    - Pretrained encoder (e.g., ModernBERT)
    - Pretrained or from-scratch decoder
    - Multiple cross-attention types via args.cross_attn_type:
        "softmax", "linear", "gla", "delta_rule", "gated_delta_rule",
        "kda", "retention", "based", "simple_gla", "mamba2", "rwkv6",
        "hgrn2", "abc"
    - Packed sequences via cu_seqlens
    - Ring attention via sp_group
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self._init_encoder(args)
        self._init_decoder(args)

    def _init_encoder(self, args: ModelArgs):
        """Initialize pretrained encoder."""
        config = AutoConfig.from_pretrained(args.encoder_name)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "flash_attention_2"
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        self.encoder = AutoModel.from_pretrained(
            args.encoder_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )

        self.encoder_dim = config.hidden_size
        self.encoder_vocab_size = config.vocab_size

        # Freeze encoder if requested
        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.is_pretrained = True
                param.is_pretrained_encoder = True

    def _init_decoder(self, args: ModelArgs):
        """Initialize decoder with cross-attention adapters."""
        if args.decoder_name:
            self._init_pretrained_decoder(args)
        else:
            self._init_from_scratch_decoder(args)

    def _init_from_scratch_decoder(self, args: ModelArgs):
        """Initialize decoder from scratch."""
        self.decoder_dim = self.encoder_dim
        self.decoder_vocab_size = self.encoder_vocab_size
        self.n_decoder_layers = args.num_decoder_layers

        # Infer head config from encoder
        enc_config = AutoConfig.from_pretrained(args.encoder_name)
        n_heads = enc_config.num_attention_heads
        n_kv_heads = getattr(enc_config, "num_key_value_heads", n_heads)
        head_dim = self.decoder_dim // n_heads

        # Embeddings
        self.decoder_tok_embeddings = nn.Embedding(self.decoder_vocab_size, self.decoder_dim)

        # Copy encoder embeddings
        encoder_embeds = self._get_encoder_embeddings()
        if encoder_embeds is not None:
            self.decoder_tok_embeddings.weight.data.copy_(encoder_embeds)

        # RoPE
        self.decoder_rope = RotaryEmbedding(
            theta=10000.0,
            head_dim=head_dim,
            max_seqlen=args.decoder_max_len,
        )

        # Decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            EncoderDecoderBlock(
                dim=self.decoder_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                cross_attn_type=args.cross_attn_type,
            )
            for _ in range(args.num_decoder_layers)
        ])

        # Output
        self.decoder_norm = RMSNorm(self.decoder_dim)
        self.decoder_output = nn.Linear(self.decoder_dim, self.decoder_vocab_size, bias=False)
        self.decoder_output.weight = self.decoder_tok_embeddings.weight

    def _init_pretrained_decoder(self, args: ModelArgs):
        """Initialize decoder from pretrained causal LM with cross-attention adapters."""
        config = AutoConfig.from_pretrained(args.decoder_name)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"

        self.decoder_dim = config.hidden_size
        self.decoder_vocab_size = config.vocab_size
        self.n_decoder_layers = config.num_hidden_layers

        full_model = AutoModelForCausalLM.from_pretrained(
            args.decoder_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )

        # Extract components
        self.decoder_tok_embeddings = full_model.model.embed_tokens
        self.pretrained_decoder_layers = full_model.model.layers
        self.decoder_norm = full_model.model.norm
        self.decoder_output = full_model.lm_head

        # RoPE
        n_heads = config.num_attention_heads
        head_dim = self.decoder_dim // n_heads
        self.decoder_rope = RotaryEmbedding(
            theta=getattr(config, "rope_theta", 10000.0),
            head_dim=head_dim,
            max_seqlen=args.decoder_max_len,
        )

        # Create cross-attention adapters (injected between layers)
        self.cross_attn_adapters = nn.ModuleList([
            CrossAttentionAdapter(
                dim=self.decoder_dim,
                n_heads=args.cross_attn_heads,
                head_dim=self.decoder_dim // args.cross_attn_heads,
                attn_type=args.cross_attn_type,
            )
            for _ in range(self.n_decoder_layers)
        ])

        # Mark pretrained params
        for module in [self.decoder_tok_embeddings, self.pretrained_decoder_layers,
                       self.decoder_norm, self.decoder_output]:
            for p in module.parameters():
                p.is_pretrained = True
                p.is_pretrained_decoder = True

        # Projection if dims don't match
        self.encoder_projection = None
        if self.encoder_dim != self.decoder_dim:
            self.encoder_projection = nn.Linear(self.encoder_dim, self.decoder_dim, bias=False)

        del full_model

    def _get_encoder_embeddings(self) -> Optional[torch.Tensor]:
        """Get encoder embedding weights."""
        if hasattr(self.encoder, "embeddings"):
            if hasattr(self.encoder.embeddings, "word_embeddings"):
                return self.encoder.embeddings.word_embeddings.weight.data.float()
            elif hasattr(self.encoder.embeddings, "tok_embeddings"):
                return self.encoder.embeddings.tok_embeddings.weight.data.float()
        return None

    def encode(
        self,
        batch: TokenizedBatch,
        sp_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Encode input documents.

        Args:
            batch: TokenizedBatch containing encoder_tokens
            sp_group: sequence parallel group

        Returns:
            encoder_hidden: [total_enc_tokens, hidden_dim]
        """
        tokens = batch.encoder_tokens.tokens
        attention_mask = (tokens != 0).long()  # Simple padding mask

        # HF expects [B, T]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.encoder(
            input_ids=tokens,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state

        # Flatten back to packed format
        if hidden.dim() == 3:
            hidden = hidden.view(-1, hidden.shape[-1])

        # Project to decoder dim if needed
        if hasattr(self, "encoder_projection") and self.encoder_projection is not None:
            hidden = self.encoder_projection(hidden)

        return hidden

    def forward(
        self,
        batch: TokenizedBatch,
        sp_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode then decode with cross-attention.

        Args:
            batch: TokenizedBatch containing:
                - encoder_tokens: packed documents
                - decoder_tokens: packed query+target
                - labels: target token ids
                - example_doc_indices: cross-attention mapping
            sp_group: sequence parallel group

        Returns:
            logits: [total_dec_tokens, vocab_size]
        """
        # Encode
        encoder_output = self.encode(batch, sp_group)

        # Decode
        decoder_tokens = batch.decoder_tokens.tokens
        seqlen = decoder_tokens.shape[0]

        h = self.decoder_tok_embeddings(decoder_tokens)
        freq_cis = self.decoder_rope(seqlen=seqlen)

        # Get cu_seqlens for packed attention
        cu_seqlens_q = batch.decoder_tokens.cu_seqlens
        cu_seqlens_kv = batch.encoder_tokens.cu_seqlens
        max_seqlen_q = batch.decoder_tokens.max_seqlen
        max_seqlen_kv = batch.encoder_tokens.max_seqlen

        if hasattr(self, "decoder_layers"):
            # From-scratch decoder with integrated cross-attention
            for layer in self.decoder_layers:
                h = layer(
                    h.unsqueeze(0) if h.dim() == 2 else h,
                    encoder_output.unsqueeze(0) if encoder_output.dim() == 2 else encoder_output,
                    freq_cis,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                )
                if h.dim() == 3:
                    h = h.squeeze(0)
        else:
            # Pretrained decoder with injected cross-attention adapters
            for layer, cross_attn in zip(self.pretrained_decoder_layers, self.cross_attn_adapters):
                # Self-attention through HF layer
                h = h.unsqueeze(0) if h.dim() == 2 else h
                outputs = layer(h, use_cache=False)
                h = outputs[0]
                if h.dim() == 3:
                    h = h.squeeze(0)

                # Cross-attention adapter
                h = cross_attn(
                    h, encoder_output,
                    cu_seqlens_q, cu_seqlens_kv,
                    max_seqlen_q, max_seqlen_kv,
                )

        h = self.decoder_norm(h)
        logits = self.decoder_output(h)

        return logits

    def generate(
        self,
        batch: TokenizedBatch,
        gen_args: GenerationArgs,
    ) -> torch.Tensor:
        """
        Generate tokens given encoded documents.

        Args:
            batch: TokenizedBatch with encoder_tokens and query prefix
            gen_args: generation configuration

        Returns:
            output_ids: [batch_size, max_new_tokens]
        """
        # Pre-compute encoder output
        encoder_output = self.encode(batch)

        tokens = batch.decoder_tokens.tokens.clone()

        for _ in range(gen_args.max_new_tokens):
            # Forward pass
            logits = self.forward(batch)
            next_logits = logits[-1]

            if gen_args.do_sample:
                next_logits = next_logits / gen_args.temperature
                probs = F.softmax(next_logits, dim=-1)
                if gen_args.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > gen_args.top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum()
                    probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=0)
            batch.decoder_tokens.tokens = tokens

        return tokens.unsqueeze(0)

    @classmethod
    def from_pretrained(
        cls,
        encoder_name: str,
        decoder_name: Optional[str],
        args: ModelArgs,
    ) -> "EncoderDecoder":
        """Load pretrained encoder and optionally decoder weights."""
        args.encoder_name = encoder_name
        args.decoder_name = decoder_name
        return cls(args)
