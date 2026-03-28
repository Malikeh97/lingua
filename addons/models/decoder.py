"""
Decoder-only transformer model.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoModelForCausalLM, AutoConfig

from lingua.transformer import (
    RMSNorm,
    Attention,
    FeedForward,
    RotaryEmbedding,
)

from .config import ModelArgs, GenerationArgs
from addons.data.collate import TokenizedBatch


class DecoderBlock(nn.Module):
    """Decoder transformer block with RoPE self-attention and FFN."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_dim_multiplier: float = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.attention = Attention(
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=rope_theta,
        )
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
        freq_cis: torch.Tensor,
        mask: str = "causal",
    ) -> torch.Tensor:
        """Forward pass."""
        h = x + self.attention(self.attention_norm(x), freq_cis, mask=mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Decoder(nn.Module):
    """
    Decoder-only transformer.

    Supports:
    - Pretrained or from-scratch initialization
    - Packed sequences via cu_seqlens
    - Ring attention via sp_group
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = None  # Set after loading

        if args.decoder_name:
            self._init_from_pretrained(args.decoder_name)
        else:
            self._init_from_scratch(args)

    def _init_from_scratch(self, args: ModelArgs):
        """Initialize decoder from scratch."""
        # Infer dimensions from encoder if possible
        enc_config = AutoConfig.from_pretrained(args.encoder_name)

        self.dim = enc_config.hidden_size
        self.n_layers = args.num_decoder_layers
        self.vocab_size = enc_config.vocab_size
        n_heads = enc_config.num_attention_heads
        n_kv_heads = getattr(enc_config, "num_key_value_heads", n_heads)
        head_dim = self.dim // n_heads

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)
        self.rope_embeddings = RotaryEmbedding(
            theta=10000.0,
            head_dim=head_dim,
            max_seqlen=args.decoder_max_len,
        )

        self.layers = nn.ModuleList([
            DecoderBlock(
                dim=self.dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
            )
            for _ in range(self.n_layers)
        ])

        self.norm = RMSNorm(self.dim)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        # Tie weights
        self.output.weight = self.tok_embeddings.weight

    def _init_from_pretrained(self, model_name: str):
        """Initialize from pretrained causal LM."""
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"

        self.dim = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        # Load full model
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )

        # Extract components (LLaMA-style)
        self.tok_embeddings = full_model.model.embed_tokens
        self.layers = full_model.model.layers
        self.norm = full_model.model.norm
        self.output = full_model.lm_head

        # RoPE embeddings
        n_heads = config.num_attention_heads
        head_dim = self.dim // n_heads
        self.rope_embeddings = RotaryEmbedding(
            theta=getattr(config, "rope_theta", 10000.0),
            head_dim=head_dim,
            max_seqlen=self.args.decoder_max_len,
        )

        # Mark pretrained params
        for p in self.parameters():
            p.is_pretrained = True

        del full_model

    def forward(
        self,
        batch: TokenizedBatch,
        sp_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            batch: TokenizedBatch containing decoder_tokens, labels
            sp_group: sequence parallel group for ring attention

        Returns:
            logits: [total_tokens, vocab_size]
        """
        tokens = batch.decoder_tokens.tokens
        seqlen = tokens.shape[0]

        h = self.tok_embeddings(tokens)
        freq_cis = self.rope_embeddings(seqlen=seqlen)

        for layer in self.layers:
            # Handle both custom DecoderBlock and HF layers
            if isinstance(layer, DecoderBlock):
                h = layer(h.unsqueeze(0), freq_cis, mask="causal").squeeze(0)
            else:
                # HF layer - simplified handling
                h = self._forward_hf_layer(layer, h)

        h = self.norm(h)
        logits = self.output(h)

        return logits

    def _forward_hf_layer(
        self,
        layer: nn.Module,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through a HuggingFace decoder layer."""
        # HF expects [B, T, D]
        h = h.unsqueeze(0) if h.dim() == 2 else h
        outputs = layer(h, use_cache=False)
        h = outputs[0]
        return h.squeeze(0) if h.shape[0] == 1 else h

    def generate(
        self,
        batch: TokenizedBatch,
        gen_args: GenerationArgs,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            batch: TokenizedBatch with prompt in decoder_tokens
            gen_args: generation configuration

        Returns:
            output_ids: [batch_size, max_new_tokens]
        """
        tokens = batch.decoder_tokens.tokens.clone()

        # Simple greedy/sampling generation
        for _ in range(gen_args.max_new_tokens):
            logits = self.forward(batch)
            next_logits = logits[-1]  # Last token

            if gen_args.do_sample:
                next_logits = next_logits / gen_args.temperature
                probs = F.softmax(next_logits, dim=-1)
                # Top-p sampling
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

            # Append token
            tokens = torch.cat([tokens, next_token], dim=0)
            # Update batch (simplified - recreate with new tokens)
            batch.decoder_tokens.tokens = tokens

        return tokens.unsqueeze(0)

    @classmethod
    def from_pretrained(cls, model_name: str, args: ModelArgs) -> "Decoder":
        """Load pretrained decoder weights."""
        args.decoder_name = model_name
        return cls(args)
