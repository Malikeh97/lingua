"""
Encoder-decoder transformer model.
"""

import os
from typing import Optional

_DEBUG = os.environ.get("DEBUG") == "1"

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
from .utils import make_packed_causal_mask, gather_encoder_states
from addons.data.collate import TokenizedBatch


try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """ModernBERT-style RoPE: pairs (x[k], x[D/2+k]) rather than (x[2k], x[2k+1])."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _apply_rotary_packed(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply halves-style rotary embedding to packed Q/K.

    Q, K shape: [total_tokens, n_heads, head_dim]
    cos, sin shape: [1, total_tokens, head_dim] (from HF ModernBertRotaryEmbedding)
    """
    if cos.dim() == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = cos.unsqueeze(1)  # [T, 1, D] broadcasts over n_heads
    sin = sin.unsqueeze(1)
    q_out = (q.float() * cos.float() + _rotate_half(q).float() * sin.float()).to(q.dtype)
    k_out = (k.float() * cos.float() + _rotate_half(k).float() * sin.float()).to(k.dtype)
    return q_out, k_out


def _patch_packed_attention(attn: nn.Module):
    """Replace a ModernBertAttention's forward with a packed flash-attn varlen path.

    The replacement reads `cu_seqlens` / `max_seqlen` set on the module before each
    encoder forward pass. Keeps the original Wqkv / Wo parameters untouched so the
    pretrained weights work as-is.
    """
    if not _HAS_FLASH_ATTN:
        raise RuntimeError("flash_attn is required for packed encoder attention")

    config = attn.config
    n_heads = config.num_attention_heads
    head_dim = attn.head_dim
    hidden_size = config.hidden_size
    # TODO: sliding-window support. For now use full attention everywhere — fine
    # when max_seqlen <= config.local_attention/2, otherwise matches differ.
    window_size = (-1, -1)

    def packed_forward(hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        # hidden_states: [total_tokens, hidden_size]
        qkv = attn.Wqkv(hidden_states)
        qkv = qkv.view(-1, 3, n_heads, head_dim)
        q, k, v = qkv.unbind(dim=1)  # each: [T, H, D]

        cos, sin = position_embeddings
        q, k = _apply_rotary_packed(q, k, cos, sin)

        out = flash_attn_varlen_func(
            q, k, v,
            attn._cu_seqlens, attn._cu_seqlens,
            attn._max_seqlen, attn._max_seqlen,
            causal=False,
            window_size=window_size,
        )
        out = out.reshape(-1, hidden_size).contiguous()
        return attn.out_drop(attn.Wo(out)), None

    attn.forward = packed_forward


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
        enc_cu_seqlens: torch.Tensor,
        example_doc_indices: list,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """
        Args:
            x: decoder hidden states [total_dec, dim]
            encoder_output: deduplicated encoder states [total_unique_enc, dim]
            enc_cu_seqlens: cumulative lengths of unique docs
            example_doc_indices: per-example doc index mapping
            cu_seqlens_q: cumulative decoder lengths
            max_seqlen_q: max decoder length
        """
        # Gather per-example encoder states (recomputed under grad checkpointing)
        encoder_final_states, cu_seqlens_kv, max_seqlen_kv = gather_encoder_states(
            encoder_output, enc_cu_seqlens, example_doc_indices,
        )

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(
            x,
            cu_seqlens_q,
            max_seqlen_q,
            kv=encoder_final_states,
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
        enc_cu_seqlens: torch.Tensor,
        example_doc_indices: list,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Self-attention (use block-diagonal mask if provided, else causal)
        mask = self_attn_mask if self_attn_mask is not None else "causal"
        h = x + self.attention(self.attention_norm(x), freq_cis, mask=mask)

        # Cross-attention (gather happens inside, friendly to grad checkpointing)
        h = self.cross_attn_adapter(
            h, encoder_output,
            enc_cu_seqlens, example_doc_indices,
            cu_seqlens_q, max_seqlen_q,
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
        self._debug_cross_attn_printed = False
        self._init_encoder(args)
        self._init_decoder(args)

    def _init_encoder(self, args: ModelArgs):
        """Initialize encoder by reusing HF ModernBERT components with packed attention.

        Keeps HF's embeddings, encoder layers, final_norm, and rotary embedding
        module unchanged, then patches each layer's self-attention to use
        flash_attn_varlen_func with cu_seqlens. This avoids having to port RoPE
        conventions, activations, LayerNorm bias, per-layer-type rope_theta, etc.
        """
        config = AutoConfig.from_pretrained(args.encoder_name)
        self.encoder_dim = config.hidden_size
        self.encoder_vocab_size = config.vocab_size

        # Load HF ModernBERT and adopt its components.
        hf_model = AutoModel.from_pretrained(args.encoder_name, torch_dtype=torch.bfloat16)
        self.encoder_embeddings = hf_model.embeddings
        self.encoder_layers = hf_model.layers
        self.encoder_final_norm = hf_model.final_norm
        self.encoder_rotary_emb = hf_model.rotary_emb
        self.encoder_layer_types = list(config.layer_types)

        # Replace each attention's forward with a packed flash-attn path.
        for layer in self.encoder_layers:
            _patch_packed_attention(layer.attn)

        # Freeze encoder if requested
        if args.freeze_encoder:
            for name, param in self.named_parameters():
                if name.startswith("encoder_"):
                    param.requires_grad = False

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
            max_seqlen=args.decoder_max_position,
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
            max_seqlen=args.decoder_max_position,
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
        return self.encoder_embeddings.tok_embeddings.weight.data.float()

    def encode(
        self,
        batch: TokenizedBatch,
        sp_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """
        Encode input documents using packed cu_seqlens attention.

        No padding — tokens go directly through the transformer.

        Args:
            batch: TokenizedBatch containing encoder_tokens
            sp_group: sequence parallel group

        Returns:
            encoder_hidden: [total_enc_tokens, hidden_dim]
        """
        packed = batch.encoder_tokens
        device = packed.tokens.device

        if packed.num_seqs == 0:
            return torch.zeros(0, self.encoder_dim, device=device,
                               dtype=self.encoder_embeddings.tok_embeddings.weight.dtype)

        # Embed: HF ModernBertEmbeddings applies dropout + LayerNorm to token embeddings.
        h = self.encoder_embeddings(packed.tokens.unsqueeze(0)).squeeze(0)  # [T, D]

        # Build position_ids: each doc restarts from 0.
        cu = packed.cu_seqlens
        T = packed.tokens.shape[0]
        position_ids = torch.zeros(T, dtype=torch.long, device=device)
        for i in range(packed.num_seqs):
            s = cu[i].item()
            e = cu[i + 1].item()
            position_ids[s:e] = torch.arange(e - s, device=device)
        position_ids_2d = position_ids.unsqueeze(0)  # [1, T] for HF rotary_emb

        # Precompute RoPE per layer type (ModernBERT uses different theta for full vs sliding).
        unique_types = set(self.encoder_layer_types)
        rope_by_type = {
            lt: self.encoder_rotary_emb(h, position_ids_2d, layer_type=lt)
            for lt in unique_types
        }

        # Forward through encoder layers with packed attention.
        for i, layer in enumerate(self.encoder_layers):
            layer.attn._cu_seqlens = cu
            layer.attn._max_seqlen = packed.max_seqlen
            pos_emb = rope_by_type[self.encoder_layer_types[i]]
            h = layer(h, position_embeddings=pos_emb)

        h = self.encoder_final_norm(h)

        # Project to decoder dim if needed
        if hasattr(self, "encoder_projection") and self.encoder_projection is not None:
            h = self.encoder_projection(h)

        return h

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
        if _DEBUG:
            enc = batch.encoder_tokens
            dec = batch.decoder_tokens
            print(f"[forward] batch: {dec.num_seqs} examples, "
                  f"enc={enc.tokens.numel()} tokens ({enc.num_seqs} docs, max={enc.max_seqlen}), "
                  f"dec={dec.tokens.numel()} tokens (max={dec.max_seqlen})")

        # Encode
        encoder_output = self.encode(batch, sp_group)
        encoder_output = encoder_output.to(self.decoder_tok_embeddings.weight.dtype)

        # Decode
        decoder_tokens = batch.decoder_tokens.tokens
        seqlen = decoder_tokens.shape[0]

        h = self.decoder_tok_embeddings(decoder_tokens)

        # Build per-token position indices for packed sequences
        # Each sub-sequence restarts positions from 0
        cu = batch.decoder_tokens.cu_seqlens
        tok_idx = torch.zeros(seqlen, dtype=torch.long, device=decoder_tokens.device)
        for i in range(len(batch.decoder_tokens.lengths)):
            start = cu[i].item()
            end = cu[i + 1].item()
            tok_idx[start:end] = torch.arange(end - start, device=decoder_tokens.device)
        freq_cis = self.decoder_rope(tok_idx=tok_idx)

        # Get cu_seqlens for packed attention
        cu_seqlens_q = batch.decoder_tokens.cu_seqlens
        max_seqlen_q = batch.decoder_tokens.max_seqlen

        # Block-diagonal causal mask: isolate self-attention per example
        self_attn_mask = make_packed_causal_mask(
            cu_seqlens_q, seqlen, decoder_tokens.device, dtype=h.dtype,
        )

        # Deduplicated encoder states + mapping for cross-attention
        # Gather happens inside each layer (friendly to gradient checkpointing)
        enc_cu_seqlens = batch.encoder_tokens.cu_seqlens
        example_doc_indices = batch.example_doc_indices

        if _DEBUG and not self._debug_cross_attn_printed:
            self._debug_cross_attn_printed = True
            # Gather once just for debug printing
            _, dbg_cu_kv, _ = gather_encoder_states(
                encoder_output, enc_cu_seqlens, example_doc_indices,
            )
            dec_cu = cu_seqlens_q
            print("[cross-attn DEBUG] First forward pass mapping:")
            print(f"  decoder: {len(batch.decoder_tokens.lengths)} seqs, cu_seqlens_q={dec_cu.tolist()}")
            print(f"  encoder states: {len(example_doc_indices)} chunks, cu_seqlens_kv={dbg_cu_kv.tolist()}")
            for i in range(min(2, len(batch.decoder_tokens.lengths))):
                dq_s, dq_e = dec_cu[i].item(), dec_cu[i + 1].item()
                kv_s, kv_e = dbg_cu_kv[i].item(), dbg_cu_kv[i + 1].item()
                print(f"  example[{i}]: dec_tokens[{dq_s}:{dq_e}] ({dq_e - dq_s} tokens) "
                      f"attends to enc_states[{kv_s}:{kv_e}] ({kv_e - kv_s} tokens) "
                      f"from docs {example_doc_indices[i]}")

        if hasattr(self, "decoder_layers"):
            # From-scratch decoder with integrated cross-attention
            for layer in self.decoder_layers:
                h = layer(
                    h.unsqueeze(0) if h.dim() == 2 else h,
                    encoder_output,
                    freq_cis,
                    enc_cu_seqlens,
                    example_doc_indices,
                    cu_seqlens_q,
                    max_seqlen_q,
                    self_attn_mask=self_attn_mask,
                )
                if h.dim() == 3:
                    h = h.squeeze(0)
        else:
            # Pretrained decoder with injected cross-attention adapters
            for layer, cross_attn in zip(self.pretrained_decoder_layers, self.cross_attn_adapters):
                # Self-attention through HF layer
                h = h.unsqueeze(0) if h.dim() == 2 else h
                outputs = layer(h, attention_mask=self_attn_mask, use_cache=False)
                h = outputs[0]
                if h.dim() == 3:
                    h = h.squeeze(0)

                # Cross-attention adapter (gather inside)
                h = cross_attn(
                    h, encoder_output,
                    enc_cu_seqlens, example_doc_indices,
                    cu_seqlens_q, max_seqlen_q,
                )

        h = self.decoder_norm(h)
        logits = self.decoder_output(h)

        return logits

    def generate(
        self,
        batch: TokenizedBatch,
        gen_args: GenerationArgs,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens given encoded documents.

        Single-example only. Stops at eos_token_id if provided.

        Args:
            batch: TokenizedBatch with a single example
            gen_args: generation configuration
            eos_token_id: stop generation when this token is produced

        Returns:
            output_ids: [1, seq_len] generated token ids
        """
        tokens = batch.decoder_tokens.tokens.clone()
        prompt_len = tokens.shape[0]

        for _ in range(gen_args.max_new_tokens):
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

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            tokens = torch.cat([tokens, next_token], dim=0)
            batch.decoder_tokens.tokens = tokens
            batch.decoder_tokens.lengths[-1] += 1
            batch.decoder_tokens.cu_seqlens[-1] = tokens.shape[0]

        # Return only generated tokens (strip prompt)
        return tokens[prompt_len:].unsqueeze(0)

    @staticmethod
    def tokenizer_factory(args: ModelArgs):
        """Return a callable that creates (encoder_tok, decoder_tok).

        Serializable by Ray (closure over strings). Each worker calls it
        once to load tokenizers locally.
        """
        encoder_name = args.encoder_name
        decoder_name = args.decoder_name

        def factory():
            from transformers import AutoTokenizer
            enc = AutoTokenizer.from_pretrained(encoder_name)
            dec = AutoTokenizer.from_pretrained(decoder_name) if decoder_name else enc
            return enc, dec

        return factory

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


def _make_ckpt_forward(module: nn.Module):
    """Wrap a module's forward with torch checkpoint."""
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    original_forward = module.forward

    def ckpt_forward(*args, **kwargs):
        return torch_checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module.forward = ckpt_forward


def apply_activation_checkpointing(model: nn.Module):
    """Apply activation checkpointing to decoder layers.

    Works for both from-scratch (EncoderDecoderBlock) and pretrained
    (HF layers + CrossAttentionAdapter) decoder paths.

    With the gather-inside-layer design, checkpointing means
    the per-example encoder states are recomputed per layer during
    backward rather than persisting — only the deduplicated
    encoder output is kept.
    """
    if hasattr(model, "decoder_layers"):
        for layer in model.decoder_layers:
            _make_ckpt_forward(layer)
    elif hasattr(model, "pretrained_decoder_layers"):
        for layer, cross_attn in zip(
            model.pretrained_decoder_layers, model.cross_attn_adapters
        ):
            _make_ckpt_forward(layer)
            _make_ckpt_forward(cross_attn)
