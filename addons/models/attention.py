"""
Attention modules with packing and ring distribution support.

Public modules:
- SoftmaxAttention: Standard scaled dot-product attention
- LinearAttention: Unified linear attention with configurable FLA variant

Supported FLA variants (when flash-linear-attention is installed):
- linear: Basic linear attention (S = K^T V)
- gla: Gated Linear Attention
- delta_rule: Delta rule linear attention
- gated_delta_rule: Gated delta rule
- kda: Key-driven attention (per-element decay)
- retention: Retentive Network style
- based: Based attention
- simple_gla: Simplified GLA
- mamba2: Mamba-2 SSD
- rwkv6: RWKV-6
- hgrn2: HGRN-2
- abc: Attention with Bounded Context

Uses efficient kernels when available:
- flash_attn for softmax attention
- fla (flash-linear-attention) for linear variants
"""

from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from addons.distributed import ring_send_recv, ring_send, ring_recv


# =============================================================================
# Kernel Registry
# =============================================================================


@dataclass
class FLAKernel:
    """Metadata for an FLA kernel."""
    name: str
    fn: Optional[Callable] = None
    available: bool = False
    has_gate: bool = False  # Whether kernel uses gating (g parameter)
    has_beta: bool = False  # Whether kernel uses write gate (beta parameter)
    gate_per_element: bool = False  # Whether gate is per-element (H*d) vs per-head (H)


# Registry of FLA kernels
FLA_KERNELS: Dict[str, FLAKernel] = {}


def _register_fla_kernel(name: str, import_path: str, fn_name: str, **kwargs):
    """Try to import and register an FLA kernel."""
    kernel = FLAKernel(name=name, **kwargs)
    try:
        module = __import__(import_path, fromlist=[fn_name])
        kernel.fn = getattr(module, fn_name)
        kernel.available = True
    except (ImportError, AttributeError):
        pass
    FLA_KERNELS[name] = kernel


# Register all FLA kernels
_register_fla_kernel("linear", "fla.ops.linear_attn", "chunk_linear_attn")
_register_fla_kernel("gla", "fla.ops.gla", "chunk_gla", has_gate=True)
_register_fla_kernel("delta_rule", "fla.ops.delta_rule", "chunk_delta_rule", has_beta=True)
_register_fla_kernel("gated_delta_rule", "fla.ops.gated_delta_rule", "chunk_gated_delta_rule", has_gate=True, has_beta=True)
_register_fla_kernel("kda", "fla.ops.kda", "chunk_kda", has_gate=True, has_beta=True, gate_per_element=True)
_register_fla_kernel("retention", "fla.ops.retention", "chunk_retention")
_register_fla_kernel("based", "fla.ops.based", "parallel_based")
_register_fla_kernel("simple_gla", "fla.ops.simple_gla", "chunk_simple_gla", has_gate=True)
_register_fla_kernel("mamba2", "fla.ops.gsa", "chunk_gsa", has_gate=True)  # GSA is Mamba-2 SSD
_register_fla_kernel("rwkv6", "fla.ops.rwkv6", "chunk_rwkv6", has_gate=True)
_register_fla_kernel("hgrn2", "fla.ops.hgrn2", "chunk_hgrn2", has_gate=True)
_register_fla_kernel("abc", "fla.ops.abc", "chunk_abc")

# Flash attention for softmax
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def get_available_variants() -> list:
    """Return list of available FLA variants."""
    return ["softmax"] + [k for k, v in FLA_KERNELS.items() if v.available]


# =============================================================================
# Public Modules
# =============================================================================


class SoftmaxAttention(nn.Module):
    """
    Standard scaled dot-product attention.

    Supports packed sequences and ring attention.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or dim // n_heads

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        kv: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_kv: Optional[int] = None,
        sp_group: Optional[dist.ProcessGroup] = None,
        causal: bool = False,
        freq_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [total_tokens, dim] packed queries (and KV if kv is None)
            cu_seqlens: [num_seqs + 1] cumulative sequence lengths
            max_seqlen: max sequence length
            kv: [total_kv, dim] optional separate KV input (for cross-attention)
            cu_seqlens_kv: KV cumulative lengths (required if kv is not None)
            max_seqlen_kv: max KV length
            sp_group: sequence parallel group for ring attention
            causal: apply causal masking
            freq_cis: [total_tokens, ...] RoPE frequencies (applied to Q and K if provided)
        """
        q = self.wq(x).view(-1, self.n_heads, self.head_dim)

        kv_input = kv if kv is not None else x
        k = self.wk(kv_input).view(-1, self.n_kv_heads, self.head_dim)
        v = self.wv(kv_input).view(-1, self.n_kv_heads, self.head_dim)

        if freq_cis is not None:
            from lingua.transformer import apply_rotary_emb
            # [T, H, D] -> [1, T, H, D] for apply_rotary_emb, then squeeze back
            q, k = apply_rotary_emb(q.unsqueeze(0), k.unsqueeze(0), 1, freq_cis)
            q, k = q.squeeze(0), k.squeeze(0)

        cu_kv = cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens
        max_kv = max_seqlen_kv if max_seqlen_kv is not None else max_seqlen

        if sp_group is None:
            out = _packed_softmax_attention(q, k, v, cu_seqlens, cu_kv, max_seqlen, max_kv, causal)
        else:
            out = _packed_ring_softmax(q, k, v, cu_seqlens, cu_kv, sp_group, causal)

        out = out.view(-1, self.n_heads * self.head_dim)
        return self.wo(out)


class LinearAttention(nn.Module):
    """
    Unified linear attention supporting multiple FLA variants.

    Variants:
        - "linear": Basic S = K^T V
        - "gla": Gated Linear Attention
        - "delta_rule": Delta rule
        - "gated_delta_rule": Gated delta rule
        - "kda": Key-driven attention (per-element decay, best for CEPE)
        - "retention": Retentive Network
        - "based": Based attention
        - "simple_gla": Simplified GLA
        - "mamba2": Mamba-2 SSD
        - "rwkv6": RWKV-6
        - "hgrn2": HGRN-2
        - "abc": Attention with Bounded Context
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = False,
        variant: str = "linear",
        gate_bias_init: float = 5.0,
    ):
        super().__init__()
        self.variant = variant
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or dim // n_heads
        self.dim = dim

        # Check variant availability
        if variant not in FLA_KERNELS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(FLA_KERNELS.keys())}")

        kernel = FLA_KERNELS[variant]
        self.kernel = kernel
        if not kernel.available:
            print(f"Warning: FLA kernel '{variant}' not available, using manual fallback")

        # Core projections
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)

        # Gate projections (for variants that need them)
        self.g_proj = None
        self.beta_proj = None

        if kernel.has_gate:
            if kernel.gate_per_element:
                # Per-element gate (kda style): [dim] -> [H * head_dim]
                self.g_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
            else:
                # Per-head gate: [dim] -> [H]
                self.g_proj = nn.Linear(dim, self.n_heads, bias=True)

        if kernel.has_beta:
            # Write gate: [dim] -> [H]
            self.beta_proj = nn.Linear(dim, self.n_heads, bias=True)

        self._init_weights(gate_bias_init)

    def _init_weights(self, gate_bias_init: float):
        """Initialize weights for stable training."""
        # Zero-init output projection (CEPE-style identity start)
        nn.init.zeros_(self.wo.weight)
        if self.wo.bias is not None:
            nn.init.zeros_(self.wo.bias)

        # Gate init for healthy retention at start
        if self.g_proj is not None:
            nn.init.zeros_(self.g_proj.weight)
            nn.init.constant_(self.g_proj.bias, gate_bias_init)

        if self.beta_proj is not None:
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Swish + L2 normalization."""
        x = F.silu(x)
        return F.normalize(x, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        kv: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_kv: Optional[int] = None,
        sp_group: Optional[dist.ProcessGroup] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Same interface as SoftmaxAttention."""
        kv_input = kv if kv is not None else x
        cu_kv = cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens

        # Project Q, K, V
        q = self.wq(x).view(-1, self.n_heads, self.head_dim)
        k = self.wk(kv_input).view(-1, self.n_kv_heads, self.head_dim)
        v = self.wv(kv_input).view(-1, self.n_kv_heads, self.head_dim)

        # Compute gates if needed
        g = None
        beta = None

        if self.g_proj is not None:
            g_raw = self.g_proj(kv_input)
            if self.kernel.gate_per_element:
                g = F.logsigmoid(g_raw).view(-1, self.n_heads, self.head_dim)
            else:
                g = F.logsigmoid(g_raw).view(-1, self.n_heads)

        if self.beta_proj is not None:
            beta = torch.sigmoid(self.beta_proj(kv_input)).view(-1, self.n_heads)

        # Apply feature maps for variants that need them
        if self.variant in ("linear", "delta_rule", "gated_delta_rule", "kda"):
            q = self._feature_map(q)
            k = self._feature_map(k)
            v = F.silu(v)

        # Dispatch to appropriate implementation
        if sp_group is None:
            out = self._forward_packed(q, k, v, g, beta, cu_seqlens, cu_kv, causal)
        else:
            out = self._forward_ring(q, k, v, g, beta, cu_seqlens, cu_kv, sp_group, causal)

        out = out.view(-1, self.n_heads * self.head_dim)
        return self.wo(out)

    def _forward_packed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Forward pass for packed sequences."""
        kernel = self.kernel
        num_seqs = len(cu_seqlens_q) - 1

        # Try FLA kernel first
        if kernel.available and causal:
            return self._forward_fla(q, k, v, g, beta, cu_seqlens_q, cu_seqlens_kv)

        # Manual fallback
        outputs = []
        for i in range(num_seqs):
            q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_s, k_e = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()

            g_seq = g[k_s:k_e] if g is not None else None
            beta_seq = beta[k_s:k_e] if beta is not None else None

            out_i = self._forward_single(
                q[q_s:q_e], k[k_s:k_e], v[k_s:k_e],
                g_seq, beta_seq, causal
            )
            outputs.append(out_i)

        return torch.cat(outputs, dim=0)

    def _forward_fla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Forward using FLA kernel."""
        kernel = self.kernel
        num_seqs = len(cu_seqlens_q) - 1
        scale = self.head_dim ** -0.5
        outputs = []

        for i in range(num_seqs):
            q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_s, k_e = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()

            # Add batch dim: [T, H, D] -> [1, T, H, D]
            q_i = q[q_s:q_e].unsqueeze(0)
            k_i = k[k_s:k_e].unsqueeze(0)
            v_i = v[k_s:k_e].unsqueeze(0)

            # Build kwargs based on kernel requirements
            kwargs: Dict[str, Any] = {"scale": scale}

            if kernel.has_gate and g is not None:
                g_i = g[k_s:k_e].unsqueeze(0)  # [1, T, H] or [1, T, H, D]
                kwargs["g"] = g_i

            if kernel.has_beta and beta is not None:
                beta_i = beta[k_s:k_e].unsqueeze(0)  # [1, T, H]
                kwargs["beta"] = beta_i

            # Call kernel
            out_i = kernel.fn(q_i, k_i, v_i, **kwargs)

            # Handle tuple output (some kernels return (output, state))
            if isinstance(out_i, tuple):
                out_i = out_i[0]

            outputs.append(out_i.squeeze(0))

        return torch.cat(outputs, dim=0)

    def _forward_single(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """Manual fallback for single sequence."""
        seq_q, n_heads, head_dim = q.shape
        seq_kv = k.shape[0]

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        if g is not None:
            g = g.transpose(0, 1)
        if beta is not None:
            beta = beta.transpose(0, 1)

        # Choose implementation based on variant
        if self.variant in ("delta_rule", "gated_delta_rule", "kda"):
            return self._forward_delta_rule(q, k, v, g, beta, causal)
        else:
            return self._forward_linear(q, k, v, g, causal)

    def _forward_linear(
        self,
        q: torch.Tensor,  # [H, seq_q, D]
        k: torch.Tensor,  # [H, seq_kv, D]
        v: torch.Tensor,  # [H, seq_kv, D]
        g: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """Basic linear attention (S = K^T V)."""
        n_heads, seq_q, head_dim = q.shape
        seq_kv = k.shape[1]

        if causal:
            out = torch.zeros_like(q)
            S = torch.zeros(n_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)
            z = torch.zeros(n_heads, head_dim, device=q.device, dtype=q.dtype)

            for t in range(max(seq_q, seq_kv)):
                if t < seq_kv:
                    # Apply decay if gated
                    if g is not None:
                        if g.dim() == 3:  # Per-element: [H, T, D]
                            decay = g[:, t].exp().unsqueeze(-1)  # [H, D, 1]
                            S = S * decay
                        else:  # Per-head: [H, T]
                            decay = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)  # [H, 1, 1]
                            S = S * decay

                    S = S + torch.einsum("hd,he->hde", k[:, t], v[:, t])
                    z = z + k[:, t]

                if t < seq_q:
                    qS = torch.einsum("hd,hde->he", q[:, t], S)
                    qz = torch.einsum("hd,hd->h", q[:, t], z).clamp(min=1e-6)
                    out[:, t] = qS / qz.unsqueeze(-1)
        else:
            S = torch.einsum("hsd,hse->hde", k, v)
            z = k.sum(dim=1)
            qS = torch.einsum("hsd,hde->hse", q, S)
            qz = torch.einsum("hsd,hd->hs", q, z).clamp(min=1e-6)
            out = qS / qz.unsqueeze(-1)

        return out.transpose(0, 1)

    def _forward_delta_rule(
        self,
        q: torch.Tensor,  # [H, seq_q, D]
        k: torch.Tensor,  # [H, seq_kv, D]
        v: torch.Tensor,  # [H, seq_kv, D]
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """Delta rule recurrence."""
        n_heads, seq_q, head_dim = q.shape
        seq_kv = k.shape[1]

        out = torch.zeros_like(q)
        S = torch.zeros(n_heads, head_dim, head_dim, device=q.device, dtype=q.dtype)

        if causal:
            for t in range(max(seq_q, seq_kv)):
                if t < seq_kv:
                    # Decay state
                    if g is not None:
                        if g.dim() == 3:  # Per-element: [H, T, D]
                            decay = g[:, t].exp().unsqueeze(-1)  # [H, D, 1]
                        else:  # Per-head: [H, T]
                            decay = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
                        S = S * decay

                    # Delta update with optional write gate
                    k_t, v_t = k[:, t], v[:, t]
                    delta = torch.einsum("hd,he->hde", k_t, v_t) - torch.einsum("hd,hde->hde", k_t, S)

                    if beta is not None:
                        b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                        S = S + b_t * delta
                    else:
                        S = S + delta

                if t < seq_q:
                    out[:, t] = F.normalize(torch.einsum("hd,hde->he", q[:, t], S), p=2, dim=-1)
        else:
            # Non-causal: process all KV first
            for t in range(seq_kv):
                if g is not None:
                    if g.dim() == 3:
                        decay = g[:, t].exp().unsqueeze(-1)
                    else:
                        decay = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
                    S = S * decay

                k_t, v_t = k[:, t], v[:, t]
                delta = torch.einsum("hd,he->hde", k_t, v_t) - torch.einsum("hd,hde->hde", k_t, S)

                if beta is not None:
                    b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                    S = S + b_t * delta
                else:
                    S = S + delta

            qS = torch.einsum("hsd,hde->hse", q, S)
            out = F.normalize(qS, p=2, dim=-1)

        return out.transpose(0, 1)

    def _forward_ring(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        sp_group: dist.ProcessGroup,
        causal: bool,
    ) -> torch.Tensor:
        """Ring attention for distributed processing."""
        sp_rank = dist.get_rank(sp_group)
        sp_size = dist.get_world_size(sp_group)
        num_seqs = len(cu_seqlens_q) - 1
        n_heads, head_dim = q.shape[1], q.shape[2]
        head_dim_v = v.shape[2]

        # Per-sequence states
        S_list = [torch.zeros(n_heads, head_dim, head_dim_v, device=q.device, dtype=q.dtype)
                  for _ in range(num_seqs)]

        # Receive states from previous rank
        if causal and sp_rank > 0:
            for i in range(num_seqs):
                S_list[i] = ring_recv(S_list[i], sp_group)

        outputs = []

        for i in range(num_seqs):
            q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_s, k_e = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()

            g_seq = g[k_s:k_e] if g is not None else None
            beta_seq = beta[k_s:k_e] if beta is not None else None

            out_i, S_list[i] = self._forward_with_state(
                q[q_s:q_e], k[k_s:k_e], v[k_s:k_e],
                g_seq, beta_seq, S_list[i], causal
            )
            outputs.append(out_i)

        # Send states to next rank
        if causal and sp_rank < sp_size - 1:
            for i in range(num_seqs):
                ring_send(S_list[i], sp_group)

        return torch.cat(outputs, dim=0)

    def _forward_with_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        S_init: torch.Tensor,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with initial state for ring attention."""
        seq_q, n_heads, head_dim = q.shape
        seq_kv = k.shape[0]

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if g is not None:
            g = g.transpose(0, 1)
        if beta is not None:
            beta = beta.transpose(0, 1)

        S = S_init.clone()
        out = torch.zeros_like(q)

        if causal:
            for t in range(max(seq_q, seq_kv)):
                if t < seq_kv:
                    if g is not None:
                        if g.dim() == 3:
                            decay = g[:, t].exp().unsqueeze(-1)
                        else:
                            decay = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
                        S = S * decay

                    k_t, v_t = k[:, t], v[:, t]

                    if self.variant in ("delta_rule", "gated_delta_rule", "kda"):
                        delta = torch.einsum("hd,he->hde", k_t, v_t) - torch.einsum("hd,hde->hde", k_t, S)
                        if beta is not None:
                            b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                            S = S + b_t * delta
                        else:
                            S = S + delta
                    else:
                        S = S + torch.einsum("hd,he->hde", k_t, v_t)

                if t < seq_q:
                    if self.variant in ("delta_rule", "gated_delta_rule", "kda"):
                        out[:, t] = F.normalize(torch.einsum("hd,hde->he", q[:, t], S), p=2, dim=-1)
                    else:
                        qS = torch.einsum("hd,hde->he", q[:, t], S)
                        # For non-delta variants, would need z normalization
                        out[:, t] = qS
        else:
            for t in range(seq_kv):
                if g is not None:
                    if g.dim() == 3:
                        decay = g[:, t].exp().unsqueeze(-1)
                    else:
                        decay = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
                    S = S * decay

                k_t, v_t = k[:, t], v[:, t]

                if self.variant in ("delta_rule", "gated_delta_rule", "kda"):
                    delta = torch.einsum("hd,he->hde", k_t, v_t) - torch.einsum("hd,hde->hde", k_t, S)
                    if beta is not None:
                        b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)
                        S = S + b_t * delta
                    else:
                        S = S + delta
                else:
                    S = S + torch.einsum("hd,he->hde", k_t, v_t)

            qS = torch.einsum("hsd,hde->hse", q, S)
            if self.variant in ("delta_rule", "gated_delta_rule", "kda"):
                out = F.normalize(qS, p=2, dim=-1)
            else:
                out = qS

        return out.transpose(0, 1), S


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================


class LinearKDAAttention(LinearAttention):
    """Alias for LinearAttention with kda variant (backward compatibility)."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bias=bias,
            variant="kda",
        )


# =============================================================================
# Private Helpers - Softmax
# =============================================================================


def _packed_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    causal: bool,
) -> torch.Tensor:
    """Packed softmax attention, uses flash_attn when available."""
    if HAS_FLASH_ATTN:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            causal=causal,
        )
    return _packed_softmax_manual(q, k, v, cu_seqlens_q, cu_seqlens_kv, causal)


def _packed_softmax_manual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Manual fallback for packed softmax attention."""
    num_seqs = len(cu_seqlens_q) - 1
    outputs = []

    for i in range(num_seqs):
        q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_s, k_e = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()

        out_i, _ = _softmax_attn_with_lse(q[q_s:q_e], k[k_s:k_e], v[k_s:k_e], causal)
        outputs.append(out_i)

    return torch.cat(outputs, dim=0)


def _softmax_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-sequence softmax attention returning output and LSE."""
    seq_q, n_heads, head_dim = q.shape
    seq_kv = k.shape[0]

    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)

    if causal:
        mask = torch.triu(torch.ones(seq_q, seq_kv, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(0, 1)

    return out, lse


def _packed_ring_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    sp_group: dist.ProcessGroup,
    causal: bool,
) -> torch.Tensor:
    """Packed ring attention for softmax with LSE accumulation."""
    sp_rank = dist.get_rank(sp_group)
    sp_size = dist.get_world_size(sp_group)
    num_seqs = len(cu_seqlens_q) - 1

    output = torch.zeros_like(q)
    lse = torch.full((q.shape[0], q.shape[1]), float("-inf"), device=q.device, dtype=torch.float32)

    k_block, v_block = k, v
    cu_kv_block = cu_seqlens_kv

    for step in range(sp_size):
        block_rank = (sp_rank - step) % sp_size

        if causal and block_rank > sp_rank:
            if step < sp_size - 1:
                k_block = ring_send_recv(k_block, sp_group)
                v_block = ring_send_recv(v_block, sp_group)
                cu_kv_block = ring_send_recv(cu_kv_block, sp_group)
            continue

        for i in range(num_seqs):
            q_s, q_e = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_s, k_e = cu_kv_block[i].item(), cu_kv_block[i + 1].item()

            if q_e <= q_s or k_e <= k_s:
                continue

            block_causal = causal and (block_rank == sp_rank)
            out_i, lse_i = _softmax_attn_with_lse(q[q_s:q_e], k_block[k_s:k_e], v_block[k_s:k_e], block_causal)

            _accumulate_lse_inplace(output[q_s:q_e], lse[q_s:q_e], out_i, lse_i.transpose(0, 1))

        if step < sp_size - 1:
            k_block = ring_send_recv(k_block, sp_group)
            v_block = ring_send_recv(v_block, sp_group)
            cu_kv_block = ring_send_recv(cu_kv_block, sp_group)

    return output


def _accumulate_lse_inplace(
    acc_out: torch.Tensor,
    acc_lse: torch.Tensor,
    new_out: torch.Tensor,
    new_lse: torch.Tensor,
) -> None:
    """In-place LSE accumulation for ring softmax."""
    max_lse = torch.maximum(acc_lse, new_lse)
    exp_acc = torch.exp(acc_lse - max_lse)
    exp_new = torch.exp(new_lse - max_lse)
    combined_lse = max_lse + torch.log(exp_acc + exp_new)

    acc_w = torch.exp(acc_lse - combined_lse).unsqueeze(-1)
    new_w = torch.exp(new_lse - combined_lse).unsqueeze(-1)

    acc_out.copy_(acc_w * acc_out + new_w * new_out)
    acc_lse.copy_(combined_lse)
