"""Data configuration shared across train and eval."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataArgs:
    """Data pipeline configuration.

    Used by both training (with packer/mixer) and evaluation (sources + truncation only).
    """

    # Task sources: list of {task: name, weight: float, ...task_kwargs}
    # e.g. [{task: squad, weight: 0.5, query_in_encoder: true}, {task: hotpotqa, weight: 0.3}]
    sources: List[Dict[str, Any]] = field(default_factory=lambda: [{"task": "squad", "weight": 1.0}])

    # Truncation
    doc_max_tokens: int = 8192  # Max tokens per document
    target_max_tokens: int = 2048  # Max tokens per target sequence

    # Pipeline (used by training only)
    target_tokens: int = 65536
    batch_size: int = 32  # Examples per reader fetch
    packer_buffer_size: int = 50
    seed: int = 42
    enc_token_cost: float = 1.0  # Encoder token weight for packing budget
    dec_token_cost: float = 1.0  # Decoder token weight (higher = fewer decoder tokens per batch)
