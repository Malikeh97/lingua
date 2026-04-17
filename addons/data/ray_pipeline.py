"""
Ray-based data pipeline.

Components:
- DatasetReader: Actor per source, holds task state
- Mixer: Samples from readers by weight
- Packer: Bin-packs into batches

Architecture:
    DatasetReaders ──► Mixer Actor ──► Packer Actor ──► Training
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import random

import ray

from addons.tasks.schema import ContextBasedExample, BatchedContextBasedExamples
from addons.tasks.base import BaseTask
from addons.data.collate import TokenizedBatch


# ==================== Dataset Reader ====================


@ray.remote
class DatasetReader:
    """
    Stateful reader for a single data source.

    Reads raw examples, tokenizes them, returns tokenized ContextBasedExamples.
    """

    def __init__(
        self,
        task_cls: Type[BaseTask],
        split: str,
        tokenizer_factory,
        doc_max_tokens: int,
        target_max_tokens: int,
        **kwargs,
    ):
        """
        Args:
            task_cls: Task class (not instance)
            split: Data split
            tokenizer_factory: Callable that returns (enc_tokenizer, dec_tokenizer)
            doc_max_tokens: Max tokens per document
            target_max_tokens: Max tokens per decoder sequence
            **kwargs: Passed to task's prepare_data and init_state
        """
        self.task_cls = task_cls
        self.split = split
        self.kwargs = kwargs
        self.doc_max_tokens = doc_max_tokens
        self.target_max_tokens = target_max_tokens

        # Load tokenizers locally
        self.enc_tok, self.dec_tok = tokenizer_factory()

        # Load data and init state
        self.data = task_cls.prepare_data(split, **kwargs)
        self.state = task_cls.init_state(split, **kwargs)

    def get_batch(self, n: int) -> List[ContextBasedExample]:
        """Get next n units, tokenize, return tokenized examples."""
        old_epoch = self.state.get("epoch", 0)
        units, self.state = self.task_cls.read(self.data, self.state, n)

        # Re-prepare data on epoch transition (e.g., re-shuffle)
        new_epoch = self.state.get("epoch", 0)
        if new_epoch > old_epoch:
            self.data = self.task_cls.prepare_data(
                self.split, **{**self.kwargs, "seed": self.kwargs.get("seed", 42) + new_epoch}
            )

        # Flatten and tokenize
        results = []
        for unit in units:
            if isinstance(unit, ContextBasedExample):
                results.append(unit.tokenize(
                    self.enc_tok, self.dec_tok,
                    self.doc_max_tokens, self.target_max_tokens,
                ))
            else:  # BatchedContextBasedExamples
                for ex in unit:
                    results.append(ex.tokenize(
                        self.enc_tok, self.dec_tok,
                        self.doc_max_tokens, self.target_max_tokens,
                    ))
        return results

    def is_exhausted(self) -> bool:
        """Check if this source is done."""
        return self.task_cls.is_exhausted(self.state)

    def state_dict(self) -> Dict[str, Any]:
        """For checkpointing. State is already serializable."""
        return self.state

    def load_state_dict(self, state: Dict[str, Any]):
        """Restore from checkpoint. Data already loaded in __init__."""
        self.state = state


# ==================== Mixer ====================


@ray.remote
class Mixer:
    """
    Mixes examples from multiple readers according to weights.
    """

    def __init__(self, source_names: List[str], weights: List[float], seed: int = 42):
        """
        Args:
            source_names: Names of data sources
            weights: Mixing weights (will be normalized)
            seed: Random seed for reproducibility
        """
        self.source_names = source_names
        self.weights = [w / sum(weights) for w in weights]
        self.rng = random.Random(seed)
        self.counts = {name: 0 for name in source_names}

    def sample_source(self) -> str:
        """Sample which source to pull from next."""
        return self.rng.choices(self.source_names, self.weights)[0]

    def record_sample(self, source: str):
        """Record that we sampled from this source."""
        self.counts[source] += 1

    def get_counts(self) -> Dict[str, int]:
        return self.counts.copy()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "counts": self.counts.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.rng.setstate(state["rng_state"])
        self.counts = state["counts"]


# ==================== Packer ====================


@ray.remote
class Packer:
    """
    Packs tokenized examples into fixed-token batches.

    Receives tokenized ContextBasedExamples with exact token counts.
    Emits TokenizedBatch (on CPU — moved to GPU by DataIterator).
    Uses best-fit decreasing: always pick the largest unit that fits.
    """

    def __init__(self, target_tokens: int, buffer_size: int = 50):
        """
        Args:
            target_tokens: Target tokens per batch
            buffer_size: Number of units to buffer for bin-packing
        """
        self.target_tokens = target_tokens
        self.buffer_size = buffer_size

        # (example, token_cost)
        self.buffer: List[Tuple[ContextBasedExample, int]] = []
        self.current_batch: List[ContextBasedExample] = []
        self.current_tokens = 0
        self.batches_emitted = 0

    def can_fit(self, token_count: int) -> bool:
        """Check if a unit with given token count can fit in current batch."""
        return self.current_tokens + token_count <= self.target_tokens

    def add(self, example: ContextBasedExample, token_cost: int):
        """Add tokenized example to buffer."""
        self.buffer.append((example, token_cost))

    def try_pack(self) -> Optional[TokenizedBatch]:
        """
        Try to pack a batch from buffer using best-fit decreasing.

        Algorithm:
        1. Find all units that fit in remaining space
        2. Pick the largest one (best-fit)
        3. Emit batch if ≥95% full or nothing fits

        Returns:
            TokenizedBatch if ready, None otherwise
        """
        gap = self.target_tokens - self.current_tokens

        candidates = [(u, tc) for u, tc in self.buffer if tc <= gap]

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            self.buffer.remove(best)
            self.current_batch.append(best[0])
            self.current_tokens += best[1]

            if self.current_tokens >= self.target_tokens * 0.95:
                return self._emit_batch()
        elif self.current_batch:
            return self._emit_batch()
        elif self.buffer:
            # Oversized unit — emit alone to avoid deadlock
            oversized_ex, oversized_tc = self.buffer.pop(0)
            print(f"[Packer WARNING] Oversized unit: {oversized_tc} tokens > target {self.target_tokens}")
            self.current_batch.append(oversized_ex)
            self.current_tokens = oversized_tc
            return self._emit_batch()

        return None

    def pack_until_batch(self) -> Optional[TokenizedBatch]:
        """
        Keep packing from buffer until a batch is ready.

        Returns:
            TokenizedBatch when ready, None if buffer exhausted
        """
        while self.buffer:
            batch = self.try_pack()
            if batch is not None:
                return batch
        return None

    def _emit_batch(self) -> TokenizedBatch:
        """Emit current batch as TokenizedBatch (on CPU), reset state."""
        batch = TokenizedBatch.from_tokenized_examples(self.current_batch)
        self.current_batch = []
        self.current_tokens = 0
        self.batches_emitted += 1
        return batch

    def flush(self) -> Optional[TokenizedBatch]:
        """Flush remaining units as final batch."""
        if self.current_batch:
            return self._emit_batch()
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "buffer": self.buffer.copy(),
            "current_batch": self.current_batch.copy(),
            "current_tokens": self.current_tokens,
            "batches_emitted": self.batches_emitted,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        self.buffer = state["buffer"]
        self.current_batch = state["current_batch"]
        self.current_tokens = state["current_tokens"]
        self.batches_emitted = state["batches_emitted"]


# ==================== Pipeline ====================


@dataclass
class PipelineConfig:
    """Configuration for Ray data pipeline."""

    target_tokens: int = 1_000_000
    batch_size: int = 64  # examples per reader fetch
    packer_buffer_size: int = 50
    seed: int = 42
    enc_token_cost: float = 1.0
    dec_token_cost: float = 1.0


@dataclass
class PipelineState:
    """Full pipeline state for checkpointing."""

    readers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mixer: Dict[str, Any] = field(default_factory=dict)
    packer: Dict[str, Any] = field(default_factory=dict)


@ray.remote
class PipelineCoordinator:
    """
    Coordinates the data pipeline.

    Owns readers, mixer, packer. Handles checkpointing.
    """

    def __init__(
        self,
        task_classes: Dict[str, Type[BaseTask]],
        weights: Dict[str, float],
        config: PipelineConfig,
        split: str = "train",
        tokenizer_factory=None,
        doc_max_tokens: int = 8192,
        target_max_tokens: int = 2048,
        task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Args:
            task_classes: {name: TaskClass}
            weights: {name: weight}
            config: Pipeline config
            split: Data split
            tokenizer_factory: Callable returning (enc_tok, dec_tok)
            doc_max_tokens: Max tokens per document
            target_max_tokens: Max tokens per decoder sequence
            task_kwargs: {name: kwargs} for task init_state
        """
        self.config = config
        self.split = split
        self.source_names = list(task_classes.keys())
        task_kwargs = task_kwargs or {}

        # Create reader actors (each loads its own tokenizer via factory)
        self.readers: Dict[str, ray.actor.ActorHandle] = {}
        for name, task_cls in task_classes.items():
            kwargs = task_kwargs.get(name, {})
            self.readers[name] = DatasetReader.remote(
                task_cls, split, tokenizer_factory,
                doc_max_tokens, target_max_tokens, **kwargs,
            )

        # Create mixer actor
        weight_list = [weights[name] for name in self.source_names]
        self.mixer = Mixer.remote(self.source_names, weight_list, config.seed)

        # Create packer actor
        self.packer = Packer.remote(config.target_tokens, config.packer_buffer_size)

        # Track exhausted sources
        self.exhausted_sources: set = set()

    def get_batch(self) -> Optional[TokenizedBatch]:
        """
        Get next batch (blocking).

        Returns:
            TokenizedBatch (on CPU), or None if all sources exhausted
        """
        while True:
            batch = ray.get(self.packer.pack_until_batch.remote())
            if batch is not None:
                return batch

            if len(self.exhausted_sources) >= len(self.source_names):
                return ray.get(self.packer.flush.remote())

            source = ray.get(self.mixer.sample_source.remote())

            if source in self.exhausted_sources:
                continue

            if ray.get(self.readers[source].is_exhausted.remote()):
                self.exhausted_sources.add(source)
                continue

            # Reader returns tokenized examples with exact token counts
            examples = ray.get(
                self.readers[source].get_batch.remote(self.config.batch_size)
            )

            if not examples:
                self.exhausted_sources.add(source)
                continue

            ray.get(self.mixer.record_sample.remote(source))
            for ex in examples:
                token_cost = ex.token_cost(
                    enc_cost=self.config.enc_token_cost,
                    dec_cost=self.config.dec_token_cost,
                )
                ray.get(self.packer.add.remote(ex, token_cost))

    def get_batch_async(self) -> ray.ObjectRef:
        """Get next batch (non-blocking, returns ObjectRef)."""
        return self._get_batch_internal.remote(self)

    @ray.method(num_returns=1)
    def _get_batch_internal(self) -> Optional[TokenizedBatch]:
        """Internal method for async batch retrieval."""
        return self.get_batch()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        mixer_counts = ray.get(self.mixer.get_counts.remote())
        packer_state = ray.get(self.packer.state_dict.remote())
        return {
            "source_counts": mixer_counts,
            "batches_emitted": packer_state["batches_emitted"],
            "buffer_size": len(packer_state["buffer"]),
            "current_batch_tokens": packer_state["current_tokens"],
            "exhausted_sources": list(self.exhausted_sources),
        }

    def checkpoint(self) -> PipelineState:
        """Collect state from all actors."""
        # Gather reader states
        reader_states = {}
        for name, reader in self.readers.items():
            reader_states[name] = ray.get(reader.state_dict.remote())

        # Gather mixer and packer states
        mixer_state = ray.get(self.mixer.state_dict.remote())
        packer_state = ray.get(self.packer.state_dict.remote())

        return PipelineState(
            readers=reader_states,
            mixer=mixer_state,
            packer=packer_state,
        )

    def restore(self, state: PipelineState):
        """Restore all actors from state."""
        # Restore reader states
        restore_futures = []
        for name, reader_state in state.readers.items():
            if name in self.readers:
                restore_futures.append(
                    self.readers[name].load_state_dict.remote(reader_state)
                )

        # Restore mixer and packer
        restore_futures.append(self.mixer.load_state_dict.remote(state.mixer))
        restore_futures.append(self.packer.load_state_dict.remote(state.packer))

        # Wait for all restores
        ray.get(restore_futures)

        # Rebuild exhausted sources set from reader states
        self.exhausted_sources = set()
        for name, reader in self.readers.items():
            if ray.get(reader.is_exhausted.remote()):
                self.exhausted_sources.add(name)


# ==================== Convenience Functions ====================


def create_pipeline(
    task_configs: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    config: Optional[PipelineConfig] = None,
    split: str = "train",
    tokenizer_factory=None,
    doc_max_tokens: int = 8192,
    target_max_tokens: int = 2048,
) -> ray.actor.ActorHandle:
    """
    Create a data pipeline from task configurations.

    Args:
        task_configs: {name: {"task_class": TaskClass, **kwargs}}
        weights: {name: weight}
        config: Pipeline config (default: PipelineConfig())
        split: Data split
        tokenizer_factory: Callable returning (enc_tok, dec_tok)
        doc_max_tokens: Max tokens per document
        target_max_tokens: Max tokens per decoder sequence

    Returns:
        PipelineCoordinator actor handle
    """
    from addons.tasks.registry import get_task

    if config is None:
        config = PipelineConfig()

    # Extract task classes and kwargs
    task_classes = {}
    task_kwargs = {}
    for name, cfg in task_configs.items():
        if "task_class" in cfg:
            task_classes[name] = cfg["task_class"]
        elif "task_name" in cfg:
            task_classes[name] = type(get_task(cfg["task_name"]))
        else:
            raise ValueError(f"Task config for {name} must have 'task_class' or 'task_name'")

        task_kwargs[name] = {k: v for k, v in cfg.items() if k not in ("task_class", "task_name")}

    return PipelineCoordinator.remote(
        task_classes=task_classes,
        weights=weights,
        config=config,
        split=split,
        tokenizer_factory=tokenizer_factory,
        doc_max_tokens=doc_max_tokens,
        target_max_tokens=target_max_tokens,
        task_kwargs=task_kwargs,
    )


def create_pipeline_from_names(
    task_names: List[str],
    weights: List[float],
    config: Optional[PipelineConfig] = None,
    split: str = "train",
    **kwargs,
) -> ray.actor.ActorHandle:
    """
    Create a pipeline from registered task names.

    Args:
        task_names: List of registered task names (e.g., ["squad", "hotpotqa"])
        weights: Mixing weights (same order as task_names)
        config: Pipeline config
        split: Data split
        **kwargs: Passed to all tasks' init_state

    Returns:
        PipelineCoordinator actor handle
    """
    task_configs = {}
    weight_dict = {}
    for name, weight in zip(task_names, weights):
        task_configs[name] = {"task_name": name, **kwargs}
        weight_dict[name] = weight

    return create_pipeline(task_configs, weight_dict, config, split)
