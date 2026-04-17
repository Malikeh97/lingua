"""
Base task interface.

Each task defines:
1. State initialization and reading (functional style)
2. How to evaluate predictions
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from addons.tasks.schema import ContextBasedExample, BatchedContextBasedExamples


class BaseTask(ABC):
    """
    Base class for tasks.

    Reading uses functional state-passing with separate data and state:
    - prepare_data(split, **kwargs) -> data (heavy, not serializable)
    - init_state(split, **kwargs) -> state dict (lightweight, serializable)
    - read(data, state, n) -> (examples, next_state)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name (e.g., 'squad', 'hotpotqa')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        """List of metric names this task computes."""
        raise NotImplementedError

    # ==================== Data Loading (functional) ====================

    @classmethod
    @abstractmethod
    def prepare_data(cls, split: str = "train", **kwargs) -> Any:
        """
        Load/prepare the dataset. May be heavy (download, memory).

        The returned object is opaque — the task decides its type.
        Not serializable; re-created on restore.

        Args:
            split: Data split
            **kwargs: Task-specific options

        Returns:
            Data object (HF Dataset, iterator wrapper, etc.)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        """
        Initialize lightweight, serializable reading state.

        Must be JSON-serializable for checkpointing.
        Examples: {"idx": 0, "epoch": 0, "seed": 42}

        Args:
            split: Data split
            **kwargs: Task-specific options

        Returns:
            State dict (must be serializable)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def read(cls, data: Any, state: Dict[str, Any], batch_size: int) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]],
        Dict[str, Any],
    ]:
        """
        Read examples from data using state, return next state.

        Pure function: (data, state, n) -> (examples, next_state)

        Args:
            data: Data object from prepare_data()
            state: Current reading state
            batch_size: Number of examples to read

        Returns:
            (examples, next_state) tuple
        """
        raise NotImplementedError

    @classmethod
    def is_exhausted(cls, state: Dict[str, Any]) -> bool:
        """Check if reading is done (for finite datasets)."""
        return state.get("exhausted", False)

    # ==================== Evaluation ====================

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[dict[str, any]],
    ) -> Dict[str, float]:
        """
        Evaluate predictions using task-specific metrics.

        Args:
            predictions: Model predictions (one per example)
            example_miscs: Misc dict per example (contains ground truth, metadata)

        Returns:
            Dict of metric names to values
        """
        raise NotImplementedError
