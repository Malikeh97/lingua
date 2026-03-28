"""FineInstructions Nemotron task implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


# Common separators used in instruction-response formats
RESPONSE_SEPARATORS = [
    "\n\nAnswer:",
    "\n\nResponse:",
    "\n\nA:",
    "\nAnswer:",
    "\nResponse:",
    "\n\n",
]


def parse_instruction_response(
    text: str, separator: Optional[str] = None
) -> Tuple[str, str]:
    """
    Parse instruction and response from text.

    Args:
        text: Raw text containing instruction and response
        separator: Custom separator (auto-detect if None)

    Returns:
        Tuple of (instruction, response)
    """
    if separator:
        # Use user-specified separator
        if separator in text:
            parts = text.split(separator, 1)
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        return text.strip(), ""

    # Auto-detect separator
    for sep in RESPONSE_SEPARATORS:
        if sep in text:
            parts = text.split(sep, 1)
            instruction = parts[0].strip()
            response = parts[1].strip() if len(parts) > 1 else ""
            # Only use this split if both parts are non-empty
            if instruction and response:
                return instruction, response

    # Fallback: treat entire text as instruction with empty response
    return text.strip(), ""


@register_task("fineinstructions")
class FineInstructionsTask(BaseTask):
    """
    FineInstructions Nemotron task.

    ~1.2B row synthetic instruction-answer dataset from CommonCrawl.
    """

    @property
    def name(self) -> str:
        return "fineinstructions"

    @property
    def metrics(self) -> List[str]:
        return ["avg_prediction_length", "avg_reference_length", "num_examples"]

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        """
        Initialize reading state.

        Args:
            split: Data split
            **kwargs:
                response_separator: Custom separator for parsing
                streaming: Use streaming mode (default True for large dataset)
                max_examples: Optional cap on examples
                seed: Random seed
        """
        streaming = kwargs.get("streaming", True)
        response_separator = kwargs.get("response_separator")

        dataset = load_dataset(
            "fineinstructions/fineinstructions_nemotron",
            split=split,
            streaming=streaming,
        )

        if streaming:
            # For streaming, we use an iterator
            iterator = iter(dataset)
            return {
                "iterator": iterator,
                "buffer": [],
                "response_separator": response_separator,
                "exhausted": False,
                "total_read": 0,
                "max_examples": kwargs.get("max_examples"),
            }
        else:
            # For non-streaming, shuffle and limit if needed
            shuffle = kwargs.get("shuffle", split == "train")
            seed = kwargs.get("seed", 42)
            if shuffle:
                dataset = dataset.shuffle(seed=seed)

            max_examples = kwargs.get("max_examples")
            if max_examples is not None:
                dataset = dataset.select(range(min(max_examples, len(dataset))))

            return {
                "dataset": dataset,
                "idx": 0,
                "response_separator": response_separator,
                "exhausted": False,
            }

    @classmethod
    def read(
        cls, state: Dict[str, Any], batch_size: int
    ) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]], Dict[str, Any]
    ]:
        """Read batch_size examples."""
        if "iterator" in state:
            return cls._read_streaming(state, batch_size)
        else:
            return cls._read_indexed(state, batch_size)

    @classmethod
    def _read_streaming(
        cls, state: Dict[str, Any], batch_size: int
    ) -> Tuple[List[ContextBasedExample], Dict[str, Any]]:
        """Read from streaming iterator."""
        iterator = state["iterator"]
        separator = state.get("response_separator")
        max_examples = state.get("max_examples")
        total_read = state.get("total_read", 0)
        examples = []

        for _ in range(batch_size):
            if max_examples is not None and total_read >= max_examples:
                break
            try:
                raw = next(iterator)
                ex = cls._map_example(raw, separator)
                if ex is not None:  # Skip null entries
                    examples.append(ex)
                    total_read += 1
            except StopIteration:
                new_state = {**state, "exhausted": True, "total_read": total_read}
                return examples, new_state

        new_state = {**state, "total_read": total_read}
        return examples, new_state

    @classmethod
    def _read_indexed(
        cls, state: Dict[str, Any], batch_size: int
    ) -> Tuple[List[ContextBasedExample], Dict[str, Any]]:
        """Read from indexed dataset."""
        dataset = state["dataset"]
        idx = state["idx"]
        separator = state.get("response_separator")
        examples = []

        end_idx = min(idx + batch_size, len(dataset))
        for i in range(idx, end_idx):
            raw = dataset[i]
            ex = cls._map_example(raw, separator)
            if ex is not None:  # Skip null entries
                examples.append(ex)

        new_state = {
            **state,
            "idx": end_idx,
            "exhausted": end_idx >= len(dataset),
        }
        return examples, new_state

    @classmethod
    def _map_example(
        cls, raw: Dict[str, Any], separator: Optional[str] = None
    ) -> Optional[ContextBasedExample]:
        """Transform FineInstructions example to unified schema."""
        text = raw.get("text")

        # Skip null entries
        if text is None or not text.strip():
            return None

        warc_record_id = raw.get("warc_record_id", "")
        token_count = raw.get("token_count", 0)
        template_i = raw.get("template_i", 0)

        # Parse instruction and response
        instruction, response = parse_instruction_response(text, separator)

        # Generate example_id
        example_id = (
            f"{warc_record_id}_{template_i}"
            if warc_record_id
            else f"fi_{hash(text) % 1000000}"
        )

        return ContextBasedExample.from_raw(
            documents=[],  # No external context for instruction-following
            query=instruction,
            target_text=response,
            example_id=example_id,
            source="fineinstructions",
            instruction=None,  # The query itself is the instruction
            misc={
                "warc_record_id": warc_record_id,
                "token_count": token_count,
                "template_i": template_i,
                "raw_text": text,
            },
        )

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate predictions.

        FineInstructions is primarily for training. Returns length-based metrics.
        """
        if not predictions or not example_miscs:
            return {}

        total_pred_len = sum(len(p) for p in predictions)
        total_ref_len = sum(
            len(m.get("target_text", "")) for m in example_miscs
        )
        n = len(predictions)

        return {
            "avg_prediction_length": total_pred_len / n,
            "avg_reference_length": total_ref_len / n if total_ref_len > 0 else 0.0,
            "num_examples": float(n),
        }


@register_task("fineinstructions_filtered")
class FineInstructionsFilteredTask(FineInstructionsTask):
    """
    FineInstructions with quality filtering.

    Use with judge files to filter by quality score.
    """

    @property
    def name(self) -> str:
        return "fineinstructions_filtered"

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        """
        Initialize with filtering parameters.

        Additional kwargs:
            min_quality_score: Minimum quality score (1-5)
            min_token_count: Minimum tokens to include
            max_token_count: Maximum tokens to include
        """
        state = super().init_state(split, **kwargs)
        state["min_quality_score"] = kwargs.get("min_quality_score", 3)
        state["min_token_count"] = kwargs.get("min_token_count", 10)
        state["max_token_count"] = kwargs.get("max_token_count", 8192)
        return state

    @classmethod
    def _map_example(
        cls,
        raw: Dict[str, Any],
        separator: Optional[str] = None,
        min_quality_score: int = 3,
        min_token_count: int = 10,
        max_token_count: int = 8192,
    ) -> Optional[ContextBasedExample]:
        """Transform with quality filtering."""
        # Filter by token count
        token_count = raw.get("token_count", 0)
        if token_count < min_token_count or token_count > max_token_count:
            return None

        # Filter by quality score if available
        quality_score = raw.get("quality_score")
        if quality_score is not None and quality_score < min_quality_score:
            return None

        # Delegate to parent
        result = FineInstructionsTask._map_example(raw, separator)

        # Add quality score to misc if available
        if result is not None and quality_score is not None:
            result.misc["quality_score"] = quality_score

        return result
