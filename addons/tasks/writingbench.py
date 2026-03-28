"""WritingBench task implementation."""

from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.metrics import compute_rouge
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


# Task-type specific meta-instructions
WRITING_INSTRUCTIONS = {
    "creative": "You are a creative writer. Follow the instructions to produce engaging and original content.",
    "academic": "You are an academic writer. Follow the instructions to produce clear, well-structured, and scholarly content.",
    "technical": "You are a technical writer. Follow the instructions to produce precise, accurate, and well-documented content.",
    "business": "You are a business writer. Follow the instructions to produce professional and effective business communication.",
    "narrative": "You are a narrative writer. Follow the instructions to craft a compelling story.",
    "persuasive": "You are writing to persuade. Follow the instructions to construct a convincing argument.",
    "descriptive": "You are a descriptive writer. Follow the instructions to create vivid and detailed descriptions.",
    "expository": "You are an expository writer. Follow the instructions to explain concepts clearly and thoroughly.",
}


@register_task("writingbench")
class WritingBenchTask(BaseTask):
    """
    WritingBench (writing evaluation) task.

    Covers various writing styles: creative, academic, technical, etc.
    """

    @property
    def name(self) -> str:
        return "writingbench"

    @property
    def metrics(self) -> List[str]:
        return ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        """
        Initialize reading state.

        Args:
            split: Data split
            **kwargs:
                dataset_path: HuggingFace dataset path (default: "writingbench/WritingBench")
                max_examples: Optional cap on examples
                shuffle: Whether to shuffle (default True for train)
                seed: Random seed for shuffling
        """
        dataset_path = kwargs.get("dataset_path", "writingbench/WritingBench")

        try:
            dataset = load_dataset(dataset_path, split=split)
        except Exception:
            # Fallback: create empty dataset if not available
            raise ValueError(
                f"WritingBench dataset not found at {dataset_path}. "
                "Please provide a valid dataset_path."
            )

        # Optional shuffle
        shuffle = kwargs.get("shuffle", split == "train")
        seed = kwargs.get("seed", 42)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        # Optional limit
        max_examples = kwargs.get("max_examples")
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        return {
            "dataset": dataset,
            "idx": 0,
            "exhausted": False,
        }

    @classmethod
    def read(
        cls, state: Dict[str, Any], batch_size: int
    ) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]], Dict[str, Any]
    ]:
        """Read batch_size examples."""
        dataset = state["dataset"]
        idx = state["idx"]
        examples = []

        end_idx = min(idx + batch_size, len(dataset))
        for i in range(idx, end_idx):
            raw = dataset[i]
            ex = cls._map_example(raw)
            examples.append(ex)

        new_state = {
            **state,
            "idx": end_idx,
            "exhausted": end_idx >= len(dataset),
        }
        return examples, new_state

    @classmethod
    def _map_example(cls, raw: Dict[str, Any]) -> ContextBasedExample:
        """Transform WritingBench example to unified schema."""
        instruction = raw.get("instruction", "")
        input_text = raw.get("input", "")
        reference = raw.get("reference", raw.get("output", ""))
        task_type = raw.get("task_type", "")
        constraints = raw.get("constraints", [])
        criteria = raw.get("criteria", [])

        # Build documents from input if available
        documents = [input_text] if input_text else []

        # Build query from instruction and constraints
        query = instruction
        if constraints:
            constraints_text = "\n".join(f"- {c}" for c in constraints)
            query = f"{instruction}\n\nConstraints:\n{constraints_text}"

        # Build meta-instruction based on task type
        meta_instruction = WRITING_INSTRUCTIONS.get(
            task_type.lower(),
            "Follow the writing instructions carefully to produce high-quality content.",
        )

        return ContextBasedExample.from_raw(
            documents=documents,
            query=query,
            target_text=reference,
            example_id=raw.get("id", ""),
            source="writingbench",
            instruction=meta_instruction,
            misc={
                "task_type": task_type,
                "constraints": constraints,
                "criteria": criteria,
                "original_instruction": instruction,
            },
        )

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate using ROUGE metrics.

        Note: WritingBench ideally uses LLM-as-judge. ROUGE is a proxy metric.
        """
        # Filter examples with reference outputs
        valid_pairs = [
            (pred, misc.get("target_text", ""))
            for pred, misc in zip(predictions, example_miscs)
            if misc.get("target_text")
        ]

        if not valid_pairs:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        preds, refs = zip(*valid_pairs)
        return compute_rouge(list(preds), list(refs))
