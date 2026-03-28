"""SQUAD task implementation."""

from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.metrics import compute_em_f1_multi_ref
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


@register_task("squad")
class SQUADTask(BaseTask):
    """
    SQUAD (Stanford Question Answering Dataset) task.

    Single-document extractive question answering.
    """

    @property
    def name(self) -> str:
        return "squad"

    @property
    def metrics(self) -> List[str]:
        return ["exact_match", "f1"]

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        """
        Initialize reading state.

        Args:
            split: Data split ("train" or "validation")
            **kwargs:
                max_examples: Optional cap on examples
                shuffle: Whether to shuffle (default True for train)
                seed: Random seed for shuffling
        """
        # Load dataset
        dataset = load_dataset("squad", split=split)

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
        """Transform SQUAD example to unified schema."""
        context = raw["context"]
        question = raw["question"]
        answers = raw["answers"]

        # Use first answer as target, store all for multi-ref evaluation
        target_text = answers["text"][0] if answers["text"] else ""

        return ContextBasedExample.from_raw(
            documents=[context],
            query=question,
            target_text=target_text,
            example_id=raw["id"],
            source="squad",
            instruction="Answer the question based on the context.",
            misc={
                "all_answers": answers["text"],
                "answer_starts": answers["answer_start"],
                "title": raw.get("title", ""),
            },
        )

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate using Exact Match and F1."""
        ground_truths_list = []
        for misc in example_miscs:
            all_answers = misc.get("all_answers", [])
            if not all_answers:
                all_answers = [misc.get("target_text", "")]
            ground_truths_list.append(all_answers)

        em, f1 = compute_em_f1_multi_ref(predictions, ground_truths_list)
        return {"exact_match": em, "f1": f1}
