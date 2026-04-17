"""MultiNews task implementation."""

from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.metrics import compute_rouge
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


@register_task("multinews")
class MultiNewsTask(BaseTask):
    """
    Multi-News (multi-document summarization) task.

    Summarizes multiple news articles about the same topic.
    """

    DOCUMENT_SEPARATOR = "|||||"

    @property
    def name(self) -> str:
        return "multinews"

    @property
    def metrics(self) -> List[str]:
        return ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    @classmethod
    def prepare_data(cls, split: str = "train", **kwargs):
        dataset = load_dataset("multi_news", split=split)
        shuffle = kwargs.get("shuffle", split == "train")
        seed = kwargs.get("seed", 42)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        max_examples = kwargs.get("max_examples")
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        return dataset

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        return {"idx": 0, "exhausted": False}

    @classmethod
    def read(
        cls, data, state: Dict[str, Any], batch_size: int
    ) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]], Dict[str, Any]
    ]:
        idx = state["idx"]
        examples = []
        end_idx = min(idx + batch_size, len(data))
        for i in range(idx, end_idx):
            raw = data[i]
            ex = cls._map_example(raw, i)
            examples.append(ex)
        new_state = {**state, "idx": end_idx, "exhausted": end_idx >= len(data)}
        return examples, new_state

    @classmethod
    def _map_example(cls, raw: Dict[str, Any], idx: int) -> ContextBasedExample:
        """Transform Multi-News example to unified schema."""
        document_text = raw["document"]
        summary = raw["summary"]

        # Split documents by separator
        documents = [
            doc.strip()
            for doc in document_text.split(cls.DOCUMENT_SEPARATOR)
            if doc.strip()
        ]

        # Generate example_id if not present
        example_id = raw.get("id", f"multinews_{idx}")

        return ContextBasedExample.from_raw(
            documents=documents,
            query="Summarize the key information from all the above articles into a coherent summary.",
            target_text=summary,
            example_id=example_id,
            source="multinews",
            instruction="You are given multiple news articles about the same topic. Write a comprehensive summary that captures the key information from all articles.",
            misc={
                "num_documents": len(documents),
            },
        )

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate using ROUGE metrics."""
        references = [m.get("target_text", "") for m in example_miscs]
        return compute_rouge(predictions, references)
