"""ArXiv task implementation."""

from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.metrics import compute_rouge
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


@register_task("arxiv")
class ArxivTask(BaseTask):
    """
    ArXiv (scientific paper summarization) task.

    Generates abstracts from full paper text.
    """

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def metrics(self) -> List[str]:
        return ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        """
        Initialize reading state.

        Args:
            split: Data split ("train", "validation", or "test")
            **kwargs:
                max_examples: Optional cap on examples
                shuffle: Whether to shuffle (default True for train)
                seed: Random seed for shuffling
        """
        # scientific_papers/arxiv dataset
        dataset = load_dataset("scientific_papers", "arxiv", split=split)

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
            ex = cls._map_example(raw, i)
            examples.append(ex)

        new_state = {
            **state,
            "idx": end_idx,
            "exhausted": end_idx >= len(dataset),
        }
        return examples, new_state

    @classmethod
    def _map_example(cls, raw: Dict[str, Any], idx: int) -> ContextBasedExample:
        """Transform arXiv example to unified schema."""
        # Handle different formats
        if "article" in raw:
            article_text = raw["article"]
        elif "sections" in raw:
            # Concatenate sections
            sections = raw["sections"]
            article_text = "\n\n".join(
                f"## {sec.get('heading', 'Section')}\n{sec.get('text', '')}"
                for sec in sections
            )
        else:
            # Fallback to any text field
            article_text = raw.get("text", raw.get("body", ""))

        abstract = raw.get("abstract", "")
        article_id = raw.get("article_id", raw.get("id", str(idx)))
        title = raw.get("title", "")

        # Build document with title if available
        if title:
            document = f"# {title}\n\n{article_text}"
        else:
            document = article_text

        return ContextBasedExample.from_raw(
            documents=[document],
            query="Write a concise abstract summarizing the key contributions, methods, and findings of this paper.",
            target_text=abstract,
            example_id=article_id,
            source="arxiv",
            instruction="You are given a scientific paper. Write an abstract that summarizes the paper's main contributions, methodology, and key findings.",
            misc={
                "title": title,
                "categories": raw.get("categories", raw.get("section_names", "")),
                "article_id": article_id,
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
