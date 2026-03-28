"""HotpotQA task implementation."""

from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.metrics import compute_em_f1_multi_ref
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


@register_task("hotpotqa")
class HotpotQATask(BaseTask):
    """
    HotpotQA (multi-hop question answering) task.

    Requires reasoning across multiple documents.
    """

    @property
    def name(self) -> str:
        return "hotpotqa"

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
                subset: Dataset subset ("fullwiki" or "distractor", default "distractor")
                max_examples: Optional cap on examples
                shuffle: Whether to shuffle (default True for train)
                seed: Random seed for shuffling
        """
        subset = kwargs.get("subset", "distractor")
        dataset = load_dataset("hotpot_qa", subset, split=split)

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
        """Transform HotpotQA example to unified schema."""
        question = raw["question"]
        answer = raw["answer"]

        # Build documents from context
        context = raw["context"]
        titles = (
            context["title"]
            if isinstance(context["title"], list)
            else list(context["title"])
        )
        sentences_list = (
            context["sentences"]
            if isinstance(context["sentences"], list)
            else list(context["sentences"])
        )

        documents = []
        for title, sentences in zip(titles, sentences_list):
            doc_text = f"{title}\n" + " ".join(sentences)
            documents.append(doc_text)

        # Extract supporting facts info
        supporting_facts = raw.get("supporting_facts", {})

        return ContextBasedExample.from_raw(
            documents=documents,
            query=question,
            target_text=answer,
            example_id=raw["id"],
            source="hotpotqa",
            instruction="Answer the question based on the provided context. This may require reasoning across multiple documents.",
            misc={
                "type": raw.get("type", ""),
                "level": raw.get("level", ""),
                "supporting_facts": supporting_facts,
                "titles": titles,
            },
        )

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate using Exact Match and F1 (single reference)."""
        # HotpotQA uses single reference answers
        # target_text should be in misc from evaluation context
        ground_truths_list = []
        for misc in example_miscs:
            target = misc.get("target_text", misc.get("answer", ""))
            ground_truths_list.append([target])

        em, f1 = compute_em_f1_multi_ref(predictions, ground_truths_list)
        return {"exact_match": em, "f1": f1}
