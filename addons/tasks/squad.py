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

    # When True, query is prepended to the document (encoder side) so the
    # decoder only generates the answer.  This gives C/Q//A format which
    # performs much better than C//Q/A (~80% vs ~20% EM).
    query_in_encoder: bool = True

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
                query_in_encoder: Put query in encoder (default True)
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

        # Query placement flag
        query_in_encoder = kwargs.get("query_in_encoder", cls.query_in_encoder)

        return {
            "dataset": dataset,
            "idx": 0,
            "epoch": 0,
            "split": split,
            "seed": seed,
            "exhausted": False,
            "query_in_encoder": query_in_encoder,
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

        query_in_encoder = state.get("query_in_encoder", cls.query_in_encoder)

        end_idx = min(idx + batch_size, len(dataset))
        for i in range(idx, end_idx):
            raw = dataset[i]
            ex = cls._map_example(raw, query_in_encoder=query_in_encoder)
            examples.append(ex)

        split = state.get("split", "train")

        if end_idx >= len(dataset) and split == "train":
            # Start new epoch: reset idx, bump epoch, re-shuffle
            epoch = state.get("epoch", 0) + 1
            base_seed = state.get("seed", 42)
            new_state = {
                **state,
                "dataset": dataset.shuffle(seed=base_seed + epoch),
                "idx": 0,
                "epoch": epoch,
                "exhausted": False,
            }
        else:
            new_state = {
                **state,
                "idx": end_idx,
                "exhausted": end_idx >= len(dataset),
            }
        return examples, new_state

    @classmethod
    def _map_example(
        cls, raw: Dict[str, Any], query_in_encoder: bool = True
    ) -> ContextBasedExample:
        """Transform SQUAD example to unified schema.

        When query_in_encoder=True (default), the question is prepended to the
        context document so both go through the encoder (C/Q//A format).
        The decoder query is left empty and only generates the answer.

        When query_in_encoder=False, the question stays in the decoder query
        field (C//Q/A format).
        """
        context = raw["context"]
        question = raw["question"]
        answers = raw["answers"]

        # Use first answer as target, store all for multi-ref evaluation
        target_text = answers["text"][0] if answers["text"] else ""

        if query_in_encoder:
            # C/Q//A: question + context → encoder, answer → decoder
            document = f"{question}\n\n{context}"
            query = ""
        else:
            # C//Q/A: context → encoder, question + answer → decoder
            document = context
            query = question

        return ContextBasedExample.from_raw(
            documents=[document],
            query=query,
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
