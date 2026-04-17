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
    def prepare_data(cls, split: str = "train", **kwargs):
        """Load and prepare SQUAD dataset."""
        dataset = load_dataset("squad", split=split)

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
        """Initialize lightweight reading state."""
        return {
            "idx": 0,
            "epoch": 0,
            "split": split,
            "seed": kwargs.get("seed", 42),
            "exhausted": False,
            "query_in_encoder": kwargs.get("query_in_encoder", cls.query_in_encoder),
        }

    @classmethod
    def read(
        cls, data, state: Dict[str, Any], batch_size: int
    ) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]], Dict[str, Any]
    ]:
        """Read batch_size examples."""
        idx = state["idx"]
        query_in_encoder = state.get("query_in_encoder", cls.query_in_encoder)
        examples = []

        end_idx = min(idx + batch_size, len(data))
        for i in range(idx, end_idx):
            raw = data[i]
            ex = cls._map_example(raw, query_in_encoder=query_in_encoder)
            examples.append(ex)

        if end_idx >= len(data) and state.get("split") == "train":
            epoch = state["epoch"] + 1
            new_state = {
                **state,
                "idx": 0,
                "epoch": epoch,
                "exhausted": False,
            }
        else:
            new_state = {
                **state,
                "idx": end_idx,
                "exhausted": end_idx >= len(data),
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
