"""LongBench task implementation."""

from typing import Any, Dict, List, Tuple, Union

from datasets import load_dataset

from addons.tasks.base import BaseTask
from addons.tasks.metrics import compute_em_f1_multi_ref, compute_rouge
from addons.tasks.registry import register_task
from addons.tasks.schema import BatchedContextBasedExamples, ContextBasedExample


# Subtask categories for metric selection
QA_DATASETS = {
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "triviaqa",
}
SUMMARIZATION_DATASETS = {
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "samsum",
}

# Task-specific instructions
LONGBENCH_INSTRUCTIONS = {
    # Single-document QA
    "narrativeqa": "Answer the question based on the narrative.",
    "qasper": "Answer the question based on the scientific paper.",
    "multifieldqa_en": "Answer the question based on the document.",
    "multifieldqa_zh": "根据文档回答问题。",
    # Multi-document QA
    "hotpotqa": "Answer the question using information from multiple documents.",
    "2wikimqa": "Answer the question using information from multiple Wikipedia articles.",
    "musique": "Answer the multi-hop question based on the provided documents.",
    # Summarization
    "gov_report": "Summarize the government report.",
    "qmsum": "Summarize the meeting transcript based on the query.",
    "multi_news": "Summarize the news articles.",
    "vcsum": "总结会议内容。",
    # Few-shot learning
    "trec": "Classify the question into one of the given categories.",
    "triviaqa": "Answer the trivia question.",
    "samsum": "Summarize the dialogue.",
    "lsht": "对文本进行分类。",
    # Code
    "lcc": "Complete the code based on the context.",
    "repobench-p": "Complete the code based on the repository context.",
    # Synthetic
    "passage_count": "Count the number of passages.",
    "passage_retrieval_en": "Identify the relevant passage.",
    "passage_retrieval_zh": "找出相关段落。",
}


@register_task("longbench")
class LongBenchTask(BaseTask):
    """
    LongBench (long-context benchmark) task.

    Covers 20+ subtasks across QA, summarization, and code.
    """

    @property
    def name(self) -> str:
        return "longbench"

    @property
    def metrics(self) -> List[str]:
        # Returns both QA and summarization metrics
        return ["exact_match", "f1", "rouge1", "rouge2", "rougeL"]

    @classmethod
    def prepare_data(cls, split: str = "test", **kwargs):
        subtask = kwargs.get("subtask")
        if not subtask:
            raise ValueError(
                "LongBench requires 'subtask' parameter. "
                f"Available: {list(LONGBENCH_INSTRUCTIONS.keys())}"
            )
        dataset = load_dataset("THUDM/LongBench", subtask, split=split)
        shuffle = kwargs.get("shuffle", False)
        seed = kwargs.get("seed", 42)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        max_examples = kwargs.get("max_examples")
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        return dataset

    @classmethod
    def init_state(cls, split: str = "test", **kwargs) -> Dict[str, Any]:
        return {"subtask": kwargs.get("subtask"), "idx": 0, "exhausted": False}

    @classmethod
    def read(
        cls, data, state: Dict[str, Any], batch_size: int
    ) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]], Dict[str, Any]
    ]:
        subtask = state["subtask"]
        idx = state["idx"]
        examples = []
        end_idx = min(idx + batch_size, len(data))
        for i in range(idx, end_idx):
            raw = data[i]
            ex = cls._map_example(raw, subtask)
            examples.append(ex)
        new_state = {**state, "idx": end_idx, "exhausted": end_idx >= len(data)}
        return examples, new_state

    @classmethod
    def _map_example(cls, raw: Dict[str, Any], subtask: str) -> ContextBasedExample:
        """Transform LongBench example to unified schema."""
        input_text = raw.get("input", "")
        context = raw.get("context", "")
        answers = raw.get("answers", [])
        dataset_name = raw.get("dataset", subtask)

        # Use context if available, otherwise use input as document
        if context:
            documents = [context]
            query = input_text
        else:
            documents = [input_text]
            query = "Based on the above context, provide the answer."

        target_text = answers[0] if answers else ""
        instruction = LONGBENCH_INSTRUCTIONS.get(
            dataset_name, "Complete the task based on the provided context."
        )

        return ContextBasedExample.from_raw(
            documents=documents,
            query=query,
            target_text=target_text,
            example_id=raw.get("_id", ""),
            source="longbench",
            instruction=instruction,
            misc={
                "all_answers": answers,
                "dataset": dataset_name,
                "language": raw.get("language", "en"),
                "length": raw.get("length", 0),
                "all_classes": raw.get("all_classes"),
            },
        )

    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        example_miscs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate using task-appropriate metrics.

        Uses F1/EM for QA tasks, ROUGE for summarization.
        """
        if not predictions or not example_miscs:
            return {}

        # Determine task type from first example
        dataset = example_miscs[0].get("dataset", "")

        if dataset in SUMMARIZATION_DATASETS:
            references = [m.get("target_text", "") for m in example_miscs]
            return compute_rouge(predictions, references)
        else:
            # Default to QA metrics
            ground_truths_list = [
                m.get("all_answers", [m.get("target_text", "")]) for m in example_miscs
            ]
            em, f1 = compute_em_f1_multi_ref(predictions, ground_truths_list)
            return {"exact_match": em, "f1": f1}
