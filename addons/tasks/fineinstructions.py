"""FineInstructions Nemotron task implementation."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

_DEBUG = os.environ.get("DEBUG") == "1"

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
    def prepare_data(cls, split: str = "train", **kwargs):
        """Load dataset via streaming, materialize first max_examples into a list."""
        max_examples = kwargs.get("max_examples")

        ds = load_dataset(
            "fineinstructions/fineinstructions_nemotron",
            split=split,
            streaming=True,
        )

        # Materialize from stream — only downloads what's needed
        rows = []
        null_count = 0
        for raw in ds:
            if raw is None:
                null_count += 1
                print(f"[fineinstructions] WARNING: null entry at position {len(rows) + null_count} "
                      f"(valid={len(rows)}, null={null_count})")
                continue
            rows.append(raw)
            if max_examples is not None and len(rows) >= max_examples:
                break
        print(f"[fineinstructions] Loaded {len(rows)} examples ({null_count} null skipped)")

        return rows

    @classmethod
    def init_state(cls, split: str = "train", **kwargs) -> Dict[str, Any]:
        return {
            "idx": 0,
            "response_separator": kwargs.get("response_separator"),
            "exhausted": False,
            "total_read": 0,
            "max_doc_tokens": kwargs.get("max_doc_tokens"),
            "micro_batch_size": kwargs.get("micro_batch_size", 4),
            # Buffer for grouping by warc_record_id
            "pending_group": [],
            "pending_warc_id": None,
            "pending_doc": None,
        }

    @classmethod
    def read(
        cls, data, state: Dict[str, Any], batch_size: int
    ) -> Tuple[
        List[Union[ContextBasedExample, BatchedContextBasedExamples]], Dict[str, Any]
    ]:
        """Read examples, grouping by warc_record_id.

        Consecutive examples sharing the same warc_record_id are bundled into
        BatchedContextBasedExamples with a shared document, up to micro_batch_size.
        """
        separator = state.get("response_separator")
        total_read = state.get("total_read", 0)
        micro_batch = state.get("micro_batch_size", 4)
        idx = state.get("idx", 0)

        pending_group = list(state.get("pending_group", []))
        pending_warc_id = state.get("pending_warc_id")
        pending_doc = state.get("pending_doc")

        results: List[Union[ContextBasedExample, BatchedContextBasedExamples]] = []
        exhausted = False

        def flush_group():
            """Convert pending group into examples with shared document."""
            nonlocal pending_group, pending_warc_id, pending_doc
            if _DEBUG and pending_group:
                print(f"[fineinstructions] flush_group: warc={pending_warc_id}, group_size={len(pending_group)}")
            if not pending_group or not pending_doc:
                pending_group = []
                return
            # Skip group if shared document exceeds token limit
            max_doc_tok = state.get("max_doc_tokens")
            if max_doc_tok and len(pending_doc) / 4 > max_doc_tok:
                pending_group = []
                return
            examples = []
            for raw in pending_group:
                instruction = raw.get("instantiated_instruction") or ""
                response = raw.get("answer") or ""
                if not instruction.strip() or not response.strip():
                    instruction, response = parse_instruction_response(
                        raw.get("text", ""), separator
                    )
                if not instruction or not response:
                    continue
                warc_id = raw.get("warc_record_id", "")
                template_id = raw.get("template_id", "")
                examples.append(ContextBasedExample.from_raw(
                    documents=[pending_doc],
                    query=instruction,
                    target_text=response,
                    example_id=f"{warc_id}_{template_id}" if warc_id else f"fi_{hash(instruction) % 1000000}",
                    source="fineinstructions",
                    misc={
                        "warc_record_id": warc_id,
                        "token_count": raw.get("token_count", 0),
                        "template_id": template_id,
                    },
                ))
            if len(examples) == 1:
                results.append(examples[0])
            elif len(examples) > 1:
                results.append(BatchedContextBasedExamples.from_examples(examples))
            pending_group = []

        while len(results) < batch_size:
            if idx >= len(data):
                exhausted = True
                break
            raw = data[idx]
            idx += 1

            if raw is None:
                continue
            # Skip rows with no instruction or answer
            if not (raw.get("instantiated_instruction") or "").strip():
                continue

            total_read += 1
            warc_id = raw.get("warc_record_id", "")

            # New group or group full?
            if warc_id != pending_warc_id or len(pending_group) >= micro_batch:
                flush_group()
                pending_warc_id = warc_id
                # First example in group provides the document
                pending_doc = (raw.get("text") or "").strip()

            pending_group.append(raw)

        # Flush remaining group
        flush_group()

        new_state = {
            **state,
            "idx": idx,
            "total_read": total_read,
            "exhausted": exhausted,
            "pending_group": pending_group,
            "pending_warc_id": pending_warc_id,
            "pending_doc": pending_doc,
        }
        return results, new_state

    @classmethod
    def _map_example(
        cls, raw: Dict[str, Any], separator: Optional[str] = None,
        document: Optional[str] = None,
    ) -> Optional[ContextBasedExample]:
        """Transform FineInstructions example to unified schema."""
        text = raw.get("text")

        if text is None or not text.strip():
            return None

        warc_record_id = raw.get("warc_record_id", "")
        token_count = raw.get("token_count", 0)
        template_i = raw.get("template_i", 0)

        instruction, response = parse_instruction_response(text, separator)

        example_id = (
            f"{warc_record_id}_{template_i}"
            if warc_record_id
            else f"fi_{hash(text) % 1000000}"
        )

        docs = [document] if document else [text]

        return ContextBasedExample.from_raw(
            documents=docs,
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
