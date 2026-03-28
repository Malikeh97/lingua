"""
Task definitions for FineSearch.

Each task provides:
- Data loading (functional state-passing: init_state, read)
- Format conversion (map_example)
- Evaluation metrics (evaluate)
"""

from addons.tasks.base import BaseTask
from addons.tasks.registry import register_task, get_task, list_tasks
from addons.tasks.schema import ContextBasedExample, BatchedContextBasedExamples
from addons.tasks.metrics import (
    compute_squad_metrics,
    compute_em_f1_multi_ref,
    compute_rouge,
    compute_bleu,
    get_metric,
)

# Import tasks to register them
from addons.tasks.squad import SQUADTask
from addons.tasks.hotpotqa import HotpotQATask
from addons.tasks.longbench import LongBenchTask
from addons.tasks.multinews import MultiNewsTask
from addons.tasks.arxiv import ArxivTask
from addons.tasks.writingbench import WritingBenchTask
from addons.tasks.fineinstructions import FineInstructionsTask, FineInstructionsFilteredTask

__all__ = [
    # Base classes
    "BaseTask",
    # Registry
    "register_task",
    "get_task",
    "list_tasks",
    # Schema
    "ContextBasedExample",
    "BatchedContextBasedExamples",
    # Metrics
    "compute_squad_metrics",
    "compute_em_f1_multi_ref",
    "compute_rouge",
    "compute_bleu",
    "get_metric",
    # Tasks
    "SQUADTask",
    "HotpotQATask",
    "LongBenchTask",
    "MultiNewsTask",
    "ArxivTask",
    "WritingBenchTask",
    "FineInstructionsTask",
    "FineInstructionsFilteredTask",
]
