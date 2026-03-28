"""
Task registry for discovering and loading tasks.
"""

from typing import Dict, Type

from addons.tasks.base import BaseTask


# Global registry
_TASKS: Dict[str, Type[BaseTask]] = {}


def register_task(name: str):
    """Decorator to register a task class."""

    def decorator(cls: Type[BaseTask]):
        _TASKS[name] = cls
        return cls

    return decorator


def get_task(name: str, **kwargs) -> BaseTask:
    """Get task instance by name."""
    if name not in _TASKS:
        raise ValueError(f"Unknown task: {name}. Available: {list(_TASKS.keys())}")
    return _TASKS[name](**kwargs)


def list_tasks() -> list:
    """List all registered task names."""
    return list(_TASKS.keys())
