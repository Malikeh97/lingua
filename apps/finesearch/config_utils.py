"""
Config utilities: YAML loading, CLI parsing, dataclass conversion.

Replaces OmegaConf with stdlib yaml + simple key=value CLI parsing.
"""

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_yaml(path: str) -> dict:
    """Load a YAML file as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def dump_yaml(data, path: str):
    """Dump dataclass or dict to YAML."""
    d = dataclasses.asdict(data) if dataclasses.is_dataclass(data) else data
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def parse_cli(argv: list[str] | None = None) -> Dict[str, Any]:
    """
    Parse CLI args in key=value format into a nested dict.

    Supports:
        config=path.yaml          -> {"config": "path.yaml"}
        model.encoder_name=bert   -> {"model": {"encoder_name": "bert"}}
        steps=5000                -> {"steps": 5000}
        eval.tasks='[squad,hot]'  -> {"eval": {"tasks": ["squad", "hot"]}}
        distributed.compile=true  -> {"distributed": {"compile": True}}
    """
    if argv is None:
        argv = sys.argv[1:]

    result: Dict[str, Any] = {}

    for arg in argv:
        if "=" not in arg:
            continue

        key, value = arg.split("=", 1)
        parsed_value = _parse_value(value)

        # Handle dotted keys: a.b.c=val -> {"a": {"b": {"c": val}}}
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = parsed_value

    return result


def _parse_value(value: str) -> Any:
    """Parse a CLI value string into a Python object."""
    # Strip surrounding quotes
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]

    # YAML-style list: [a,b,c]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = [_parse_value(item.strip()) for item in inner.split(",")]
        return items

    # Booleans
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # None
    if value.lower() in ("null", "none", "~"):
        return None

    # Numbers
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass

    # String
    return value


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def dataclass_defaults(cls) -> dict:
    """Extract defaults from a dataclass, recursing into nested dataclasses."""
    result = {}
    for f in dataclasses.fields(cls):
        if dataclasses.is_dataclass(f.type):
            result[f.name] = dataclass_defaults(f.type)
        elif f.default is not dataclasses.MISSING:
            result[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            result[f.name] = f.default_factory()
    return result


def _unwrap_optional(tp):
    """Extract T from Optional[T] (i.e. Union[T, None])."""
    import typing
    origin = getattr(tp, "__origin__", None)
    if origin is Union:
        args = [a for a in tp.__args__ if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def dict_to_dataclass(cls: Type[T], data: dict) -> T:
    """Convert a dict to a dataclass, recursing into nested dataclasses.
    Warns and skips keys not present in the dataclass."""
    kwargs = {}
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    for k, v in data.items():
        if k not in field_types:
            logger.warning(f"Ignoring unknown config key '{k}' for {cls.__name__}")
            continue
        ft = _unwrap_optional(field_types[k])
        if ft and dataclasses.is_dataclass(ft) and isinstance(v, dict):
            kwargs[k] = dict_to_dataclass(ft, v)
        else:
            # Coerce types: yaml.safe_load parses "1e-4" as str, not float
            if ft is float and isinstance(v, str):
                v = float(v)
            elif ft is int and isinstance(v, str):
                v = int(v)
            elif ft is bool and isinstance(v, str):
                v = v.lower() in ("true", "yes", "1")
            kwargs[k] = v
    return cls(**kwargs)


def load_config(cls: Type[T], argv: list[str] | None = None) -> T:
    """
    Full config loading pipeline: defaults -> YAML files -> CLI overrides -> dataclass.

    Supports multiple config files merged left to right:
        config=base.yaml,train.yaml  key=value ...
    """
    cli = parse_cli(argv)

    if "config" not in cli:
        print(f"Usage: python -m <module> config=<config.yaml>[,<overlay.yaml>,...] [overrides]")
        sys.exit(1)

    config_arg = cli.pop("config")
    if isinstance(config_arg, list):
        config_paths = config_arg
    elif "," in str(config_arg):
        config_paths = [p.strip() for p in str(config_arg).split(",")]
    else:
        config_paths = [config_arg]

    merged = dataclass_defaults(cls)
    for path in config_paths:
        merged = deep_merge(merged, load_yaml(path))
    merged = deep_merge(merged, cli)

    return dict_to_dataclass(cls, merged)
