"""YAML configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from a config dict."""
    node = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node
