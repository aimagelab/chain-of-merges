from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve(path: str, config_dir: Path) -> Path:
    p = Path(path)
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    return p if p.is_absolute() or p.exists() else config_dir / p


def load_config(path: str, overrides: Iterable[str] = (), config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    file = _resolve(path, config_dir)
    raw = _read(file)
    merged: Dict[str, Any] = {}
    for default in raw.pop("defaults", []):
        merged = deep_merge(merged, load_config(default, config_dir=config_dir))
    merged = deep_merge(merged, raw)
    for item in overrides:
        merged = apply_override(merged, item)
    return merged


def apply_override(cfg: Dict[str, Any], item: str) -> Dict[str, Any]:
    if "=" not in item:
        raise ValueError(f"override '{item}' must look like key.sub=value")
    key, raw_value = item.split("=", 1)
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    out = deepcopy(cfg)
    node = out
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value
    return out
