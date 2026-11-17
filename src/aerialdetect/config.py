"""YAML run configuration with defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "aoi": {"bbox": None, "path": None},
    "source": {"type": "naip", "year": None},
    "model": {
        "weights": None,
        "gsd": 0.3,        # metres/pixel of the chips the model was trained on
        "conf": 0.25,
        "iou": 0.5,
        "device": None,    # e.g. 0 for first GPU, "cpu"
    },
    "detect": {
        "tile_px": 640,
        "overlap_px": 128,
        "bands": [1, 2, 3],
        "batch_size": 8,
        "edge_margin_px": 2.0,
        "nms_iou": 0.5,
        "nms_ios": 0.8,
        "classes": None,    # optional keep-list of class names; None = keep every class
        "min_area_m2": {},  # optional per-class size filter, e.g. {car: 4}
        "max_area_m2": {},
    },
    "analysis": {
        "tracts": {"enabled": False, "year": 2023, "count_classes": [], "footprint_classes": []},
        "parking": {"enabled": False, "lots_path": None, "vehicle_classes": [], "m2_per_stall": 30.0},
    },
    "output": {"dir": "outputs/run", "gpkg": "results.gpkg"},
    "cache_dir": "data/cache",
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    """``set_dotted(cfg, "source.year", 2020)`` sets ``cfg["source"]["year"] = 2020``."""
    *parents, last = key.split(".")
    node = cfg
    for k in parents:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"config: {key!r} goes through a non-mapping value")
    node[last] = value


def load_config(path: str | Path, overrides: list[tuple[str, Any]] | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    for k, v in overrides or []:
        set_dotted(user, k, v)
    cfg = deep_merge(DEFAULTS, user)
    if not cfg["model"]["weights"]:
        raise ValueError("config: model.weights is required")
    return cfg
