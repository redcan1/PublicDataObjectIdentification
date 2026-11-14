"""Imagery sources. Every source ends with the same thing: local, georeferenced GeoTIFF(s).

Adding a new sensor (the drone, a state orthophoto program, Maxar...) means writing
one ``RasterSource`` subclass; detection and analysis code never changes.
"""

from __future__ import annotations

from typing import Any

from .base import RasterSource
from .local import LocalRasterSource
from .naip import NaipSource

__all__ = ["RasterSource", "LocalRasterSource", "NaipSource", "build_source"]


def build_source(cfg: dict[str, Any]) -> RasterSource:
    kind = cfg.get("type", "naip").lower()
    if kind == "naip":
        return NaipSource(year=cfg.get("year"), min_coverage=cfg.get("min_coverage", 0.99))
    if kind in ("local", "drone", "geotiff"):
        paths = cfg.get("paths") or ([cfg["path"]] if cfg.get("path") else [])
        return LocalRasterSource(paths)
    raise ValueError(f"unknown source type {kind!r} (expected naip | local | drone)")
