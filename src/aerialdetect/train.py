"""Fine-tune an Ultralytics YOLO model on aerial chips."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Aerial imagery has no "up": vertical flips are as valid as horizontal ones.
# Scale jitter is kept small because the pipeline fixes GSD; objects should stay
# roughly the size they will be at inference.
AERIAL_AUGMENT: dict[str, Any] = {
    "flipud": 0.5,
    "fliplr": 0.5,
    "scale": 0.25,
    "degrees": 0.0,  # rotation inflates axis-aligned boxes; raise it for OBB models only
    "mosaic": 1.0,
}


def train(
    data: str | Path,
    model: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    project: str | Path = "runs",
    name: str = "train",
    **overrides: Any,
) -> Path:
    """Train and return the path to ``best.pt``.

    ``data`` is a dataset YAML: e.g. the one from ``prep-xview``, or Ultralytics'
    built-in ``DOTAv1.yaml`` (use an ``*-obb.pt`` model) or ``DOTA8.yaml`` for a smoke test.
    Extra keyword arguments go straight to ``YOLO.train`` (``workers``, ``patience``, ...).
    """
    from ultralytics import YOLO

    args = {**AERIAL_AUGMENT, **overrides}
    yolo = YOLO(model)
    yolo.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
               project=str(Path(project).resolve()), name=name, **args)
    best = Path(yolo.trainer.best)
    log.info("best weights: %s", best)
    return best
