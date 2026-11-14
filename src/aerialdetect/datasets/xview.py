"""Convert xView (https://xviewdataset.org) to YOLO chips at a chosen GSD.

xView ships ~850 labelled WorldView-3 scenes at ~0.3 m with ~1M boxes in 60 classes
(labels only for the train split, so we carve our own val split from it).

GSD matching: to make a model that works on 0.6 m NAIP, use ``degrade_gsd=0.6`` and
``chip_gsd=0.3``. Each scene is blurred down to 0.6 m worth of detail, then
resampled back to a 0.3 m pixel grid. That is exactly what NAIP 0.6 m looks like after
the detector upsamples it 2x at inference time, so train and inference images match.
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
import yaml
from tqdm import tqdm

from ..tiling import axis_starts

log = logging.getLogger(__name__)

XVIEW_NATIVE_GSD = 0.3


def load_class_map(path: str | Path) -> dict[str, list[int]]:
    """Read ``{new_class_name: [xview_type_id, ...]}`` from YAML (key ``classes``)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {str(k): [int(t) for t in v] for k, v in data["classes"].items()}


def resample_to_gsd(img: np.ndarray, native_gsd: float, chip_gsd: float,
                    degrade_gsd: float | None = None) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    if degrade_gsd and degrade_gsd > native_gsd:
        k = native_gsd / degrade_gsd
        small = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    f = native_gsd / chip_gsd
    if abs(f - 1) > 1e-6:
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                         interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    return img


def clip_boxes_to_tile(boxes: np.ndarray, classes: np.ndarray, x0: int, y0: int,
                       tw: int, th: int, min_visible: float) -> list[str]:
    """YOLO label lines for boxes (x0,y0,x1,y1 image px) that are sufficiently inside a tile."""
    if len(boxes) == 0:
        return []
    b = boxes - np.array([x0, y0, x0, y0], dtype="float64")
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, tw)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, th)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    visible = np.where(area > 0, (cw.clip(0) * ch.clip(0)) / np.maximum(area, 1e-9), 0)
    keep = (visible >= min_visible) & (cw >= 2) & (ch >= 2)
    lines = []
    for cls, (bx0, by0, bx1, by1) in zip(classes[keep], c[keep]):
        lines.append(f"{int(cls)} {(bx0 + bx1) / 2 / tw:.6f} {(by0 + by1) / 2 / th:.6f} "
                     f"{(bx1 - bx0) / tw:.6f} {(by1 - by0) / th:.6f}")
    return lines


def convert_xview(
    geojson: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    classes: dict[str, list[int]],
    chip_gsd: float = 0.3,
    degrade_gsd: float | None = None,
    native_gsd: float = XVIEW_NATIVE_GSD,
    tile_px: int = 640,
    overlap_px: int = 64,
    val_fraction: float = 0.1,
    min_visible: float = 0.4,
    background_fraction: float = 0.05,
    seed: int = 0,
) -> Path:
    """Write ``images/{train,val}``, ``labels/{train,val}`` and ``data.yaml``. Returns data.yaml path."""
    import cv2

    type_to_cls: dict[int, int] = {}
    for i, (name, ids) in enumerate(classes.items()):
        for t in ids:
            if t in type_to_cls:
                raise ValueError(f"xView type_id {t} mapped to more than one class")
            type_to_cls[t] = i

    with open(geojson, encoding="utf-8") as f:
        features = json.load(f)["features"]
    boxes: dict[str, list[tuple]] = defaultdict(list)
    labelled_images: set[str] = set()
    for feat in features:
        p = feat["properties"]
        labelled_images.add(p["image_id"])
        cls = type_to_cls.get(int(p["type_id"]))
        if cls is None:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in p["bounds_imcoords"].split(","))
        except (KeyError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            boxes[p["image_id"]].append((cls, x0, y0, x1, y1))

    image_dir, out_dir = Path(image_dir), Path(out_dir)
    image_ids = sorted(p.name for p in image_dir.glob("*.tif") if p.name in labelled_images)
    if not image_ids:
        raise FileNotFoundError(f"no labelled .tif images found in {image_dir}")
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_ids = set(shuffled[:n_val])
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)

    f = native_gsd / chip_gsd
    step = tile_px - overlap_px
    stats: Counter = Counter()
    for image_id in tqdm(image_ids, desc="xView", unit="img"):
        split = "val" if image_id in val_ids else "train"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(image_dir / image_id) as src:
                img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(resample_to_gsd(img, native_gsd, chip_gsd, degrade_gsd))

        recs = boxes.get(image_id, [])
        b = np.array([r[1:] for r in recs], dtype="float64").reshape(-1, 4) * f
        c = np.array([r[0] for r in recs], dtype="int64")
        h, w = img.shape[:2]
        for ty in axis_starts(h, tile_px, step):
            for tx in axis_starts(w, tile_px, step):
                chip = img[ty:ty + tile_px, tx:tx + tile_px]
                th, tw = chip.shape[:2]
                lines = clip_boxes_to_tile(b, c, tx, ty, tw, th, min_visible)
                if not lines and rng.random() > background_fraction:
                    continue  # keep only a few empty chips as negatives
                stem = f"{Path(image_id).stem}_{tx}_{ty}"
                cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"),
                            cv2.cvtColor(chip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                stats[f"{split}_chips"] += 1
                stats[f"{split}_boxes"] += len(lines)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(classes)},
        # Not used by Ultralytics; recorded so inference can use the right GSD.
        "chip_gsd": chip_gsd,
        "degrade_gsd": degrade_gsd,
    }, sort_keys=False), encoding="utf-8")
    log.info("xView conversion done: %s", dict(stats))
    return data_yaml
