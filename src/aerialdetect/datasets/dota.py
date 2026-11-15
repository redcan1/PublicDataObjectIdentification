"""DOTA v1 -> GSD-preserving YOLO-OBB chips.

DOTA scenes are up to ~4000 px across. Training on them directly at ``imgsz=640`` would shrink
each whole scene to 640 px, making a car one or two pixels long. Instead we cut the scenes into
``crop_px`` windows at native resolution (Ultralytics' ``split_dota``), so objects keep their real
pixel size, and train at ``imgsz == crop_px``, which is also the pipeline's ``tile_px``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DOTA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"

DOTA_NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
    "roundabout", "soccer ball field", "swimming pool",
]


def find_dota_root(base: Path) -> Path | None:
    """Return the folder under ``base`` that contains ``images/train`` and ``labels/train``."""
    for cand in (base, *sorted(p for p in base.rglob("*") if p.is_dir())):
        if (cand / "images" / "train").is_dir() and (cand / "labels" / "train").is_dir():
            return cand
    return None


def download_dota(raw_dir: str | Path) -> Path:
    """Download and unzip DOTA v1 (~2 GB) into ``raw_dir`` unless it is already there."""
    raw_dir = Path(raw_dir)
    root = find_dota_root(raw_dir) if raw_dir.exists() else None
    if root is None:
        from ultralytics.utils.downloads import download

        raw_dir.mkdir(parents=True, exist_ok=True)
        download(DOTA_URL, dir=raw_dir, unzip=True, delete=True)
        root = find_dota_root(raw_dir)
        if root is None:
            raise FileNotFoundError(f"DOTA images/train + labels/train not found under {raw_dir}")
    log.info("DOTA root: %s", root)
    return root


def prune_background(out_dir: str | Path, keep_frac: float = 0.1, seed: int = 0) -> dict[str, int]:
    """Delete all but ``keep_frac`` of the chips that have no objects, per split.

    Splitting DOTA at 640 px yields ~57% empty chips. A few teach the model what "nothing" looks
    like; the rest just make each epoch slower. Returns the number of chips removed per split.
    """
    import random

    out_dir, rng, removed = Path(out_dir), random.Random(seed), {}
    for split in ("train", "val"):
        im_dir, lb_dir = out_dir / "images" / split, out_dir / "labels" / split
        empty = sorted(p for p in im_dir.glob("*") if not (lb_dir / f"{p.stem}.txt").is_file())
        drop = [p for p in empty if rng.random() >= keep_frac]
        for p in drop:
            p.unlink()
        for cache in lb_dir.parent.glob(f"{split}*.cache"):  # stale Ultralytics label cache
            cache.unlink()
        removed[split] = len(drop)
        log.info("%s: removed %d of %d empty chips", split, len(drop), len(empty))
    return removed


def prep_dota(raw_dir: str | Path, out_dir: str | Path, crop_px: int = 640, gap_px: int = 128,
              download: bool = True, keep_background: float = 0.1) -> Path:
    """Split DOTA train/val into ``crop_px`` chips at native resolution and write ``data.yaml``."""
    from ultralytics.data.split_dota import split_trainval

    raw_dir, out_dir = Path(raw_dir), Path(out_dir).resolve()
    root = download_dota(raw_dir) if download else find_dota_root(raw_dir)
    if root is None:
        raise FileNotFoundError(f"DOTA images/train + labels/train not found under {raw_dir}")

    split_trainval(data_root=str(root), save_dir=str(out_dir), crop_size=crop_px, gap=gap_px, rates=(1.0,))
    if keep_background < 1:
        prune_background(out_dir, keep_background)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(DOTA_NAMES)),
    }, sort_keys=False), encoding="utf-8")
    n_train = len(list((out_dir / "images" / "train").glob("*")))
    n_val = len(list((out_dir / "images" / "val").glob("*")))
    log.info("DOTA chips: %d train, %d val -> %s", n_train, n_val, data_yaml)
    return data_yaml
