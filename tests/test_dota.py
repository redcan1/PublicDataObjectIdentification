from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")

from aerialdetect.datasets.dota import DOTA_NAMES, prep_dota, prune_background  # noqa: E402


def _fake_dota(root: Path, w: int = 1000, h: int = 800) -> None:
    """One scene per split with a single 'small vehicle' near the top-left."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(root / "images" / split / "P0001.png"), np.full((h, w, 3), 90, np.uint8))
        x0, y0, x1, y1 = 100 / w, 100 / h, 115 / w, 107 / h  # 15 x 7 px car
        (root / "labels" / split / "P0001.txt").write_text(
            f"10 {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}\n", encoding="utf-8")


def test_prep_dota_chips_at_native_resolution(tmp_path):
    raw = tmp_path / "raw" / "DOTAv1"
    _fake_dota(raw)
    data_yaml = prep_dota(tmp_path / "raw", tmp_path / "out", crop_px=640, gap_px=128, download=False,
                          keep_background=1.0)

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert cfg["names"][10] == "small vehicle" and len(cfg["names"]) == len(DOTA_NAMES)

    chips = sorted((tmp_path / "out" / "images" / "train").glob("*.jpg"))
    assert chips and all(cv2.imread(str(c)).shape[:2] == (640, 640) for c in chips)

    # the car is still ~15 px long inside its chip, i.e. not rescaled
    first = tmp_path / "out" / "labels" / "train" / f"{chips[0].stem}.txt"
    xs = np.array(first.read_text().split()[1::2], float) * 640
    assert 14 <= xs.max() - xs.min() <= 16


def test_prune_background_keeps_labelled_chips(tmp_path):
    for split in ("train", "val"):
        im, lb = tmp_path / "images" / split, tmp_path / "labels" / split
        im.mkdir(parents=True)
        lb.mkdir(parents=True)
        for i in range(50):
            (im / f"c{i}.jpg").write_bytes(b"x")
        for i in range(5):  # first five chips have objects
            (lb / f"c{i}.txt").write_text("10 0 0 0.1 0 0.1 0.1 0 0.1\n")

    removed = prune_background(tmp_path, keep_frac=0.0)

    assert removed == {"train": 45, "val": 45}
    assert sorted(p.name for p in (tmp_path / "images" / "train").iterdir()) == [f"c{i}.jpg" for i in range(5)]
