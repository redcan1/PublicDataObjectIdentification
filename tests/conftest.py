from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from aerialdetect.detect import PixelDetection

ORIGIN = (500_000.0, 3_700_000.0)  # UTM 12N, roughly Arizona
CRS = "EPSG:32612"


def make_raster(path: Path, size: int, gsd: float, squares=(), background: int = 60) -> Path:
    """Gray RGB GeoTIFF with bright squares at (col, row, size_px) in native pixels."""
    arr = np.full((3, size, size), background, dtype=np.uint8)
    for col, row, s in squares:
        arr[:, row:row + s, col:col + s] = 255
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8",
                       crs=CRS, transform=from_origin(*ORIGIN, gsd, gsd)) as dst:
        dst.write(arr)
    return path


class BrightSquarePredictor:
    """Fake detector: one box around all bright pixels in a tile."""

    names = {0: "square"}

    def __call__(self, images):
        out = []
        for im in images:
            mask = im[..., 0] > 200
            dets = []
            if mask.any():
                ys, xs = np.nonzero(mask)
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                dets.append(PixelDetection(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), 0, 0.9))
            out.append(dets)
        return out


@pytest.fixture
def predictor():
    return BrightSquarePredictor()
