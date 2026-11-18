"""Cut a large raster into model-sized tiles at a fixed ground sample distance (GSD).

The key idea that makes the pipeline sensor-agnostic: the model is trained on chips
at one GSD (e.g. 0.3 m/pixel), so at inference every source is resampled to that
same GSD on the fly. NAIP at 0.6 m gets upsampled 2x; a drone orthomosaic at 3 cm
gets downsampled 10x. A car is then roughly the same number of pixels no matter
where the imagery came from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


@dataclass(frozen=True)
class Tile:
    """A read window in native raster pixels plus the size it is resampled to."""

    window: Window
    out_width: int
    out_height: int
    # True when that edge is shared with a neighbouring tile (False = raster boundary).
    interior_left: bool
    interior_top: bool
    interior_right: bool
    interior_bottom: bool

    def transform(self, raster_transform: Affine) -> Affine:
        """Affine mapping *resampled tile* pixel coords -> raster CRS coords."""
        return window_transform(self.window, raster_transform) * Affine.scale(
            self.window.width / self.out_width, self.window.height / self.out_height
        )


def axis_starts(length: int, win: int, step: int) -> list[int]:
    """Start offsets along one axis so windows of size ``win`` cover ``[0, length)``."""
    if length <= win:
        return [0]
    starts = list(range(0, length - win + 1, step))
    if starts[-1] + win < length:
        starts.append(length - win)  # final window flush with the edge
    return starts


def plan_tiles(
    width: int,
    height: int,
    native_gsd: float,
    model_gsd: float,
    tile_px: int = 640,
    overlap_px: int = 128,
) -> list[Tile]:
    """Plan overlapping tiles. ``tile_px`` / ``overlap_px`` are in *model* pixels."""
    if not 0 <= overlap_px < tile_px:
        raise ValueError("overlap_px must be >= 0 and smaller than tile_px")
    scale = model_gsd / native_gsd  # native pixels per model pixel
    win = max(1, round(tile_px * scale))
    step = max(1, round((tile_px - overlap_px) * scale))

    tiles = []
    for row in axis_starts(height, win, step):
        for col in axis_starts(width, win, step):
            w, h = min(win, width - col), min(win, height - row)
            tiles.append(
                Tile(
                    window=Window(col, row, w, h),
                    out_width=max(1, round(w / scale)),
                    out_height=max(1, round(h / scale)),
                    interior_left=col > 0,
                    interior_top=row > 0,
                    interior_right=col + w < width,
                    interior_bottom=row + h < height,
                )
            )
    return tiles


def compute_stretch(
    src: rasterio.io.DatasetReader, bands: tuple[int, ...], pct: tuple[float, float] = (2, 98)
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-band percentile stretch for non-8-bit imagery, computed once per raster.

    Returns ``None`` for uint8 imagery (used as-is). Computing the stretch globally
    rather than per tile keeps brightness consistent across tiles.
    """
    if all(src.dtypes[b - 1] == "uint8" for b in bands):
        return None
    factor = max(src.width, src.height) / 1024
    out_shape = None
    if factor > 1:
        out_shape = (len(bands), max(1, int(src.height / factor)), max(1, int(src.width / factor)))
    arr = src.read(list(bands), out_shape=out_shape, masked=True).astype("float64")
    lo, hi = [], []
    for band in arr:
        vals = band.compressed()
        if vals.size == 0:
            lo.append(0.0)
            hi.append(1.0)
        else:
            lo.append(float(np.percentile(vals, pct[0])))
            hi.append(float(np.percentile(vals, pct[1])))
    return np.asarray(lo, dtype="float32"), np.asarray(hi, dtype="float32")


def _band_histograms(src: rasterio.io.DatasetReader, bands: tuple[int, ...], size: int = 2048) -> np.ndarray:
    """(n_bands, 256) pixel-value histograms of a uint8 raster (decimated read, nodata/black excluded)."""
    factor = max(src.width, src.height) / size
    out_shape = None
    if factor > 1:
        out_shape = (len(bands), max(1, int(src.height / factor)), max(1, int(src.width / factor)))
    arr = src.read(list(bands), out_shape=out_shape)
    valid = arr.max(axis=0) > 0  # NAIP pads outside the mosaic with 0 in every band
    return np.stack([np.bincount(b[valid].ravel(), minlength=256)[:256] for b in arr]).astype("float64")


def compute_lut(
    src: rasterio.io.DatasetReader,
    bands: tuple[int, ...],
    method: str,
    reference: str | None = None,
    pct: tuple[float, float] = (1, 99),
) -> np.ndarray:
    """Per-band 256-entry lookup table that normalises a uint8 raster's radiometry.

    ``stretch``: linear stretch of each band's ``pct`` percentiles to 0..255 (per raster, no reference).
    ``match``: histogram matching of each band to the same band of ``reference``, so every year's
    imagery has the reference year's brightness/contrast/colour balance.
    """
    if not all(src.dtypes[b - 1] == "uint8" for b in bands):
        raise ValueError("normalize currently supports uint8 imagery only")
    h = _band_histograms(src, bands)
    cdf = np.cumsum(h, axis=1) / np.maximum(h.sum(axis=1, keepdims=True), 1)
    v = np.arange(256, dtype="float64")
    if method == "stretch":
        lut = []
        for c in cdf:
            lo = float(np.searchsorted(c, pct[0] / 100))
            hi = float(np.searchsorted(c, pct[1] / 100))
            lut.append(np.clip((v - lo) / max(hi - lo, 1.0) * 255.0, 0, 255))
        return np.rint(np.stack(lut)).astype("uint8")
    if method == "match":
        if not reference:
            raise ValueError("normalize.method=match needs normalize.reference (a raster path)")
        with rasterio.open(reference) as ref:
            hr = _band_histograms(ref, bands)
        cr = np.cumsum(hr, axis=1) / np.maximum(hr.sum(axis=1, keepdims=True), 1)
        return np.rint(np.stack([np.interp(c, r, v) for c, r in zip(cdf, cr)])).astype("uint8")
    raise ValueError(f"unknown normalize method: {method!r}")


def read_tile(
    src: rasterio.io.DatasetReader,
    tile: Tile,
    bands: tuple[int, ...] = (1, 2, 3),
    stretch: tuple[np.ndarray, np.ndarray] | None = None,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    """Read a tile resampled to the model GSD. Returns an RGB ``(H, W, 3)`` uint8 array."""
    downsampling = tile.window.width > tile.out_width
    arr = src.read(
        list(bands),
        window=tile.window,
        out_shape=(len(bands), tile.out_height, tile.out_width),
        # Averaging avoids aliasing when shrinking drone imagery; bilinear when enlarging.
        resampling=Resampling.average if downsampling else Resampling.bilinear,
    )
    if stretch is not None:
        lo, hi = stretch
        arr = (arr.astype("float32") - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
        arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    if lut is not None:
        nodata = arr.max(axis=0) == 0
        arr = lut[np.arange(len(arr))[:, None, None], arr]
        arr[:, nodata] = 0  # keep padding black so empty tiles are still skipped
    return np.ascontiguousarray(np.moveaxis(arr, 0, -1))
