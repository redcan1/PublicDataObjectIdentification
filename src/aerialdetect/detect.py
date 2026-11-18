"""Run a detector over a georeferenced raster and return map-coordinate polygons."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Polygon
from tqdm import tqdm

from .geo import raster_gsd
from .nms import polygon_nms
from .tiling import Tile, compute_lut, compute_stretch, plan_tiles, read_tile

log = logging.getLogger(__name__)

DETECTION_COLUMNS = ["class_id", "class_name", "confidence", "source", "native_gsd"]


@dataclass
class PixelDetection:
    """One detection in tile pixel coordinates."""

    polygon: np.ndarray  # (k, 2) array of x, y pixel coords (4 corners for boxes / OBBs)
    class_id: int
    confidence: float


# A predictor takes a batch of RGB uint8 tiles and returns detections per tile.
# YoloPredictor is the real one; tests use a fake. Anything with this signature
# plus a ``names`` dict (e.g. a different model family) plugs straight in.
Predictor = Callable[[Sequence[np.ndarray]], list[list[PixelDetection]]]


class YoloPredictor:
    """Ultralytics YOLO wrapper. Handles both axis-aligned (detect) and oriented (OBB) models."""

    def __init__(self, weights: str | Path, conf: float = 0.25, iou: float = 0.5,
                 imgsz: int = 640, device: str | int | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        self._kw = dict(conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

    def __call__(self, images: Sequence[np.ndarray]) -> list[list[PixelDetection]]:
        # Ultralytics treats numpy input as BGR (OpenCV convention); our tiles are RGB.
        bgr = [np.ascontiguousarray(im[..., ::-1]) for im in images]
        return [self._parse(r) for r in self.model.predict(bgr, **self._kw)]

    @staticmethod
    def _parse(result) -> list[PixelDetection]:
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb):
            polys = obb.xyxyxyxy.cpu().numpy()  # (n, 4, 2)
            cls, conf = obb.cls.cpu().numpy(), obb.conf.cpu().numpy()
        elif result.boxes is not None and len(result.boxes):
            x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy().T
            polys = np.stack(
                [np.stack(c, axis=-1) for c in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))], axis=1
            )
            cls, conf = result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()
        else:
            return []
        return [PixelDetection(p, int(c), float(s)) for p, c, s in zip(polys, cls, conf)]


def _touches_interior_edge(poly: np.ndarray, tile: Tile, margin: float) -> bool:
    """True if a detection is cut off by a seam shared with a neighbouring tile.

    Such partial objects are dropped: with an overlap larger than the object, the
    neighbouring tile sees it whole. Objects cut by the *raster* boundary are kept.
    """
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    return bool(
        (tile.interior_left and xmin <= margin)
        or (tile.interior_top and ymin <= margin)
        or (tile.interior_right and xmax >= tile.out_width - margin)
        or (tile.interior_bottom and ymax >= tile.out_height - margin)
    )


def detect_raster(
    path: str | Path,
    predictor: Predictor,
    class_names: Mapping[int, str],
    model_gsd: float,
    tile_px: int = 640,
    overlap_px: int = 128,
    bands: Sequence[int] = (1, 2, 3),
    batch_size: int = 8,
    edge_margin_px: float = 2.0,
    nms_iou: float = 0.5,
    nms_ios: float = 0.8,
    min_valid_fraction: float = 0.05,
    normalize: Mapping[str, object] | None = None,
) -> gpd.GeoDataFrame:
    """Detect objects in one raster. Output polygons are in the raster's CRS."""
    path = Path(path)
    bands = tuple(int(b) for b in bands)
    rows = []
    with rasterio.open(path) as src:
        native_gsd = raster_gsd(src)
        crs = src.crs
        tiles = plan_tiles(src.width, src.height, native_gsd, model_gsd, tile_px, overlap_px)
        stretch = compute_stretch(src, bands)
        lut = None
        if normalize and normalize.get("method"):
            lut = compute_lut(src, bands, str(normalize["method"]), normalize.get("reference"))
            log.info("%s: normalize=%s (reference=%s)", path.name, normalize["method"], normalize.get("reference"))
        log.info("%s: %dx%d px @ %.3f m -> %d tiles @ %.3f m",
                 path.name, src.width, src.height, native_gsd, len(tiles), model_gsd)

        for start in tqdm(range(0, len(tiles), batch_size), desc=path.name, unit="batch"):
            images, kept = [], []
            for tile in tiles[start:start + batch_size]:
                img = read_tile(src, tile, bands, stretch, lut)
                if (img.max(axis=2) > 0).mean() < min_valid_fraction:
                    continue  # nodata / outside the mosaic
                images.append(img)
                kept.append(tile)
            if not images:
                continue
            for tile, dets in zip(kept, predictor(images)):
                tf = tile.transform(src.transform)
                for d in dets:
                    poly = np.asarray(d.polygon, dtype="float64")
                    if _touches_interior_edge(poly, tile, edge_margin_px):
                        continue
                    xs, ys = tf * (poly[:, 0], poly[:, 1])
                    rows.append({
                        "class_id": int(d.class_id),
                        "confidence": float(d.confidence),
                        "geometry": Polygon(np.column_stack([xs, ys])),
                    })

    if not rows:
        return gpd.GeoDataFrame(columns=DETECTION_COLUMNS + ["geometry"], geometry="geometry", crs=crs)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf = polygon_nms(gdf, nms_iou, nms_ios)
    gdf["class_name"] = gdf["class_id"].map(lambda c: class_names.get(c, str(c)))
    gdf["source"] = path.name
    gdf["native_gsd"] = round(native_gsd, 4)
    return gdf[DETECTION_COLUMNS + ["geometry"]].reset_index(drop=True)
