"""Non-maximum suppression on georeferenced polygons.

Tiles overlap, so an object near a tile seam is detected more than once. YOLO's
own NMS only works inside one tile; this pass removes duplicates across tiles
(and across overlapping rasters) in map coordinates.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from shapely import STRtree


def polygon_nms(gdf: gpd.GeoDataFrame, iou_thr: float = 0.5, ios_thr: float = 0.8) -> gpd.GeoDataFrame:
    """Greedy per-class NMS.

    A lower-confidence polygon is dropped when its IoU with a kept polygon exceeds
    ``iou_thr`` OR when it is mostly contained in it (intersection over the smaller
    area > ``ios_thr``), which catches partial duplicates clipped at tile edges.
    """
    if gdf.empty:
        return gdf
    keep: list = []
    for _, grp in gdf.groupby("class_id", sort=False):
        grp = grp.sort_values("confidence", ascending=False)
        geoms = np.asarray(grp.geometry.values, dtype=object)
        areas = np.maximum(shapely.area(geoms), 1e-12)
        tree = STRtree(geoms)
        suppressed = np.zeros(len(grp), dtype=bool)
        for i in range(len(grp)):
            if suppressed[i]:
                continue
            keep.append(grp.index[i])
            cand = tree.query(geoms[i])
            cand = cand[(cand > i) & ~suppressed[cand]]
            if cand.size == 0:
                continue
            inter = shapely.area(shapely.intersection(geoms[i], geoms[cand]))
            iou = inter / (areas[i] + areas[cand] - inter)
            ios = inter / np.minimum(areas[i], areas[cand])
            suppressed[cand[(iou > iou_thr) | (ios > ios_thr)]] = True
    return gdf.loc[keep]
