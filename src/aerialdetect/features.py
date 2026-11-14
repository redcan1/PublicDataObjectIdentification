"""Turn raw detections into clean GIS features and write them out.

GeoPackage is the default output: one file, many layers, opens directly in
ArcGIS Pro (Add Data -> results.gpkg), QGIS, and geopandas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import pandas as pd

from .nms import polygon_nms

log = logging.getLogger(__name__)


def merge_detections(parts: list[gpd.GeoDataFrame], nms_iou: float = 0.5, nms_ios: float = 0.8) -> gpd.GeoDataFrame:
    """Combine detections from several rasters into the first raster's CRS.

    Runs NMS again so overlapping rasters (adjacent NAIP tiles, overlapping drone
    flights) do not double count.
    """
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("no detection results to merge")
    crs = parts[0].crs
    nonempty = [p.to_crs(crs) for p in parts if not p.empty]
    if not nonempty:
        return parts[0]
    merged = gpd.GeoDataFrame(pd.concat(nonempty, ignore_index=True), geometry="geometry", crs=crs)
    if len(nonempty) > 1:
        merged = polygon_nms(merged, nms_iou, nms_ios).reset_index(drop=True)
    return merged


def clip_to_aoi(dets: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep detections whose centroid falls inside the AOI."""
    if dets.empty:
        return dets
    geom = aoi.to_crs(dets.crs).union_all()
    return dets[dets.centroid.within(geom)].reset_index(drop=True)


def add_measurements(dets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``area_m2`` (computed in a local UTM zone). Useful for filtering by size."""
    dets = dets.copy()
    if dets.empty:
        dets["area_m2"] = pd.Series(dtype="float64")
        return dets
    dets["area_m2"] = dets.to_crs(dets.estimate_utm_crs()).area.round(2)
    return dets


def to_points(dets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centroid version of the detections (heat maps, kernel density, hot-spot analysis)."""
    return dets.assign(geometry=dets.centroid) if not dets.empty else dets.copy()


def write_gpkg(path: str | Path, layers: Mapping[str, gpd.GeoDataFrame | None]) -> Path:
    """Write layers to a fresh GeoPackage (any existing file is replaced). Empty layers are skipped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for name, gdf in layers.items():
        if gdf is None or gdf.empty:
            log.warning("layer %r is empty; not written", name)
            continue
        gdf.to_file(path, layer=name, driver="GPKG")
    return path
