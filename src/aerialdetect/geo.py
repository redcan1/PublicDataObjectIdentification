"""Small geospatial helpers shared across the pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

WGS84 = "EPSG:4326"
_M_PER_DEG_LAT = 111_320.0


def load_aoi(bbox: Sequence[float] | None = None, path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Return the area of interest as a one-row GeoDataFrame in EPSG:4326.

    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)``; ``path`` is any vector file
    (GeoJSON, shapefile, GeoPackage...). All features in the file are unioned.
    """
    if path is not None:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"AOI file {path} has no CRS")
        geom = gdf.to_crs(WGS84).union_all()
    elif bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        minx, miny, maxx, maxy = map(float, bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError("bbox must be ordered [min_lon, min_lat, max_lon, max_lat]")
        geom = box(minx, miny, maxx, maxy)
    else:
        raise ValueError("Provide an AOI as a bbox or a vector file path")
    return gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[geom], crs=WGS84)


def raster_gsd(src: rasterio.io.DatasetReader) -> float:
    """Ground sample distance (metres per pixel) of an open raster.

    Handles both projected CRSs (any linear unit) and geographic CRSs (degrees,
    approximated at the raster's centre latitude).
    """
    if src.crs is None:
        raise ValueError(f"{src.name} has no CRS; detections could not be placed on a map")
    xres, yres = src.res
    if src.crs.is_geographic:
        lat = math.radians((src.bounds.top + src.bounds.bottom) / 2)
        return (xres * _M_PER_DEG_LAT * math.cos(lat) + yres * _M_PER_DEG_LAT) / 2
    _, to_metres = src.crs.linear_units_factor
    return (xres + yres) / 2 * to_metres


def raster_footprints(paths: Iterable[str | Path]) -> gpd.GeoDataFrame:
    """Bounding-box footprint of each raster, in EPSG:4326."""
    names, geoms = [], []
    for p in paths:
        with rasterio.open(p) as src:
            if src.crs is None:
                raise ValueError(f"{p} has no CRS")
            geoms.append(box(*transform_bounds(src.crs, WGS84, *src.bounds, densify_pts=21)))
        names.append(Path(p).name)
    return gpd.GeoDataFrame({"source": names}, geometry=geoms, crs=WGS84)
