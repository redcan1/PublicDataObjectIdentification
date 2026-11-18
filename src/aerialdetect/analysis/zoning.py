"""Tag parking lots with the land use around them, from a zoning (or land-use) polygon layer.

The point is to compare like with like: office lots emptied during COVID while residential
lots may have filled up, so lots are grouped by the zone their centre falls in.
Zones can come from a local vector file or an ArcGIS FeatureServer layer (most US cities
publish zoning this way).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)


def load_arcgis_layer(url: str, aoi: gpd.GeoDataFrame, out_fields: str = "*", page: int = 1000,
                      timeout: int = 60) -> gpd.GeoDataFrame:
    """Polygons from an ArcGIS FeatureServer/MapServer layer that intersect the AOI's bounding box."""
    import requests

    minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
    feats, offset = [], 0
    while True:
        r = requests.get(url.rstrip("/") + "/query", timeout=timeout, params={
            "where": "1=1", "geometry": f"{minx},{miny},{maxx},{maxy}", "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": out_fields,
            "outSR": 4326, "f": "geojson", "resultOffset": offset, "resultRecordCount": page,
        })
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"ArcGIS query failed: {data['error']}")
        batch = data.get("features", [])
        feats += batch
        if len(batch) < page and not data.get("properties", {}).get("exceededTransferLimit"):
            break
        offset += len(batch)
        if not batch:
            break
    log.info("zoning: %d polygon(s) from %s", len(feats), url)
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def load_zones(cfg: dict[str, Any], aoi: gpd.GeoDataFrame, cache_dir: str | Path | None = None) -> gpd.GeoDataFrame:
    """Zones from ``cfg['path']`` (vector file) or ``cfg['url']`` (ArcGIS layer, cached as GeoPackage)."""
    if cfg.get("path"):
        return gpd.read_file(cfg["path"])
    import hashlib

    key = hashlib.sha1(f"{cfg['url']}|{aoi.to_crs(4326).total_bounds.round(5).tolist()}".encode()).hexdigest()[:10]
    cache = Path(cache_dir or "data/cache") / f"zones_{key}.gpkg"
    if cache.exists():
        return gpd.read_file(cache)
    zones = load_arcgis_layer(cfg["url"], aoi, out_fields=cfg.get("field", "*"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    zones.to_file(cache, driver="GPKG")
    return zones


def classify_lots(lots: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, field: str,
                  groups: dict[str, list[str]], default: str = "other") -> gpd.GeoDataFrame:
    """Add ``zone`` (raw code at the lot's centre) and ``lot_type`` (its group from ``groups``)."""
    lots = lots.copy()
    if lots.empty:
        lots["zone"], lots["lot_type"] = pd.Series(dtype=str), pd.Series(dtype=str)
        return lots
    crs = lots.estimate_utm_crs()
    centres = gpd.GeoDataFrame(geometry=lots.to_crs(crs).representative_point(), index=lots.index, crs=crs)
    z = zones[[field, "geometry"]].to_crs(crs)
    hit = gpd.sjoin(centres, z, predicate="within", how="left")
    hit = hit[~hit.index.duplicated()]  # overlapping zone polygons: take the first
    lots["zone"] = hit[field].reindex(lots.index)
    code_to_group = {code: g for g, codes in groups.items() for code in codes}
    lots["lot_type"] = lots["zone"].map(code_to_group).fillna(default)
    log.info("lot types: %s", lots["lot_type"].value_counts().to_dict())
    return lots
