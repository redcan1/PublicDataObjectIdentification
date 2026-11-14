"""Parking-lot occupancy: vehicles detected inside lot polygons vs. lot capacity."""

from __future__ import annotations

import logging
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Gross area per stall including drive aisles; 28-35 m2 is typical for surface lots.
DEFAULT_M2_PER_STALL = 30.0


def load_osm_parking(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Surface/multi-storey parking polygons from OpenStreetMap (``amenity=parking``)."""
    import osmnx as ox

    try:
        g = ox.features_from_polygon(aoi.to_crs("EPSG:4326").union_all(), tags={"amenity": "parking"})
    except Exception as e:  # osmnx raises InsufficientResponseError when nothing is found
        if type(e).__name__ == "InsufficientResponseError":
            log.warning("no OSM parking polygons in the AOI")
            return gpd.GeoDataFrame(columns=["lot_id", "geometry"], geometry="geometry", crs="EPSG:4326")
        raise
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()
    id_col = next((c for c in ("id", "osmid") if c in g.columns), None)
    keep = [c for c in ("name", "capacity", "parking", "access", "operator") if c in g.columns]
    out = g[keep + ["geometry"]].copy()
    out.insert(0, "osm_id", g[id_col].astype(str) if id_col else "")
    if "parking" in out.columns:
        out = out[~out["parking"].isin(["underground"])]  # invisible from the air
    out.insert(0, "lot_id", range(len(out)))
    return gpd.GeoDataFrame(out, geometry="geometry", crs=g.crs).reset_index(drop=True)


def parking_occupancy(
    detections: gpd.GeoDataFrame,
    lots: gpd.GeoDataFrame,
    vehicle_classes: Sequence[str],
    aoi: gpd.GeoDataFrame | None = None,
    m2_per_stall: float = DEFAULT_M2_PER_STALL,
    lot_id: str = "lot_id",
) -> gpd.GeoDataFrame:
    """Vehicles per lot, capacity and occupancy.

    Capacity comes from the lot's ``capacity`` attribute when present (OSM tag),
    otherwise ``area / m2_per_stall``. Occupancy is ``n_vehicles / capacity`` and can
    exceed 1 when the estimate is low or cars park in aisles. ``aoi_frac`` < 1 means
    part of the lot was outside the imagery, so its count is incomplete.
    """
    if lots.empty:
        return lots
    lots = lots.copy()
    if lot_id not in lots.columns:
        lots[lot_id] = range(len(lots))
    crs = lots.estimate_utm_crs()
    l = lots.to_crs(crs)
    l["area_m2"] = l.area.round(1)

    counts = {}
    veh = detections[detections["class_name"].isin(vehicle_classes)] if not detections.empty else detections
    if not veh.empty:
        pts = gpd.GeoDataFrame(geometry=veh.to_crs(crs).centroid, crs=crs)
        joined = gpd.sjoin(pts, l[[lot_id, "geometry"]], predicate="within", how="inner")
        joined = joined[~joined.index.duplicated()]  # overlapping lot polygons: count once
        counts = joined.groupby(lot_id).size()
    l["n_vehicles"] = l[lot_id].map(counts).fillna(0).astype(int)

    osm_cap = pd.to_numeric(l["capacity"], errors="coerce") if "capacity" in l.columns else pd.Series(np.nan, index=l.index)
    has_cap = osm_cap > 0
    l["capacity_est"] = np.where(has_cap, osm_cap, (l["area_m2"] / m2_per_stall).round()).clip(min=1).astype(int)
    l["capacity_src"] = np.where(has_cap, "attribute", "area")
    l["occupancy"] = (l["n_vehicles"] / l["capacity_est"]).round(3)

    if aoi is not None:
        aoi_geom = aoi.to_crs(crs).union_all()
        l["aoi_frac"] = (l.intersection(aoi_geom).area / l.area.where(l.area > 0)).round(3)
    return l.reset_index(drop=True)
