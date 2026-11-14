"""Count / density / footprint statistics of detections per zone (tracts, block groups, zoning...)."""

from __future__ import annotations

import re
from typing import Sequence

import geopandas as gpd


def slug(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def density_by_zone(
    detections: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    zone_id: str,
    aoi: gpd.GeoDataFrame,
    count_classes: Sequence[str] = (),
    footprint_classes: Sequence[str] = (),
) -> gpd.GeoDataFrame:
    """Per-zone statistics, normalised by the part of each zone the imagery actually covers.

    Columns added:
      * ``zone_km2``, ``covered_km2``, ``coverage`` (fraction of the zone inside the AOI)
      * per count class: ``n_<cls>`` (by centroid) and ``<cls>_per_km2`` (per covered km2)
      * per footprint class: ``<cls>_m2`` (union of polygons, clipped to the zone) and
        ``<cls>_cover_frac`` (share of covered land occupied by that class)

    Zones only partly inside the AOI get densities computed over the covered part;
    filter on ``coverage`` if you only want fully observed zones.
    """
    crs = zones.estimate_utm_crs()
    aoi_geom = aoi.to_crs(crs).union_all()
    z = zones.to_crs(crs)
    z = z[z.intersects(aoi_geom)].copy()
    z["zone_km2"] = z.area / 1e6
    z["covered_km2"] = z.intersection(aoi_geom).area / 1e6
    z["coverage"] = (z["covered_km2"] / z["zone_km2"]).clip(upper=1.0)
    covered = z["covered_km2"].where(z["covered_km2"] > 0)

    d = detections.to_crs(crs) if not detections.empty else detections
    zcols = z[[zone_id, "geometry"]]

    for cls in count_classes:
        s = slug(cls)
        sub = d[d["class_name"] == cls] if not d.empty else d
        counts = {}
        if not sub.empty:
            pts = gpd.GeoDataFrame(geometry=sub.centroid, crs=crs)
            joined = gpd.sjoin(pts, zcols, predicate="within", how="inner")
            joined = joined[~joined.index.duplicated()]  # a point on a shared border counts once
            counts = joined.groupby(zone_id).size()
        z[f"n_{s}"] = z[zone_id].map(counts).fillna(0).astype(int)
        z[f"{s}_per_km2"] = (z[f"n_{s}"] / covered).round(2)

    for cls in footprint_classes:
        s = slug(cls)
        sub = d[d["class_name"] == cls] if not d.empty else d
        area = {}
        if not sub.empty:
            # Union first so overlapping detections do not double count area.
            merged = gpd.GeoDataFrame(geometry=[sub.union_all()], crs=crs).explode(index_parts=False)
            inter = gpd.overlay(merged, zcols, how="intersection", keep_geom_type=True)
            area = inter.assign(_a=inter.area).groupby(zone_id)["_a"].sum()
        z[f"{s}_m2"] = z[zone_id].map(area).fillna(0.0).round(1)
        z[f"{s}_cover_frac"] = (z[f"{s}_m2"] / (covered * 1e6)).round(4)

    return z.reset_index(drop=True)
