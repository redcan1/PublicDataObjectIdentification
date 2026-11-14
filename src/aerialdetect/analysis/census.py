"""Census tract boundaries from the U.S. Census Bureau cartographic boundary files."""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

CB_BASE = "https://www2.census.gov/geo/tiger/GENZ{year}/shp/"


def _download(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / url.rsplit("/", 1)[-1]
    if dest.exists():
        return dest
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "aerialdetect/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    return dest


def load_tracts(aoi: gpd.GeoDataFrame, year: int = 2023, cache_dir: str | Path = "data/cache") -> gpd.GeoDataFrame:
    """Census tracts (1:500k cartographic boundaries) intersecting the AOI.

    Handles AOIs spanning state lines. Key column: ``GEOID`` (11-digit tract id), which
    joins to ACS / decennial tables (population, households, vehicles available...).
    """
    cache = Path(cache_dir) / "census"
    base = CB_BASE.format(year=year)
    states = gpd.read_file(_download(f"{base}cb_{year}_us_state_500k.zip", cache))
    aoi_geom = aoi.to_crs(states.crs).union_all()
    fips = sorted(states.loc[states.intersects(aoi_geom), "STATEFP"])
    if not fips:
        raise ValueError("AOI does not intersect any U.S. state")
    parts = [gpd.read_file(_download(f"{base}cb_{year}_{fp}_tract_500k.zip", cache)) for fp in fips]
    tracts = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=parts[0].crs)
    tracts = tracts[tracts.intersects(aoi.to_crs(tracts.crs).union_all())]
    log.info("%d tract(s) intersect the AOI", len(tracts))
    return tracts.reset_index(drop=True)
