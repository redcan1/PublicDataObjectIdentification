"""NAIP (USDA National Agriculture Imagery Program) via Microsoft Planetary Computer.

NAIP is 4-band (R, G, B, NIR) leaf-on imagery of the continental U.S., refreshed
every 2-3 years per state, at 0.6 m (most years) or 0.3 m (many states since 2022).
The Planetary Computer hosts it as Cloud-Optimized GeoTIFFs and is free to read
anonymously; URLs are signed by the ``planetary-computer`` package.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import shape
from shapely.ops import unary_union

from .base import RasterSource

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAX_MOSAIC_BYTES = 2_000_000_000  # refuse to build huge in-memory mosaics; split the AOI instead


class NaipSource(RasterSource):
    name = "naip"

    def __init__(self, year: int | None = None, min_coverage: float = 0.99):
        self.year = year
        self.min_coverage = min_coverage

        self.metadata: dict = {}

    def fetch(self, aoi: gpd.GeoDataFrame | None, out_dir: Path) -> list[Path]:
        if aoi is None:
            raise ValueError("NAIP source needs an AOI (bbox or vector file)")
        aoi_geom = aoi.to_crs("EPSG:4326").union_all()
        by_year = search_by_year(aoi_geom, self.year)
        if not by_year:
            raise RuntimeError("no NAIP imagery found for this AOI" + (f" in {self.year}" if self.year else ""))

        def coverage(year: int) -> float:
            return _coverage(by_year[year], aoi_geom)

        # Most recent year that fully covers the AOI (AOIs on state lines can mix years).
        year = next((y for y in sorted(by_year, reverse=True) if coverage(y) >= self.min_coverage), None)
        if year is None:
            year = max(by_year, key=coverage)
            log.warning("no NAIP year fully covers the AOI; using %d (%.0f%% coverage)", year, 100 * coverage(year))
        log.info("NAIP %d: %d item(s)", year, len(by_year[year]))

        key = hashlib.sha1(f"{aoi_geom.wkt}|{year}".encode()).hexdigest()[:10]
        out = Path(out_dir) / f"naip_{year}_{key}.tif"
        # Record when the imagery was actually flown: counts only mean something tied to a date.
        dates = sorted({i.datetime.date().isoformat() for i in by_year[year]
                        if shape(i.geometry).intersects(aoi_geom)})
        self.metadata = {"naip_year": year, "acquisition_dates": dates,
                         "naip_gsd_m": sorted({i.properties.get("gsd") for i in by_year[year]} - {None}),
                         "coverage": round(coverage(year), 4)}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        if out.exists():
            log.info("using cached %s", out)
            return [out]
        mosaic_to_aoi([i.assets["image"].href for i in by_year[year]], aoi_geom, out)
        return [out]


def search_by_year(aoi_geom_4326, year: int | None = None, start: int | None = None,
                   end: int | None = None) -> dict[int, list]:
    """NAIP STAC items intersecting the AOI, grouped by year."""
    import planetary_computer
    import pystac_client

    if year:
        start = end = year
    dt = f"{start or 2003}-01-01/{end or 2100}-12-31" if (start or end) else None
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(collections=["naip"], intersects=aoi_geom_4326.__geo_interface__, datetime=dt).items()
    by_year: dict[int, list] = defaultdict(list)
    for it in items:
        by_year[it.datetime.year].append(it)
    return dict(by_year)


def _coverage(items: list, aoi_geom) -> float:
    cover = unary_union([shape(i.geometry) for i in items])
    return cover.intersection(aoi_geom).area / aoi_geom.area


def survey(aoi: gpd.GeoDataFrame, start: int | None = None, end: int | None = None) -> list[dict]:
    """Every NAIP flight date over the AOI: date, weekday, GSD and the share of the AOI it covers.

    NAIP flies each state on a few days per cycle, so before comparing years check that the
    dates are comparable (weekday vs weekend, season, resolution).
    """
    aoi_geom = aoi.to_crs("EPSG:4326").union_all()
    rows = []
    for year, items in sorted(search_by_year(aoi_geom, start=start, end=end).items()):
        by_date: dict = defaultdict(list)
        for it in items:
            by_date[it.datetime.date()].append(it)
        for d, its in sorted(by_date.items()):
            rows.append({"year": year, "date": d.isoformat(), "weekday": d.strftime("%a"),
                         "gsd_m": sorted({i.properties.get("gsd") for i in its} - {None}),
                         "coverage": round(_coverage(its, aoi_geom), 3)})
    return rows


def mosaic_to_aoi(hrefs: list[str], aoi_geom_4326, out_path: Path) -> Path:
    """Mosaic remote COGs, cropped to the AOI's bounding box, into one local GeoTIFF."""
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3",
               GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE")
    with rasterio.Env(**env):
        srcs = [rasterio.open(h) for h in hrefs]
        vrts = []
        try:
            dst_crs = srcs[0].crs
            res = srcs[0].res
            datasets = []
            for s in srcs:
                if s.crs == dst_crs:
                    datasets.append(s)
                else:  # NAIP is in UTM; AOIs crossing a zone boundary need reprojection
                    vrt = WarpedVRT(s, crs=dst_crs, resampling=Resampling.bilinear)
                    vrts.append(vrt)
                    datasets.append(vrt)
            bounds = transform_bounds("EPSG:4326", dst_crs, *aoi_geom_4326.bounds, densify_pts=21)
            est_bytes = ((bounds[2] - bounds[0]) / res[0]) * ((bounds[3] - bounds[1]) / res[1]) \
                * srcs[0].count * np.dtype(srcs[0].dtypes[0]).itemsize
            if est_bytes > MAX_MOSAIC_BYTES:
                raise ValueError(f"AOI mosaic would be ~{est_bytes / 1e9:.1f} GB; use a smaller AOI or split it")

            arr, transform = merge(datasets, bounds=bounds, res=res, nodata=0, resampling=Resampling.bilinear)
            profile = dict(
                driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                dtype=arr.dtype, crs=dst_crs, transform=transform, nodata=0,
                tiled=True, blockxsize=512, blockysize=512, compress="deflate", BIGTIFF="IF_SAFER",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_name(out_path.stem + ".partial.tif")
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(arr)
            tmp.replace(out_path)
        finally:
            for v in vrts:
                v.close()
            for s in srcs:
                s.close()
    log.info("wrote %s", out_path)
    return out_path
