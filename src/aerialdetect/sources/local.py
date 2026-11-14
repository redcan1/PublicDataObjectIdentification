"""Local GeoTIFFs: drone orthomosaics, county orthophotos, anything already on disk.

For the drone: export a georeferenced orthomosaic (e.g. OpenDroneMap's
``odm_orthophoto/odm_orthophoto.tif`` or a Pix4D / DroneDeploy GeoTIFF) and point
this source at it. Its native GSD (often 1-5 cm) is read from the file and tiles are
resampled to the model's GSD automatically, so a model trained on 0.3 m satellite
chips sees cars at the scale it was trained on.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio

from .base import RasterSource


class LocalRasterSource(RasterSource):
    name = "local"

    def __init__(self, paths: list[str | Path]):
        if not paths:
            raise ValueError("local source needs at least one raster path")
        self.paths = [Path(p) for p in paths]

    def fetch(self, aoi: gpd.GeoDataFrame | None, out_dir: Path) -> list[Path]:
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(p)
            with rasterio.open(p) as src:
                if src.crs is None:
                    raise ValueError(f"{p} is not georeferenced (no CRS)")
        return self.paths
