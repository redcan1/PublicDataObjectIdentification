from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd


class RasterSource(ABC):
    """Something that can produce georeferenced rasters covering an AOI."""

    name: str = "base"

    @abstractmethod
    def fetch(self, aoi: gpd.GeoDataFrame | None, out_dir: Path) -> list[Path]:
        """Return paths to local GeoTIFFs (with a CRS) covering ``aoi``.

        ``aoi`` is in EPSG:4326 and may be ``None`` for sources that already know
        their extent (e.g. a local orthomosaic).
        """
