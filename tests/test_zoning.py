import geopandas as gpd
from shapely.geometry import box

from aerialdetect.analysis.zoning import classify_lots


def test_lots_take_group_of_zone_at_their_centre():
    zones = gpd.GeoDataFrame({"BASE_ZONE": ["C-3", "R-3A"]},
                             geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=32610)
    lots = gpd.GeoDataFrame({"lot_id": [0, 1, 2]},
                            geometry=[box(10, 10, 30, 30), box(150, 10, 170, 30), box(500, 500, 510, 510)], crs=32610)
    out = classify_lots(lots, zones, "BASE_ZONE", {"downtown_core": ["C-3"], "residential": ["R-3A"]})
    assert out["lot_type"].tolist() == ["downtown_core", "residential", "other"]
    assert out["zone"].tolist()[:2] == ["C-3", "R-3A"]
