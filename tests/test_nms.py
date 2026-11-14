import geopandas as gpd
from shapely.geometry import box

from aerialdetect.nms import polygon_nms


def _gdf(rows):
    return gpd.GeoDataFrame(
        [{"class_id": c, "confidence": s, "geometry": g} for c, s, g in rows], geometry="geometry", crs=32612
    )


def test_overlapping_same_class_keeps_highest_confidence():
    out = polygon_nms(_gdf([(0, 0.5, box(0, 0, 10, 10)), (0, 0.9, box(1, 1, 11, 11))]))
    assert len(out) == 1 and out["confidence"].iloc[0] == 0.9


def test_different_classes_not_suppressed():
    out = polygon_nms(_gdf([(0, 0.5, box(0, 0, 10, 10)), (1, 0.9, box(0, 0, 10, 10))]))
    assert len(out) == 2


def test_partial_duplicate_removed_by_intersection_over_smaller():
    # Partial box (clipped at a seam) lies inside the full one: IoU 0.3, IoS 1.0.
    out = polygon_nms(_gdf([(0, 0.9, box(0, 0, 10, 10)), (0, 0.8, box(0, 0, 3, 10))]))
    assert len(out) == 1


def test_adjacent_objects_kept():
    out = polygon_nms(_gdf([(0, 0.9, box(0, 0, 4, 2)), (0, 0.8, box(4.2, 0, 8.2, 2))]))
    assert len(out) == 2
