import geopandas as gpd
import pytest
from shapely.geometry import box

from aerialdetect.analysis.parking import parking_occupancy
from aerialdetect.analysis.zones import density_by_zone

X, Y = 500_000.0, 3_700_000.0
CRS = "EPSG:32612"


def _dets(rows):
    return gpd.GeoDataFrame(
        [{"class_name": c, "class_id": 0, "confidence": 0.9, "geometry": g} for c, g in rows],
        geometry="geometry", crs=CRS,
    )


def test_density_by_zone_normalises_by_covered_area():
    zones = gpd.GeoDataFrame({"GEOID": ["A", "B"]},
                             geometry=[box(X, Y, X + 1000, Y + 1000), box(X + 1000, Y, X + 2000, Y + 1000)], crs=CRS)
    aoi = gpd.GeoDataFrame(geometry=[box(X, Y, X + 1500, Y + 1000)], crs=CRS)  # all of A, half of B
    bldg = lambda x, y: box(X + x, Y + y, X + x + 10, Y + y + 10)  # noqa: E731  100 m2
    dets = _dets([("building", bldg(100, 100)), ("building", bldg(300, 300)), ("building", bldg(500, 500)),
                  ("building", bldg(1200, 500)), ("car", box(X + 50, Y + 50, X + 54, Y + 52))])

    out = density_by_zone(dets, zones, "GEOID", aoi, ["building", "car"], ["building"]).set_index("GEOID")
    assert out.loc["A", "n_building"] == 3 and out.loc["B", "n_building"] == 1
    assert out.loc["B", "coverage"] == pytest.approx(0.5, abs=1e-3)
    assert out.loc["A", "building_per_km2"] == pytest.approx(3.0, rel=1e-3)
    assert out.loc["B", "building_per_km2"] == pytest.approx(2.0, rel=1e-3)
    assert out.loc["A", "building_m2"] == pytest.approx(300.0, rel=1e-3)
    assert out.loc["A", "n_car"] == 1 and out.loc["B", "n_car"] == 0


def test_parking_occupancy_uses_capacity_tag_then_area():
    lots = gpd.GeoDataFrame({"capacity": ["10", None]},
                            geometry=[box(X, Y, X + 50, Y + 50), box(X + 100, Y, X + 160, Y + 50)], crs=CRS)
    car = lambda x, y: box(X + x, Y + y, X + x + 4.5, Y + y + 2)  # noqa: E731
    dets = _dets([("car", car(5, 5)), ("car", car(5, 15)), ("car", car(110, 10)), ("building", car(20, 20))])

    out = parking_occupancy(dets, lots, ["car"], m2_per_stall=30).set_index("lot_id")
    assert out.loc[0, "capacity_est"] == 10 and out.loc[0, "capacity_src"] == "attribute"
    assert out.loc[0, "n_vehicles"] == 2 and out.loc[0, "occupancy"] == pytest.approx(0.2)
    assert out.loc[1, "capacity_est"] == 100 and out.loc[1, "capacity_src"] == "area"  # 3000 m2 / 30
    assert out.loc[1, "n_vehicles"] == 1
