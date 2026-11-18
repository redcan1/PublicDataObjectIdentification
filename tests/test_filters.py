import geopandas as gpd
from shapely.geometry import box

from aerialdetect.pipeline import filter_by_size, filter_classes


def _dets():
    rows = [("small vehicle", 14.0), ("plane", 15.0), ("plane", 400.0), ("swimming pool", 80.0)]
    return gpd.GeoDataFrame(
        {"class_name": [c for c, _ in rows], "area_m2": [a for _, a in rows]},
        geometry=[box(i, 0, i + 1, 1) for i in range(len(rows))], crs=32612,
    )


def test_filter_classes_keeps_only_listed():
    out = filter_classes(_dets(), ["small vehicle", "plane"])
    assert sorted(out["class_name"]) == ["plane", "plane", "small vehicle"]


def test_filter_classes_none_keeps_all():
    assert len(filter_classes(_dets(), None)) == 4


def test_min_area_drops_car_sized_planes_only():
    out = filter_by_size(_dets(), {"plane": 100}, {})
    assert sorted(zip(out["class_name"], out["area_m2"])) == [
        ("plane", 400.0), ("small vehicle", 14.0), ("swimming pool", 80.0)]


def test_dotted_override_reaches_nested_and_new_keys(tmp_path):
    from aerialdetect.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("model: {weights: x.pt}\nsource: {type: naip, year: null}\n", encoding="utf-8")
    cfg = load_config(p, [("source.year", 2020), ("output.dir", "outputs/y2020")])
    assert cfg["source"] == {"type": "naip", "year": 2020}
    assert cfg["output"]["dir"] == "outputs/y2020" and cfg["output"]["gpkg"] == "results.gpkg"
