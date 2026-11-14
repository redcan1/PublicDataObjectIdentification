import pytest

from aerialdetect.detect import detect_raster

from .conftest import ORIGIN, make_raster


def _expected_bounds(col, row, size, gsd):
    x0, y0 = ORIGIN
    return (x0 + col * gsd, y0 - (row + size) * gsd, x0 + (col + size) * gsd, y0 - row * gsd)


def test_naip_like_upsampling_georef_and_dedup(tmp_path, predictor):
    # 0.6 m raster, model at 0.3 m -> tiles of 320 native px, step 256.
    # Square 1 straddles a tile seam; square 2 sits fully inside two overlapping tiles.
    squares = [(250, 250, 34), (700, 700, 34)]
    path = make_raster(tmp_path / "naip.tif", 1000, 0.6, squares)
    gdf = detect_raster(path, predictor, predictor.names, model_gsd=0.3)

    assert len(gdf) == 2, "seam partials should be dropped and duplicates suppressed"
    assert gdf.crs.to_epsg() == 32612
    got = sorted(tuple(g.bounds) for g in gdf.geometry)
    for (col, row, s), b in zip(squares, got):
        assert b == pytest.approx(_expected_bounds(col, row, s, 0.6), abs=0.9)
    assert set(gdf["class_name"]) == {"square"}
    assert gdf["native_gsd"].iloc[0] == pytest.approx(0.6)


def test_drone_like_downsampling(tmp_path, predictor):
    # 5 cm raster; a 20 m square is 400 native px but ~67 px at the model's 0.3 m.
    path = make_raster(tmp_path / "drone.tif", 3000, 0.05, [(1000, 1200, 400)])
    gdf = detect_raster(path, predictor, predictor.names, model_gsd=0.3)
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].bounds == pytest.approx(_expected_bounds(1000, 1200, 400, 0.05), abs=0.35)


def test_empty_raster_returns_empty_frame(tmp_path, predictor):
    path = make_raster(tmp_path / "empty.tif", 300, 0.3)
    gdf = detect_raster(path, predictor, predictor.names, model_gsd=0.3)
    assert gdf.empty and gdf.crs.to_epsg() == 32612
