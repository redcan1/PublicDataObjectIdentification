import numpy as np

from aerialdetect.tiling import axis_starts, plan_tiles


def test_axis_starts_cover_and_end_flush():
    assert axis_starts(100, 200, 50) == [0]
    s = axis_starts(1000, 320, 256)
    assert s[0] == 0 and s[-1] + 320 == 1000
    assert all(b - a <= 256 for a, b in zip(s, s[1:]))


def test_plan_tiles_covers_raster_and_scales():
    tiles = plan_tiles(1000, 700, native_gsd=0.6, model_gsd=0.3, tile_px=640, overlap_px=128)
    covered = np.zeros((700, 1000), bool)
    for t in tiles:
        w = t.window
        covered[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width] = True
        assert t.window.width <= 320  # 640 model px at 0.3 m = 320 native px at 0.6 m
        assert abs(t.out_width - 2 * t.window.width) <= 1
    assert covered.all()


def test_edge_flags():
    tiles = plan_tiles(1000, 1000, 1.0, 1.0, tile_px=400, overlap_px=100)
    first, last = tiles[0], tiles[-1]
    assert not first.interior_left and not first.interior_top and first.interior_right
    assert last.interior_left and not last.interior_right and not last.interior_bottom


def test_downsampling_drone():
    tiles = plan_tiles(3000, 3000, native_gsd=0.05, model_gsd=0.3, tile_px=640, overlap_px=128)
    assert len(tiles) == 1
    assert tiles[0].out_width == 500
