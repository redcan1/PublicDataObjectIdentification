import json

import numpy as np
import pytest
import rasterio
import yaml

from aerialdetect.datasets.xview import clip_boxes_to_tile, convert_xview

cv2 = pytest.importorskip("cv2")


def test_clip_boxes_drops_mostly_hidden():
    boxes = np.array([[10, 10, 30, 30], [90, 10, 130, 30]], float)  # second is 25% visible in a 100px tile
    lines = clip_boxes_to_tile(boxes, np.array([0, 1]), 0, 0, 100, 100, min_visible=0.4)
    assert len(lines) == 1 and lines[0].startswith("0 0.200000 0.200000 0.200000 0.200000")


def test_convert_xview_end_to_end(tmp_path):
    images = tmp_path / "train_images"
    images.mkdir()
    for name in ("1.tif", "2.tif"):
        with rasterio.open(images / name, "w", driver="GTiff", width=200, height=200, count=3, dtype="uint8") as dst:
            dst.write(np.full((3, 200, 200), 100, np.uint8))
    feats = [
        {"properties": {"image_id": "1.tif", "type_id": 18, "bounds_imcoords": "20,20,40,30"}},
        {"properties": {"image_id": "1.tif", "type_id": 73, "bounds_imcoords": "100,100,150,160"}},
        {"properties": {"image_id": "2.tif", "type_id": 99, "bounds_imcoords": "0,0,10,10"}},  # unmapped
    ]
    geojson = tmp_path / "xView_train.geojson"
    geojson.write_text(json.dumps({"features": feats}))

    data_yaml = convert_xview(geojson, images, tmp_path / "out", {"car": [18], "building": [73]},
                              chip_gsd=0.3, tile_px=200, overlap_px=0, val_fraction=0.5, background_fraction=0)
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["names"] == {0: "car", 1: "building"}
    labels = list((tmp_path / "out" / "labels").rglob("1_0_0.txt"))
    assert len(labels) == 1
    classes = sorted(int(l.split()[0]) for l in labels[0].read_text().splitlines())
    assert classes == [0, 1]
