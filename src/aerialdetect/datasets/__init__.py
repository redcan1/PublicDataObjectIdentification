"""Converters from public aerial datasets to Ultralytics YOLO format.

* xView  -> axis-aligned boxes, 0.3 m WorldView-3: ``xview.convert_xview`` (this package).
* DOTA   -> oriented boxes, mixed GSD: ``dota.prep_dota`` downloads DOTA v1 and cuts it into
            native-resolution chips (train with an ``-obb`` model at ``imgsz == crop_px``).
"""
