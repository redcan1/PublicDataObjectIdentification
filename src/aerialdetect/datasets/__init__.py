"""Converters from public aerial datasets to Ultralytics YOLO format.

* xView  -> axis-aligned boxes, 0.3 m WorldView-3: ``xview.convert_xview`` (this package).
* DOTA   -> oriented boxes, mixed GSD: use Ultralytics' built-in ``DOTAv1.yaml`` (auto-downloads
            and converts) with an ``-obb`` model; no converter needed here.
"""
