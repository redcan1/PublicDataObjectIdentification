"""aerialdetect: object detection on aerial imagery -> GIS features -> spatial analysis.

Pipeline stages (each usable on its own):

1. ``sources``   - get a georeferenced raster (NAIP download, local GeoTIFF, drone orthomosaic)
2. ``datasets``  - convert labelled public datasets (xView) into YOLO training data
3. ``train``     - fine-tune an Ultralytics YOLO model
4. ``detect``    - tile a raster at the model's ground sample distance, run YOLO, georeference boxes
5. ``features``  - clean up detections into map features (GeoPackage for ArcGIS Pro / QGIS / geopandas)
6. ``analysis``  - zonal statistics (e.g. census tracts) and parking-lot occupancy
"""

__version__ = "0.1.0"
