"""Load pipeline results into a file geodatabase and run example analyses in ArcGIS Pro.

Run with ArcGIS Pro's Python (arcpy), e.g. from the Pro Python Command Prompt:

    propy arcgis/import_to_gdb.py outputs/tempe_naip/results.gpkg outputs/tempe_naip/results.gdb

Or paste into the Python window inside Pro. You do not need this script to view the results:
Pro opens GeoPackages directly (Catalog -> Add Folder Connection -> results.gpkg).
"""

import os
import sys

import arcpy

LAYERS = ["aoi", "imagery_footprints", "detections", "detection_points", "tract_density", "parking_lots"]


def main(gpkg: str, gdb: str) -> None:
    arcpy.env.overwriteOutput = True
    folder, name = os.path.split(os.path.abspath(gdb))
    if not arcpy.Exists(gdb):
        arcpy.management.CreateFileGDB(folder, name)

    for layer in LAYERS:
        src = os.path.join(gpkg, f"main.{layer}")
        if arcpy.Exists(src):
            arcpy.conversion.ExportFeatures(src, os.path.join(gdb, layer))
            print(f"imported {layer}")

    pts = os.path.join(gdb, "detection_points")
    if not arcpy.Exists(pts):
        return

    # Example 1: hot spots of car detections on a 100 m hexagon grid (Getis-Ord Gi*).
    cars = arcpy.management.MakeFeatureLayer(pts, "cars", "class_name = 'car'")
    arcpy.stats.OptimizedHotSpotAnalysis(
        cars, os.path.join(gdb, "car_hotspots"),
        Aggregation_Method="COUNT_INCIDENTS_WITHIN_HEXAGON_GRID", Cell_Size="100 Meters",
    )
    print("wrote car_hotspots")

    # Example 2: Summarize Within tracts by class (the Python pipeline also does this in tract_density).
    tracts = os.path.join(gdb, "tract_density")
    if arcpy.Exists(tracts):
        arcpy.analysis.SummarizeWithin(
            tracts, pts, os.path.join(gdb, "tract_summary"),
            keep_all_polygons="KEEP_ALL", group_field="class_name",
            out_group_table=os.path.join(gdb, "tract_summary_by_class"),
        )
        print("wrote tract_summary")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: import_to_gdb.py <results.gpkg> <output.gdb>")
    main(sys.argv[1], sys.argv[2])
