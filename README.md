# aerialdetect: object detection on public aerial imagery → GIS analysis

Fine-tune a YOLO model on free, labelled aerial datasets (xView, DOTA). Run it over any georeferenced
imagery (NAIP today, drone orthomosaics later) and turn the detections into map features. Then analyse
them with geopandas or ArcGIS Pro, for example building density by census tract or parking-lot occupancy.

```
 imagery source ──► tile @ model GSD ──► YOLO ──► georeference + de-duplicate ──► GeoPackage ──► analysis
 (NAIP | drone)      (resample)          (HBB/OBB)  (tile seams, NMS in map space)  (ArcGIS/QGIS)   (tracts, parking)
```

## Why this is drone-ready

Every source ends with a georeferenced GeoTIFF, and every tile is **resampled to the GSD the model was
trained at** (ground sample distance, in metres per pixel). A car is about 15 × 7 px whether it came from 0.6 m NAIP
(upsampled 2×) or a 3 cm drone orthomosaic (downsampled 10×). When drone flights resume, the only change is the
`source:` block in the config. See [configs/drone_example.yaml](configs/drone_example.yaml).

## Install

Python 3.10–3.12. A CUDA GPU is strongly recommended for training. Inference on a few km² runs fine on CPU.

```powershell
python -m venv .venv
.venv\Scripts\activate
# For GPU training, install the CUDA build of torch first: https://pytorch.org/get-started/locally/
pip install -e ".[all,dev]"
pytest
```

The tests use synthetic rasters and a fake detector, so they need no data, no network, and no GPU.

## Workflow

### 1. Get training data

| Dataset | Labels | GSD | Notes |
|---|---|---|---|
| **xView** (recommended for NAIP) | ~1M axis-aligned boxes, 60 classes | 0.3 m | Register at xviewdataset.org and download `train_images` + `xView_train.geojson`. License: CC BY-NC-SA 4.0. |
| **DOTA v1** | Oriented boxes, 15 classes | mixed (≈0.1–4.5 m) | No registration: `aerialdetect prep-dota` downloads it (~2 GB) and cuts it into chips. No building class. Academic use only. |

**Matching GSD matters most.** NAIP is 0.6 m in most years. At that size a car is only about 7 × 3 px, which is too
small for a detector to see reliably. The recommended recipe takes xView at 0.3 m, **degrades it to look like 0.6 m**,
and keeps it on a 0.3 m pixel grid. At inference, NAIP is upsampled 2× onto the same 0.3 m grid, so training and
inference images look alike.

```powershell
aerialdetect prep-xview --geojson data/xview/xView_train.geojson --images data/xview/train_images `
    --out data/xview_yolo_naip --chip-gsd 0.3 --degrade-gsd 0.6
```

The classes collapse to `car, truck_bus, building, aircraft, boat, storage_tank`. Edit
[configs/xview_classes.yaml](configs/xview_classes.yaml) to change them. If your NAIP state/year is natively
0.3 m, drop `--degrade-gsd`.

**No xView? Use DOTA.** Ultralytics' `yolo11s-obb.pt` is already trained on DOTA, so you can skip
training: [configs/naip_dota.yaml](configs/naip_dota.yaml) uses it directly, with DOTA's class names
(`small vehicle`, `large vehicle`) and no building analysis (DOTA has no building class).

```powershell
aerialdetect run configs/naip_dota.yaml
```

Upsampling NAIP to 0.2 m (`model.gsd: 0.2`) worked best on Tempe 2025: about 2.3× the cars found at
0.3 m, with few false positives. A 10-epoch fine-tune on native-resolution DOTA chips did *worse* on
NAIP than the stock model. To fine-tune anyway, `aerialdetect prep-dota` downloads DOTA (~2 GB) and cuts
it into 640 px chips at native resolution (`data/dota_split640/data.yaml`); blurring them to look like
NAIP first is the more promising route.

### 2. Fine-tune YOLO

```powershell
# Smoke test (tiny DOTA subset, auto-download, ~minutes on CPU)
aerialdetect train --data DOTA8.yaml --model yolo11n-obb.pt --epochs 3 --name smoke

# Real run on xView chips
aerialdetect train --data data/xview_yolo_naip/data.yaml --model yolo11s.pt --epochs 100 --name xview_naip --device 0

# Oriented boxes from DOTA (planes, ships, storage tanks, small/large vehicles)
aerialdetect train --data data/dota_split640/data.yaml --model yolo11s-obb.pt --epochs 100 --name dota_obb --device 0 --set degrees=180
```

On a 6 GB card (e.g. GTX 1660 Super) add `--batch -1` to let Ultralytics pick the largest batch that fits.
GTX 16-series cards train in full precision (Ultralytics disables AMP on them), so they use more memory.

The default augmentations are tuned for aerial imagery: vertical flips on, scale jitter limited so GSD stays
consistent. See [train.py](src/aerialdetect/train.py). Any Ultralytics argument can be passed with `--set key=value`.
Weights land in `runs/<name>/weights/best.pt`.

### 3. Run on an area you choose

Edit the bbox in [configs/naip_example.yaml](configs/naip_example.yaml). The example is ASU in Tempe, AZ.
Then run:

```powershell
aerialdetect run configs/naip_example.yaml
```

This performs the following steps:

1. Searches NAIP on Microsoft Planetary Computer and picks the newest year that fully covers the AOI. It then
   mosaics and crops the imagery to `data/cache/imagery/`.
2. Tiles the mosaic at 0.3 m with 128 px overlap and runs YOLO.
3. Georeferences each box and drops partial objects cut by tile seams. It then removes cross-tile duplicates
   with polygon NMS (IoU or intersection-over-smaller) and filters out physically implausible sizes (`min_area_m2` / `max_area_m2`).
4. Builds the analysis layers:
   - **`tract_density`** joins 2023 census tracts (downloaded automatically). For each tract it reports counts,
     objects per km², building footprint m², and built-up fraction. All values are normalised by the
     *imaged* part of each tract (`coverage` column).
   - **`parking_lots`** uses OpenStreetMap `amenity=parking` polygons, or your own via `lots_path`. Each lot
     gets a vehicle count, capacity (the OSM `capacity` tag, else area ÷ 30 m²/stall), and occupancy.
     The `aoi_frac` column flags lots that are only partly imaged.

Outputs are written to `outputs/tempe_naip/`:

| File | Contents |
|---|---|
| `results.gpkg` | layers `aoi`, `imagery_footprints`, `detections` (polygons), `detection_points`, `tract_density`, `parking_lots` |
| `tract_density.csv`, `parking_lots.csv` | attribute tables |
| `summary.json` | counts by class, GSDs used |

You can also run each stage on its own: `fetch-naip`, `detect`, `tracts`, `parking`. Run `aerialdetect <cmd> -h` for options.

### 4. Analyse

**geopandas**

```python
import geopandas as gpd
tracts = gpd.read_file("outputs/tempe_naip/results.gpkg", layer="tract_density")
tracts[tracts.coverage > 0.9].plot(column="building_cover_frac", legend=True)

lots = gpd.read_file("outputs/tempe_naip/results.gpkg", layer="parking_lots")
lots.query("aoi_frac > 0.95").sort_values("occupancy", ascending=False).head(10)
```

To join population or vehicle ownership, pull ACS tables (for example B01003 or B25044) on the 11-digit `GEOID`.

**ArcGIS Pro**

Add `results.gpkg` directly through Catalog. Or load everything into a file geodatabase and run example
analyses (car hot spots on a 100 m hex grid, Summarize Within by tract):

```powershell
propy arcgis/import_to_gdb.py outputs/tempe_naip/results.gpkg outputs/tempe_naip/results.gdb
```

## When the drone comes back

1. Process the flight to a georeferenced orthomosaic (OpenDroneMap, Pix4D, or DroneDeploy).
2. Set `source.paths` in [configs/drone_example.yaml](configs/drone_example.yaml) and run
   `aerialdetect run configs/drone_example.yaml`. The NAIP-trained model works as-is.
3. To get better results, label some drone chips (Label Studio or CVAT export YOLO format) and fine-tune
   from `best.pt` at a finer GSD, for example 0.1 m. Then set `model.gsd` to match. At finer GSDs you can also
   add classes that are invisible at 0.6 m, such as individual parking stalls.
4. For a different sensor, write a new `RasterSource` subclass that returns GeoTIFFs. See
   [sources/](src/aerialdetect/sources/).

## Known limitations

- **NAIP is a single leaf-on, mid-day snapshot** taken every 2–3 years. Occupancy is "at that moment", not a
  typical day, and tree canopy hides cars.
- **Axis-aligned boxes overstate building footprints**, especially for rotated buildings. Use an OBB model, or swap
  in a segmentation model, if footprint area matters.
- **The area-based parking capacity is a rough estimate.** Use `capacity` attributes or a stall-count layer when you have them.
- **Licenses:** xView is non-commercial, DOTA is academic use only, OSM data is ODbL, and NAIP is public domain.

## Layout

```
src/aerialdetect/
  sources/      NAIP (Planetary Computer), local/drone GeoTIFF
  datasets/     xView -> YOLO converter (GSD matching)
  tiling.py     GSD-normalised tiling
  detect.py     YOLO predictor + georeferencing
  nms.py        cross-tile polygon NMS
  features.py   cleaning, measurements, GeoPackage export
  analysis/     tract density, parking occupancy
  pipeline.py   end-to-end run;  cli.py  command line
configs/        run + class-map configs
arcgis/         arcpy import + example analyses
tests/          synthetic-data tests (no network/GPU)
```
