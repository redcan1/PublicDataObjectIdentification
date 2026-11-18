# aerialdetect

Find objects such as cars, buildings, planes and storage tanks in free aerial imagery, put them on a map, and
measure things with them. Examples: how full each parking lot is, or how many cars per km² each census tract has.

## What it does

Give it an area (a bounding box or a polygon) and it:

1. **Downloads the imagery.** It fetches US [NAIP](https://naip-usdaonline.hub.arcgis.com/) aerial photos
   (0.6 m, public domain) from Microsoft Planetary Computer, or uses your own GeoTIFF such as a drone
   orthomosaic.
2. **Detects objects** with a YOLO model, using either a pretrained model or one you fine-tune on a public
   labelled dataset (xView or DOTA).
3. **Turns detections into map features.** Every detection becomes a polygon in real-world coordinates,
   with duplicates removed.
4. **Runs analyses:**
   - **Census tracts:** object counts and density per tract (tracts are downloaded automatically).
   - **Parking lots:** vehicle count, capacity and occupancy for each lot, using OpenStreetMap lots or your own.
5. **Writes the results** to one GeoPackage (`results.gpkg`) that opens in QGIS, ArcGIS Pro or geopandas,
   plus CSV tables and a `summary.json`.

For example, with the included `configs/naip_dota.yaml` and 2025 NAIP imagery around Arizona State University
in Tempe, it finds about 900 cars in roughly 3.4 km² and reports occupancy for each lot.

## How it works

```
 imagery source ──► tile @ model GSD ──► YOLO ──► georeference + de-duplicate ──► GeoPackage ──► analysis
 (NAIP | drone)      (resample)          (HBB/OBB)  (tile seams, NMS in map space)  (QGIS/ArcGIS)  (tracts, parking)
```

**1. Imagery sources** ([sources/](src/aerialdetect/sources/)). Every source ends with a georeferenced
GeoTIFF. NAIP is searched on Planetary Computer, which picks the newest year that fully covers the area,
then mosaics and crops it. Drone orthomosaics or other local GeoTIFFs are read directly.

**2. Tiling at a fixed ground resolution** ([tiling.py](src/aerialdetect/tiling.py)). This is the key
idea. A detector learns what a car looks like *at a certain number of pixels*. Ground sample distance (GSD)
is how many metres one pixel covers. Every tile is resampled to the GSD the model expects, so a car is about
the same size in pixels whether it came from 0.6 m NAIP (upsampled) or a 3 cm drone image (downsampled).
Tiles overlap so that objects cut by one tile appear whole in the next.

**3. Detection** ([detect.py](src/aerialdetect/detect.py)). Batches of tiles go through an
[Ultralytics](https://docs.ultralytics.com/) YOLO model. Both axis-aligned boxes and oriented boxes
(OBB, which rotate with the object) are supported.

**4. Georeferencing and clean-up** ([detect.py](src/aerialdetect/detect.py),
[nms.py](src/aerialdetect/nms.py), [features.py](src/aerialdetect/features.py)). Pixel boxes are converted
to map polygons. Partial objects cut by a tile seam are dropped, duplicates from overlapping tiles are
merged with non-maximum suppression (NMS) in map space, and implausible sizes (for example a 200 m² "car")
are filtered out.

**5. Analysis** ([analysis/](src/aerialdetect/analysis/)).
- `tract_density` gives counts, objects per km² and footprint area per census tract, normalised by the
  part of each tract that was actually imaged.
- `parking_lots` gives vehicles per lot and capacity (the OSM `capacity` tag, or lot area ÷ 30 m² per
  space), then occupancy.

**Models and training data.** Two routes:

| Route | Model | Classes | Notes |
|---|---|---|---|
| **DOTA (quickest)** | Ultralytics' `yolo11s-obb.pt`, already trained on [DOTA](https://captain-whu.github.io/DOTA/) | `small vehicle`, `large vehicle`, plane, ship, storage tank, sports fields… | No training needed. No building class. Academic use only. |
| **xView** | YOLO fine-tuned on [xView](http://xviewdataset.org/) chips made to look like NAIP | `car`, `truck_bus`, `building`, `aircraft`, `boat`, `storage_tank` | Needs registration and a 20 GB download. Adds buildings. Non-commercial licence. |

## Setup

### Requirements

- Windows, macOS or Linux, with **Python 3.10–3.12**
- About 10 GB free disk space for the environment and cached imagery (more for training data)
- Optional: an **NVIDIA GPU**. Strongly recommended for training. Detection on a few km² runs fine on a CPU.
  You only need an up-to-date NVIDIA driver, not the CUDA Toolkit, because PyTorch includes its own CUDA runtime.

Keep the project **outside cloud-synced folders** such as OneDrive or Dropbox. The virtual environment,
datasets and model runs are several GB and change constantly.

### 1. Get the code

```powershell
git clone https://github.com/redcan1/PublicDataObjectIdentification.git
cd PublicDataObjectIdentification
```

### 2. Create a virtual environment

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch

**Install PyTorch first**, so the next step doesn't pull in the CPU-only build.

With an NVIDIA GPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

CPU only, or a Mac:

```powershell
pip install torch torchvision
```

For other CUDA versions, see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 4. Install the project

```powershell
pip install -e ".[all,dev]"
```

The extras are: `ml` (YOLO), `naip` (Planetary Computer), `osm` (OpenStreetMap parking lots), and `dev`
(tests). `all` installs the first three.

### 5. Check it works

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest
```

With a GPU, the first line should print `True` and your card's name. The tests use synthetic rasters and a
fake detector, so they need no data, no network and no GPU.

### 6. First run

```powershell
aerialdetect run configs/naip_dota.yaml
```

This downloads NAIP for the example area (Tempe, AZ) and the pretrained model, detects objects, and writes
results to `outputs/tempe_dota/`. Open `results.gpkg` in QGIS or ArcGIS Pro. To use your own area, change
`aoi.bbox` in the config (`[min_lon, min_lat, max_lon, max_lat]`).

## Usage

### Configs

| Config | Use it for |
|---|---|
| [configs/naip_dota.yaml](configs/naip_dota.yaml) | NAIP with the pretrained DOTA model (start here) |
| [configs/naip_dota_2023.yaml](configs/naip_dota_2023.yaml) | The same on Tempe's 2023 NAIP (0.3 m, flown during the fall semester) |
| [configs/naip_example.yaml](configs/naip_example.yaml) | NAIP with your own xView-trained model (`runs/xview_naip/weights/best.pt`) |
| [configs/drone_example.yaml](configs/drone_example.yaml) | A drone orthomosaic or other local GeoTIFF |

Key settings: `aoi` (the area), `source` (NAIP year or file paths), `model.weights`, `model.gsd` (the GSD
tiles are resampled to), `model.conf` (confidence threshold), `detect.classes` (optional list of classes to keep),
`detect.min_area_m2` / `max_area_m2` (per-class size limits), and the class lists under `analysis`.

For the pretrained DOTA model, set `model.gsd` to about a third to a half of the imagery's resolution. On
0.6 m NAIP (Tempe 2025), `0.2` (3× upsampling) worked best and found about 2.3× as many cars as `0.3`. On 0.3 m
NAIP (Tempe 2023), `0.15` worked best: `0.2` missed whole rows of tightly parked cars. Check
`properties.gsd` of the NAIP year you use, since many states switched to 0.3 m in 2022 or later. Lowering `conf` below 0.25 mostly adds false
positives such as empty spaces and shadows.

### Commands

```powershell
aerialdetect run CONFIG              # whole pipeline
aerialdetect fetch-naip --bbox ...   # just download imagery
aerialdetect detect --raster ... --weights ... --out ...   # detect on any GeoTIFF(s)
aerialdetect tracts --detections results.gpkg ...          # census tract density
aerialdetect parking --detections results.gpkg ...         # parking occupancy
aerialdetect prep-xview ...          # xView -> YOLO training chips
aerialdetect prep-dota               # DOTA -> YOLO-OBB training chips
aerialdetect train --data ... --model ...                  # fine-tune YOLO
```

Run `aerialdetect <command> -h` for options.

### Outputs

| File | Contents |
|---|---|
| `results.gpkg` | Layers `aoi`, `imagery_footprints`, `detections` (polygons), `detection_points`, `tract_density`, `parking_lots` |
| `tract_density.csv`, `parking_lots.csv` | Attribute tables |
| `summary.json` | Counts by class and the GSDs used |

In `parking_lots`, `aoi_frac` shows how much of each lot was imaged. Filter on `aoi_frac > 0.95` for
reliable occupancy. In `tract_density`, `coverage` does the same for tracts.

### Training your own model (optional)

**xView** (adds buildings). Register at xviewdataset.org and download the training images and
`xView_train.geojson` into `data/xview/`. Then:

```powershell
aerialdetect prep-xview --geojson data/xview/xView_train.geojson --images data/xview/train_images `
    --out data/xview_yolo_naip --chip-gsd 0.3 --degrade-gsd 0.6
aerialdetect train --data data/xview_yolo_naip/data.yaml --model yolo11s.pt --epochs 100 --name xview_naip --device 0
```

`--degrade-gsd 0.6` blurs the 0.3 m xView images to look like 0.6 m NAIP, so training matches what the
model will see. The classes are set in [configs/xview_classes.yaml](configs/xview_classes.yaml).

**DOTA.** `aerialdetect prep-dota` downloads DOTA (~2 GB) and cuts it into 640 px chips at native
resolution, keeping 10% of the empty chips. Note that a 10-epoch fine-tune on these sharp chips did *worse*
on NAIP than the stock `yolo11s-obb.pt`. Blurring the chips to look like NAIP first is the more promising
route.

**Training tips:**
- Weights are saved to `runs/<name>/weights/best.pt`. To continue after stopping, run the same command with
  `--model runs/<name>/weights/last.pt --set resume=True`.
- Pass any Ultralytics setting with `--set key=value` (for example `--set patience=20`).
- On 6 GB cards such as a GTX 1660 Super, use `--batch 8`. GTX 16-series cards train in full precision,
  because Ultralytics turns off mixed precision on them.
- Training defaults to `deterministic=False`. Ultralytics' default (`True`) made training about 8× slower
  on a GTX 1660 Super. Add `--set deterministic=True` if you need runs to repeat exactly.

### Analysing results

**geopandas:**

```python
import geopandas as gpd
lots = gpd.read_file("outputs/tempe_dota/results.gpkg", layer="parking_lots")
lots.query("aoi_frac > 0.95").sort_values("occupancy", ascending=False).head(10)

tracts = gpd.read_file("outputs/tempe_dota/results.gpkg", layer="tract_density")
tracts[tracts.coverage > 0.9].plot(column="small_vehicle_per_km2", legend=True)
```

To add population or vehicle ownership, join American Community Survey (ACS) tables, for example B01003 or
B25044, on the 11-digit `GEOID`.

**ArcGIS Pro:** add `results.gpkg` through Catalog, or load it into a file geodatabase with example
analyses (car hot spots on a 100 m hex grid, Summarize Within by tract):

```powershell
propy arcgis/import_to_gdb.py outputs/tempe_dota/results.gpkg outputs/tempe_dota/results.gdb
```

### Drone imagery

1. Process the flight into a georeferenced orthomosaic (OpenDroneMap, Pix4D or DroneDeploy).
2. Set `source.paths` in [configs/drone_example.yaml](configs/drone_example.yaml) and run
   `aerialdetect run configs/drone_example.yaml`. The tiles are resampled to the model's GSD, so a
   NAIP-trained model works as-is.
3. For better results, label some drone chips (Label Studio or CVAT, exported in YOLO format), fine-tune at a
   finer GSD such as 0.1 m, and set `model.gsd` to match.
4. For a different sensor, write a `RasterSource` subclass that returns GeoTIFFs. See
   [sources/](src/aerialdetect/sources/).

## Case study: COVID and parking

[docs/covid-parking.md](docs/covid-parking.md) uses this pipeline to measure parking before, during and after
COVID in Irvine and Sacramento. During the April 2020 lockdown, Irvine Spectrum's parked cars fell to 0.63x of
their 2018 level, and office-tower lots to about a quarter. It also shows how to fine-tune the detector on
blind hand counts, which fixes the low recall described below.

## Known limitations

- **NAIP is a single snapshot** taken in the growing season, every 2–3 years. Occupancy describes that
  moment, not a typical day. Arizona's 2025 imagery is from summer, when ASU's lots are almost empty.
  Tree canopy also hides cars.
- **0.6 m imagery is coarse.** A car is about 7 × 3 pixels, so some are missed, especially dark cars and cars
  in shadow.
- **False positives occur in some classes.** On Tempe 2025 the DOTA model labelled 72 car-sized (8–40 m²)
  rooftop objects as planes. `naip_dota.yaml` sets `min_area_m2: {plane: 100}`, which removes them but keeps
  real aircraft. Use `detect.classes` to keep only the classes you need.
- **Recall in packed lots is limited.** Even on 0.3 m imagery the stock model finds roughly half the cars in
  full, tightly packed lots, so occupancy is an underestimate. Tempe 2023 (a weekday in September) gives
  4.7% overall against 2.5% for summer 2025.
- **Axis-aligned boxes overstate building footprints**, especially for rotated buildings. Use an OBB or
  segmentation model if footprint area matters.
- **Area-based parking capacity is rough.** Use OSM `capacity` tags or your own stall counts when you have them.
- **Licences:** xView is non-commercial (CC BY-NC-SA 4.0), DOTA is academic use only, OpenStreetMap data is
  ODbL, and NAIP is public domain.

## Project layout

```
src/aerialdetect/
  sources/      imagery sources: NAIP (Planetary Computer), local/drone GeoTIFF
  datasets/     xView and DOTA -> YOLO converters
  tiling.py     GSD-normalised tiling
  detect.py     YOLO predictor + georeferencing
  nms.py        cross-tile polygon NMS
  features.py   clean-up, measurements, GeoPackage export
  analysis/     census tract density, parking occupancy
  pipeline.py   end-to-end run
  cli.py        command line
  train.py      fine-tuning with aerial augmentations
configs/        run configs and the xView class map
arcgis/         arcpy import + example analyses
tests/          synthetic-data tests (no network/GPU needed)
```
