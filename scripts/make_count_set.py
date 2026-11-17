"""Build a blind hand-count set: the same parking lots cropped from every year's NAIP.

    python scripts/make_count_set.py --runs outputs/sacramento_{year} --years 2016 2020 2022 2024 \
        --out labels/sacramento --n-lots 15

Writes ``<out>/manifest.json`` (crop id -> lot, year, georeferencing; kept private to the analysis)
and ``<out>/crops/<id>.jpg``. Crop ids are shuffled so the counter cannot tell the year.
Clicks are stored in native-pixel coordinates of the crop; ``manifest[id]["transform"]`` maps
them back to map coordinates for matching against detections.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window, from_bounds

UPSCALE = 3      # crops are upsampled for display; clicks are stored at native resolution
PAD_M = 12       # context around the lot


def pick_lots(lots: gpd.GeoDataFrame, n: int, seed: int, min_m2: float, max_m2: float) -> gpd.GeoDataFrame:
    """Surface lots fully inside the AOI, spread across lot types and sizes."""
    c = lots[(~lots["parking"].isin(["multi-storey", "street_side"])) & (lots["aoi_frac"] >= 0.99)
             & lots["area_m2"].between(min_m2, max_m2)].copy()
    rng = random.Random(seed)
    types = sorted(c["lot_type"].unique())
    picked: list[int] = []
    # round-robin over lot types, taking a random lot from a different size tercile each time
    c["size_bin"] = np.digitize(c["area_m2"], np.quantile(c["area_m2"], [1 / 3, 2 / 3]))
    pools = {t: c[c["lot_type"] == t].index.tolist() for t in types}
    for p in pools.values():
        rng.shuffle(p)
    i = 0
    while len(picked) < n and any(pools.values()):
        t = types[i % len(types)]
        if pools[t]:
            want = (i // len(types) + types.index(t)) % 3  # rotate size bins per pass and per type
            j = next((k for k, idx in enumerate(pools[t]) if c.at[idx, "size_bin"] == want), 0)
            picked.append(pools[t].pop(j))
        i += 1
    return c.loc[picked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="run dir pattern with {year}")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--imagery", default=None, help="raster glob with {year}; default: the run's summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-lots", type=int, default=15)
    ap.add_argument("--min-m2", type=float, default=700)
    ap.add_argument("--max-m2", type=float, default=4500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    lots = gpd.read_file(Path(a.runs.format(year=a.years[0])) / "results.gpkg", layer="parking_lots")
    chosen = pick_lots(lots, a.n_lots, a.seed, a.min_m2, a.max_m2)
    print("lots:", chosen[["lot_id", "lot_type", "area_m2"]].to_string(index=False))

    jobs = [(int(lid), y) for lid in chosen["lot_id"] for y in a.years]
    random.Random(a.seed + 1).shuffle(jobs)
    manifest = {}
    for k, (lid, year) in enumerate(jobs):
        cid = f"c{k + 1:02d}"
        if a.imagery:
            tif = glob.glob(a.imagery.format(year=year))
        else:
            tif = json.loads((Path(a.runs.format(year=year)) / "summary.json").read_text())["rasters"]
        assert len(tif) == 1, (year, tif)
        with rasterio.open(tif[0]) as src:
            lot = chosen[chosen["lot_id"] == lid].to_crs(src.crs).geometry.iloc[0]
            x0, y0, x1, y1 = lot.buffer(PAD_M).bounds
            w = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
            w = Window(int(w.col_off), int(w.row_off), int(w.width), int(w.height))
            rgb = np.moveaxis(src.read([1, 2, 3], window=w, boundless=True), 0, -1)
            t = src.window_transform(w)
            inv = ~t
            rings = [lot.exterior] if lot.geom_type == "Polygon" else [g.exterior for g in lot.geoms]
            outline = [[[round(c, 2) for c in inv * xy] for xy in r.coords] for r in rings]
            crs = src.crs.to_string()
        img = Image.fromarray(rgb).resize((rgb.shape[1] * UPSCALE, rgb.shape[0] * UPSCALE), Image.LANCZOS)
        img.save(out / "crops" / f"{cid}.jpg", quality=82, optimize=True)
        manifest[cid] = {"lot_id": lid, "year": year, "width": rgb.shape[1], "height": rgb.shape[0],
                         "upscale": UPSCALE, "outline_px": outline, "crs": crs,
                         "transform": list(t)[:6], "lot_type": chosen.loc[chosen["lot_id"] == lid, "lot_type"].iloc[0]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} crops -> {out}")


if __name__ == "__main__":
    main()
