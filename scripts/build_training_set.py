"""Turn blind hand-count sets into a YOLO-OBB fine-tuning dataset.

    python scripts/build_training_set.py --set labels/irvine outputs/irvine_{year} \
        --set labels/sacramento outputs/sacramento_c05_{year} --out data/train_obb --holdout-frac 0.34

For every hand-counted crop: re-read the lot from the run's raster at the model GSD (same
bilinear upsampling as inference), black out everything outside the lot outline (cars there
were not clicked, so they must not be shown as background), and turn each click into an
oriented box. A click within --snap-m of an existing detection takes that detection's box
(right size and heading); other clicks take the median box size and the heading of the
nearest snapped neighbour (cars in a row share a heading), else an axis-aligned box.
Images are cut into fixed tile_px tiles and padded, so the model sees cars at inference scale.
Lots (not crops) are split into train/val so every year of a held-out lot is unseen.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import Polygon

VEHICLES = {"small vehicle", "large vehicle"}
SMALL_VEHICLE = 10  # DOTA-v1 class id; keeps the pretrained head's class layout


def box_params(poly: np.ndarray) -> tuple[float, float, float]:
    """(length, width, heading) of a 4-corner box in metres/radians."""
    e1, e2 = poly[1] - poly[0], poly[2] - poly[1]
    l1, l2 = np.hypot(*e1), np.hypot(*e2)
    e = e1 if l1 >= l2 else e2
    return max(l1, l2), min(l1, l2), float(np.arctan2(e[1], e[0]))


def make_box(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    u = np.array([np.cos(heading), np.sin(heading)]) * length / 2
    v = np.array([-np.sin(heading), np.cos(heading)]) * width / 2
    c = np.array([cx, cy])
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def crop_boxes(points_map: np.ndarray, dets: gpd.GeoDataFrame, snap_m: float,
               default: tuple[float, float]) -> tuple[list[np.ndarray], int]:
    cen = np.array([(p.x, p.y) for p in dets.geometry.centroid]).reshape(-1, 2)
    polys = [np.asarray(g.exterior.coords)[:4] for g in dets.geometry]
    boxes: list[np.ndarray | None] = [None] * len(points_map)
    headings: list[float | None] = [None] * len(points_map)
    used: set[int] = set()
    for i, p in enumerate(points_map):
        if len(cen):
            d = np.hypot(*(cen - p).T)
            for j in np.argsort(d):
                if d[j] > snap_m:
                    break
                if j not in used:
                    used.add(j)
                    L, W, h = box_params(polys[j])
                    if 3.0 <= L <= 7.5 and 1.4 <= W <= 3.0:  # plausible car; else just borrow heading
                        boxes[i] = make_box(*p, L, W, h)
                    headings[i] = h
                    break
    snapped = [k for k, h in enumerate(headings) if h is not None]
    for i, p in enumerate(points_map):
        if boxes[i] is not None:
            continue
        h = headings[i]
        if h is None and snapped:
            k = min(snapped, key=lambda k: np.hypot(*(points_map[k] - p)))
            h = headings[k]
        boxes[i] = make_box(*p, default[0], default[1], h if h is not None else 0.0)
    return boxes, sum(b is not None for b in headings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=2, action="append", metavar=("LABELS_DIR", "RUNS"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.34)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model-gsd", type=float, default=0.2)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=480)
    ap.add_argument("--snap-m", type=float, default=2.0)
    ap.add_argument("--box", type=float, nargs=2, default=(4.6, 1.9), help="default car length width (m)")
    a = ap.parse_args()

    out = Path(a.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    split_info, stats = {}, {"tiles": {"train": 0, "val": 0}, "boxes": {"train": 0, "val": 0}, "snapped": 0, "clicks": 0}

    for labels_dir, runs in a.set:
        d = Path(labels_dir)
        city = d.name
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        cnt = json.loads((d / "counts.json").read_text(encoding="utf-8"))
        pts = cnt["points_px"] if "points_px" in cnt else {k: v["points_px"] for k, v in cnt["crops"].items()}

        # stratify the hold-out by lot type so both splits see offices, retail, ...
        by_type: dict[str, list[int]] = {}
        for m in man.values():
            by_type.setdefault(m["lot_type"], [])
            if m["lot_id"] not in by_type[m["lot_type"]]:
                by_type[m["lot_type"]].append(m["lot_id"])
        val_lots: list[int] = []
        carry = 0.0
        for t in sorted(by_type):
            ids = sorted(by_type[t])
            rng.shuffle(ids)
            carry += len(ids) * a.holdout_frac
            take = int(carry)
            carry -= take
            val_lots += ids[:take]
        split_info[city] = sorted(int(x) for x in val_lots)

        dets_cache: dict[int, gpd.GeoDataFrame] = {}
        rasters: dict[int, str] = {}
        for cid, m in sorted(man.items()):
            y = m["year"]
            if y not in dets_cache:
                run = Path(runs.format(year=y))
                g = gpd.read_file(run / "results.gpkg", layer="detections")
                dets_cache[y] = g[g["class_name"].isin(VEHICLES)]
                rasters[y] = json.loads((run / "summary.json").read_text())["rasters"][0]
            split = "val" if m["lot_id"] in val_lots else "train"
            a_, b_, c_, d_, e_, f_ = m["transform"]
            w, h = m["width"], m["height"]
            up = round(abs(a_) / a.model_gsd)
            with rasterio.open(rasters[y]) as src:
                dets = dets_cache[y].to_crs(src.crs)
                win = from_bounds(c_, f_ + e_ * h, c_ + a_ * w, f_, src.transform)
                img = src.read([1, 2, 3], window=win, out_shape=(3, h * up, w * up),
                               resampling=Resampling.bilinear, boundless=True)
            H, W = img.shape[1:]
            # lot mask in upsampled pixels
            rings = [Polygon([(x * up, yy * up) for x, yy in r]) for r in m["outline_px"]]
            mask = rasterize([(r, 1) for r in rings], out_shape=(H, W), fill=0, dtype="uint8").astype(bool)
            img[:, ~mask] = 0
            img = np.moveaxis(img, 0, -1)

            p_px = np.array(pts.get(cid, []), dtype="float64").reshape(-1, 2)
            p_map = np.column_stack([a_ * p_px[:, 0] + c_, e_ * p_px[:, 1] + f_]) if len(p_px) else p_px
            near = dets[dets.intersects(Polygon([(c_, f_), (c_ + a_ * w, f_), (c_ + a_ * w, f_ + e_ * h), (c_, f_ + e_ * h)]))]
            boxes, n_snap = crop_boxes(p_map, near, a.snap_m, tuple(a.box))
            stats["snapped"] += n_snap
            stats["clicks"] += len(boxes)
            boxes_px = [np.column_stack([(bx[:, 0] - c_) / a_ * up, (bx[:, 1] - f_) / e_ * up]) for bx in boxes]

            ys = list(range(0, max(H - a.tile, 0) + 1, a.stride)) or [0]
            xs = list(range(0, max(W - a.tile, 0) + 1, a.stride)) or [0]
            if ys[-1] + a.tile < H:
                ys.append(H - a.tile)
            if xs[-1] + a.tile < W:
                xs.append(W - a.tile)
            for ty in ys:
                for tx in xs:
                    sub = mask[ty:ty + a.tile, tx:tx + a.tile]
                    if sub.mean() < 0.02:
                        continue
                    canvas = np.zeros((a.tile, a.tile, 3), dtype="uint8")
                    part = img[ty:ty + a.tile, tx:tx + a.tile]
                    canvas[:part.shape[0], :part.shape[1]] = part
                    lines = []
                    for bp in boxes_px:
                        cx, cy = bp.mean(axis=0)
                        if not (tx <= cx < tx + a.tile and ty <= cy < ty + a.tile):
                            continue
                        q = np.clip((bp - [tx, ty]) / a.tile, 0, 1)
                        lines.append(f"{SMALL_VEHICLE} " + " ".join(f"{v:.5f}" for v in q.ravel()))
                    name = f"{city}_{cid}_{ty}_{tx}"
                    Image.fromarray(canvas).save(out / "images" / split / f"{name}.png")
                    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                    stats["tiles"][split] += 1
                    stats["boxes"][split] += len(lines)

    from ultralytics import YOLO
    names = YOLO("yolo11s-obb.pt").names
    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {k}: {v}" for k, v in sorted(names.items())]
    (out / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    (out / "split.json").write_text(json.dumps({"val_lots": split_info, "stats": stats}, indent=1))
    print(json.dumps({"val_lots": split_info, "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
