"""Score vehicle detections against a blind hand-count set.

    python scripts/score_counts.py labels/sacramento --runs outputs/sacramento_{year}

For every crop: hand count vs. model count inside the lot outline, plus point matching
(hand click <-> detection centre within --match-m metres, one-to-one) for precision/recall.
Writes ``<dir>/score.csv`` and prints per-year and per-lot summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

VEHICLES = {"small vehicle", "large vehicle"}


def px_to_map(t, pts):
    a, b, c, d, e, f = t
    return [(a * x + b * y + c, d * x + e * y + f) for x, y in pts]


def match(h: np.ndarray, m: np.ndarray, r: float) -> int:
    if not len(h) or not len(m):
        return 0
    dist = np.hypot(h[:, None, 0] - m[None, :, 0], h[:, None, 1] - m[None, :, 1])
    n = 0  # greedy one-to-one, closest pairs first
    while True:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, j] > r:
            return n
        n += 1
        dist[i, :] = np.inf
        dist[:, j] = np.inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--match-m", type=float, default=3.0)
    ap.add_argument("--min-conf", type=float, default=0.0, help="extra confidence cut on top of the run's")
    a = ap.parse_args()

    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cnt = json.loads((d / "counts.json").read_text(encoding="utf-8"))
    pts = cnt["points_px"] if "points_px" in cnt else {k: v["points_px"] for k, v in cnt["crops"].items()}
    unsure = set(cnt.get("unsure", [k for k, v in cnt.get("crops", {}).items() if v.get("unsure")]))

    dets, lots = {}, {}
    rows = []
    for cid, m in sorted(man.items()):
        y = m["year"]
        if y not in dets:
            run = Path(a.runs.format(year=y)) / "results.gpkg"
            g = gpd.read_file(run, layer="detections")
            g = g[g["class_name"].isin(VEHICLES) & (g["confidence"] >= a.min_conf)]
            dets[y] = g.to_crs(m["crs"])
            lots[y] = gpd.read_file(run, layer="parking_lots").set_index("lot_id")
        lot = unary_union([Polygon(px_to_map(m["transform"], r)) for r in m["outline_px"]])
        g = dets[y]
        cen = g.geometry.centroid
        inside = cen[cen.within(lot)]
        model_xy = np.array([(p.x, p.y) for p in inside]).reshape(-1, 2)
        hand_xy = np.array(px_to_map(m["transform"], pts.get(cid, []))).reshape(-1, 2)
        tp = match(hand_xy, model_xy, a.match_m)
        lot_row = lots[y].loc[m["lot_id"]] if m["lot_id"] in lots[y].index else None
        rows.append({"crop": cid, "lot_id": m["lot_id"], "lot_type": m["lot_type"], "year": y,
                     "hand": len(hand_xy), "model": len(model_xy), "matched": tp,
                     "model_lot_n": None if lot_row is None else int(lot_row["n_vehicles"]),
                     "capacity_est": None if lot_row is None else lot_row["capacity_est"],
                     "unsure": cid in unsure})
    df = pd.DataFrame(rows)
    df.to_csv(d / "score.csv", index=False)

    pd.set_option("display.width", 200)
    tot = df[["hand", "model", "matched"]].sum()
    print(f"\nALL: hand={tot.hand} model={tot.model} matched={tot.matched}  "
          f"recall={tot.matched / tot.hand:.2f} precision={tot.matched / max(tot.model, 1):.2f} "
          f"model/hand={tot.model / tot.hand:.2f}")
    print(f"  per-crop count corr r={df.hand.corr(df.model):.3f}  MAE={np.abs(df.hand - df.model).mean():.2f}")
    by = df.groupby("year")[["hand", "model", "matched"]].sum()
    by["recall"] = by.matched / by.hand
    by["precision"] = by.matched / by.model.clip(lower=1)
    by["model/hand"] = by.model / by.hand
    print("\nBy year:\n", by.round(2).to_string())
    for col in ("hand", "model"):
        base = by.loc[by.index.min(), col]
        print(f"  {col} index vs {by.index.min()}: " + "  ".join(f"{yy}:{v / base:.2f}" for yy, v in by[col].items()))
    piv = df.pivot_table(index=["lot_id", "lot_type"], columns="year", values=["hand", "model"], aggfunc="sum")
    print("\nPer lot (hand | model):\n", piv.to_string())
    print(f"\n-> {d / 'score.csv'}")


if __name__ == "__main__":
    main()
