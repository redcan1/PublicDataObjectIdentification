"""Year-over-year parking trends over every lot in a study area, calibrated against hand counts.

    python scripts/lot_trends.py --runs outputs/irvine_{year} --years 2016 2018 2020 2022 2024 \
        --labels labels/irvine --base 2018

Model vehicles per lot come from each run's ``parking_lots`` layer (detections at the run's conf).
The model's miss/false-alarm rate differs by year, so each year's total is also scaled by a
ratio estimator from the hand-counted sample (sum hand / sum model in that year), with a
bootstrap over hand-counted lots for a 90% interval. Writes ``<labels>/trends.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--labels", required=True, help="dir with score.csv from score_counts.py")
    ap.add_argument("--score", default="score.csv", help="score file inside --labels")
    ap.add_argument("--base", type=int, required=True, help="pre-COVID reference year")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=None,
                    help="recount vehicles per lot from detections at this confidence (default: run's counts)")
    a = ap.parse_args()

    frames = []
    for y in a.years:
        gpkg = Path(a.runs.format(year=y)) / "results.gpkg"
        l = gpd.read_file(gpkg, layer="parking_lots")
        l = l[(l["aoi_frac"] >= 0.99) & (l["parking"] != "street_side")]
        n = l["n_vehicles"]
        if a.min_conf is not None:
            d = gpd.read_file(gpkg, layer="detections")
            d = d[d["class_name"].isin(["small vehicle", "large vehicle"]) & (d["confidence"] >= a.min_conf)]
            c = gpd.GeoDataFrame(geometry=d.geometry.centroid, crs=d.crs).to_crs(l.crs)
            hit = gpd.sjoin(c, l[["lot_id", "geometry"]], predicate="within")
            n = l["lot_id"].map(hit.groupby("lot_id").size()).fillna(0).astype(int)
        frames.append(pd.DataFrame({"lot_id": l["lot_id"], "lot_type": l["lot_type"], "year": y,
                                    "garage": l["parking"].eq("multi-storey"), "area_m2": l["area_m2"],
                                    "model": n}))
    lots = pd.concat(frames)
    surf = lots[~lots["garage"]]
    n_lots = surf["lot_id"].nunique()

    # per-year calibration from the hand-counted sample
    sc = pd.read_csv(Path(a.labels) / a.score)
    per_lot = sc.groupby(["lot_id", "year"])[["hand", "model"]].sum().reset_index()
    sample_ids = per_lot["lot_id"].unique()
    rng = np.random.default_rng(a.seed)

    def factors(ids):
        s = per_lot[per_lot["lot_id"].isin(ids)] if ids is not None else per_lot
        if ids is not None:  # bootstrap: duplicates count multiple times
            s = pd.concat([per_lot[per_lot["lot_id"] == i] for i in ids])
        g = s.groupby("year")[["hand", "model"]].sum()
        return (g["hand"] / g["model"].clip(lower=1)).reindex(a.years)

    f = factors(None)
    boot = np.array([factors(rng.choice(sample_ids, len(sample_ids), replace=True)).to_numpy() for _ in range(a.boot)])

    tot = surf.groupby("year")["model"].sum().reindex(a.years)
    cal = tot * f
    cal_boot = boot * tot.to_numpy()[None, :]
    bi = a.years.index(a.base)
    idx_boot = cal_boot / cal_boot[:, [bi]]

    out = pd.DataFrame({"model_raw": tot, "raw_vs_base": tot / tot[a.base], "calib_factor": f.round(2),
                        "calibrated": cal.round(0), "calib_vs_base": cal / cal[a.base],
                        "calib_vs_base_lo": np.percentile(idx_boot, 5, axis=0),
                        "calib_vs_base_hi": np.percentile(idx_boot, 95, axis=0)})
    print(f"\nSurface lots fully in AOI: {n_lots} (garages excluded: {lots[lots.garage].lot_id.nunique()}); "
          f"calibration sample: {len(sample_ids)} lots")
    print(out.round(2).to_string())

    # median per-lot change vs base (robust to a few big lots), by lot type
    piv = surf.pivot_table(index=["lot_id", "lot_type"], columns="year", values="model", aggfunc="sum")
    ok = piv[piv[a.base] >= 5]  # need a non-trivial baseline for a ratio
    rel = ok.div(ok[a.base], axis=0)
    print(f"\nMedian per-lot model count vs {a.base} (lots with >=5 vehicles in {a.base}):")
    med = rel.groupby(level="lot_type").median()
    med["n_lots"] = rel.groupby(level="lot_type").size()
    allrow = rel.median().to_frame("ALL").T
    allrow["n_lots"] = len(rel)
    print(pd.concat([med, allrow]).round(2).to_string())

    print("\nRaw model totals by lot type:")
    print(surf.pivot_table(index="lot_type", columns="year", values="model", aggfunc="sum").to_string())

    tf = Path(a.labels) / a.score.replace("score", "trends")
    out.to_csv(tf)
    print(f"\n-> {tf}")


if __name__ == "__main__":
    main()
