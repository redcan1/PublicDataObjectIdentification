"""End-to-end run: imagery -> detections -> map features -> analysis layers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd

from .detect import Predictor, YoloPredictor, detect_raster
from .features import add_measurements, clip_to_aoi, merge_detections, to_points, write_gpkg
from .geo import load_aoi, raster_footprints
from .sources import build_source

log = logging.getLogger(__name__)


def detect_step(rasters: list[Path], predictor: Predictor, names: dict[int, str],
                model_cfg: dict[str, Any], detect_cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    parts = [
        detect_raster(
            r, predictor, names,
            model_gsd=model_cfg["gsd"],
            tile_px=detect_cfg["tile_px"],
            overlap_px=detect_cfg["overlap_px"],
            bands=detect_cfg["bands"],
            batch_size=detect_cfg["batch_size"],
            edge_margin_px=detect_cfg["edge_margin_px"],
            nms_iou=detect_cfg["nms_iou"],
            nms_ios=detect_cfg["nms_ios"],
        )
        for r in rasters
    ]
    dets = add_measurements(merge_detections(parts, detect_cfg["nms_iou"], detect_cfg["nms_ios"]))
    dets = filter_classes(dets, detect_cfg.get("classes"))
    return filter_by_size(dets, detect_cfg.get("min_area_m2") or {}, detect_cfg.get("max_area_m2") or {})


def filter_classes(dets: gpd.GeoDataFrame, classes: list[str] | None) -> gpd.GeoDataFrame:
    """Keep only the listed class names (None = keep all). Drops classes you don't need or don't trust."""
    if dets.empty or classes is None:
        return dets
    unknown = set(classes) - set(dets["class_name"])
    if unknown:
        log.debug("class filter: no detections for %s", sorted(unknown))
    keep = dets["class_name"].isin(classes)
    log.info("class filter removed %d detection(s)", int((~keep).sum()))
    return dets[keep].reset_index(drop=True)


def filter_by_size(dets: gpd.GeoDataFrame, min_area: dict[str, float], max_area: dict[str, float]) -> gpd.GeoDataFrame:
    """Drop physically implausible detections (e.g. a 200 m2 'car'). A cheap false-positive filter."""
    if dets.empty or not (min_area or max_area):
        return dets
    keep = dets["area_m2"].notna()
    for cls, v in min_area.items():
        keep &= ~((dets["class_name"] == cls) & (dets["area_m2"] < v))
    for cls, v in max_area.items():
        keep &= ~((dets["class_name"] == cls) & (dets["area_m2"] > v))
    log.info("size filter removed %d detection(s)", int((~keep).sum()))
    return dets[keep].reset_index(drop=True)


def tracts_step(dets: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, cfg: dict[str, Any], cache_dir: str | Path) -> gpd.GeoDataFrame:
    from .analysis.census import load_tracts
    from .analysis.zones import density_by_zone

    tracts = load_tracts(aoi, year=cfg["year"], cache_dir=cache_dir)
    keep = [c for c in ("GEOID", "NAMELSAD", "NAME", "STATEFP", "COUNTYFP", "ALAND") if c in tracts.columns]
    return density_by_zone(dets, tracts[keep + ["geometry"]], "GEOID", aoi,
                           cfg["count_classes"], cfg["footprint_classes"])


def parking_step(dets: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, cfg: dict[str, Any],
                 cache_dir: str | Path | None = None) -> gpd.GeoDataFrame:
    from .analysis.parking import load_osm_parking, parking_occupancy

    lots = gpd.read_file(cfg["lots_path"]) if cfg.get("lots_path") else load_osm_parking(aoi)
    out = parking_occupancy(dets, lots, cfg["vehicle_classes"], aoi=aoi, m2_per_stall=cfg["m2_per_stall"])
    zcfg = cfg.get("zoning")
    if zcfg and (zcfg.get("url") or zcfg.get("path")):
        from .analysis.zoning import classify_lots, load_zones

        out = classify_lots(out, load_zones(zcfg, aoi, cache_dir), zcfg["field"], zcfg.get("groups") or {})
    return out


def run(cfg: dict[str, Any], predictor: Predictor | None = None) -> dict[str, Path]:
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi_cfg = cfg.get("aoi") or {}
    aoi = load_aoi(aoi_cfg.get("bbox"), aoi_cfg.get("path")) if (aoi_cfg.get("bbox") or aoi_cfg.get("path")) else None

    source = build_source(cfg["source"])
    rasters = source.fetch(aoi, Path(cfg["cache_dir"]) / "imagery")
    footprints = raster_footprints(rasters)
    if aoi is None:  # e.g. a drone orthomosaic: the imagery defines the study area
        aoi = gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[footprints.union_all()], crs=footprints.crs)

    if predictor is None:
        m = cfg["model"]
        predictor = YoloPredictor(m["weights"], conf=m["conf"], iou=m["iou"],
                                  imgsz=cfg["detect"]["tile_px"], device=m["device"])
    names = getattr(predictor, "names", {})

    dets = clip_to_aoi(detect_step(rasters, predictor, names, cfg["model"], cfg["detect"]), aoi)
    layers: dict[str, gpd.GeoDataFrame | None] = {
        "aoi": aoi,
        "imagery_footprints": footprints,
        "detections": dets,
        "detection_points": to_points(dets),
    }

    analysis = cfg["analysis"]
    if analysis["tracts"]["enabled"]:
        layers["tract_density"] = tracts_step(dets, aoi, analysis["tracts"], cfg["cache_dir"])
    if analysis["parking"]["enabled"]:
        layers["parking_lots"] = parking_step(dets, aoi, analysis["parking"], cfg["cache_dir"])

    gpkg = write_gpkg(out_dir / cfg["output"]["gpkg"], layers)
    outputs = {"gpkg": gpkg}
    for name in ("tract_density", "parking_lots"):
        if layers.get(name) is not None and not layers[name].empty:
            csv = out_dir / f"{name}.csv"
            layers[name].drop(columns="geometry").to_csv(csv, index=False)
            outputs[name] = csv

    summary = {
        "rasters": [str(r) for r in rasters],
        "imagery": getattr(source, "metadata", {}),
        "native_gsd_m": sorted(set(dets["native_gsd"].tolist())) if not dets.empty else [],
        "model_gsd_m": cfg["model"]["gsd"],
        "n_detections": int(len(dets)),
        "by_class": dets["class_name"].value_counts().to_dict() if not dets.empty else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["summary"] = out_dir / "summary.json"
    log.info("summary: %s", summary)
    return outputs
