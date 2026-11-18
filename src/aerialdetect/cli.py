"""Command-line interface: ``aerialdetect <command> ...``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml


def _kv(s: str) -> tuple[str, object]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    k, v = s.split("=", 1)
    return k, yaml.safe_load(v)


def _aoi_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--aoi", help="vector file defining the area of interest")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aerialdetect", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="end-to-end pipeline from a YAML config")
    s.add_argument("config")
    s.add_argument("--set", dest="overrides", type=_kv, action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set source.year=2020 --set output.dir=outputs/x")

    s = sub.add_parser("naip-dates", help="list NAIP flight dates, weekdays and resolution over an AOI")
    _aoi_args(s)
    s.add_argument("--start", type=int, help="first year (default: all)")
    s.add_argument("--end", type=int, help="last year")

    s = sub.add_parser("fetch-naip", help="download a NAIP mosaic for an AOI")
    _aoi_args(s)
    s.add_argument("--year", type=int)
    s.add_argument("--out-dir", default="data/cache/imagery")

    s = sub.add_parser("prep-xview", help="convert xView to YOLO chips")
    s.add_argument("--geojson", required=True, help="xView_train.geojson")
    s.add_argument("--images", required=True, help="folder of xView train_images/*.tif")
    s.add_argument("--out", required=True)
    s.add_argument("--classes", default="configs/xview_classes.yaml")
    s.add_argument("--chip-gsd", type=float, default=0.3)
    s.add_argument("--degrade-gsd", type=float, default=None,
                   help="simulate coarser imagery, e.g. 0.6 to match NAIP")
    s.add_argument("--tile-px", type=int, default=640)
    s.add_argument("--overlap-px", type=int, default=64)
    s.add_argument("--val-fraction", type=float, default=0.1)

    s = sub.add_parser("prep-dota", help="download DOTA v1 and cut it into native-resolution OBB chips")
    s.add_argument("--raw", default="data/DOTAv1", help="where DOTAv1.zip is (or gets) extracted")
    s.add_argument("--out", default="data/dota_split640")
    s.add_argument("--crop-px", type=int, default=640, help="chip size; train with --imgsz equal to this")
    s.add_argument("--gap-px", type=int, default=128, help="overlap between neighbouring chips")
    s.add_argument("--keep-background", type=float, default=0.1,
                   help="fraction of object-free chips to keep (1 = all)")
    s.add_argument("--no-download", action="store_true", help="fail instead of downloading if --raw is empty")

    s = sub.add_parser("train", help="fine-tune YOLO")
    s.add_argument("--data", required=True, help="dataset YAML (prep-xview output, DOTAv1.yaml, DOTA8.yaml...)")
    s.add_argument("--model", default="yolo11s.pt", help="starting weights, e.g. yolo11s.pt or yolo11s-obb.pt")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--device", default=None)
    s.add_argument("--name", default="train")
    s.add_argument("--set", dest="overrides", type=_kv, action="append", default=[],
                   metavar="KEY=VALUE", help="extra Ultralytics train args, repeatable")

    s = sub.add_parser("detect", help="run a model over GeoTIFF(s) and write a GeoPackage")
    s.add_argument("--raster", nargs="+", required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--out", required=True, help="output .gpkg")
    s.add_argument("--model-gsd", type=float, default=0.3)
    s.add_argument("--tile-px", type=int, default=640)
    s.add_argument("--overlap-px", type=int, default=128)
    s.add_argument("--bands", type=int, nargs=3, default=[1, 2, 3])
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--device", default=None)

    s = sub.add_parser("tracts", help="density of detections by census tract")
    s.add_argument("--detections", required=True, help=".gpkg from detect/run (uses its 'aoi' layer)")
    s.add_argument("--out", required=True)
    s.add_argument("--year", type=int, default=2023)
    s.add_argument("--count-classes", nargs="+", default=[])
    s.add_argument("--footprint-classes", nargs="*", default=[])
    s.add_argument("--cache-dir", default="data/cache")

    s = sub.add_parser("parking", help="parking-lot occupancy")
    s.add_argument("--detections", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--vehicle-classes", nargs="+", required=True)
    s.add_argument("--lots", help="lot polygons; default: OpenStreetMap amenity=parking")
    s.add_argument("--m2-per-stall", type=float, default=30.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "run":
        from .config import load_config
        from .pipeline import run

        for k, v in run(load_config(args.config, args.overrides)).items():
            print(f"{k}: {v}")

    elif args.cmd == "naip-dates":
        from .geo import load_aoi
        from .sources.naip import survey

        rows = survey(load_aoi(args.bbox, args.aoi), args.start, args.end)
        print(f"{'year':<6}{'date':<12}{'day':<5}{'gsd_m':<8}coverage")
        for r in rows:
            gsd = "/".join(f"{g:g}" for g in r["gsd_m"])
            print(f"{r['year']:<6}{r['date']:<12}{r['weekday']:<5}{gsd:<8}{r['coverage']:.0%}")

    elif args.cmd == "fetch-naip":
        from .geo import load_aoi
        from .sources import NaipSource

        paths = NaipSource(year=args.year).fetch(load_aoi(args.bbox, args.aoi), Path(args.out_dir))
        print("\n".join(map(str, paths)))

    elif args.cmd == "prep-xview":
        from .datasets.xview import convert_xview, load_class_map

        data_yaml = convert_xview(args.geojson, args.images, args.out, load_class_map(args.classes),
                                  chip_gsd=args.chip_gsd, degrade_gsd=args.degrade_gsd, tile_px=args.tile_px,
                                  overlap_px=args.overlap_px, val_fraction=args.val_fraction)
        print(data_yaml)

    elif args.cmd == "prep-dota":
        from .datasets.dota import prep_dota

        print(prep_dota(args.raw, args.out, crop_px=args.crop_px, gap_px=args.gap_px,
                        download=not args.no_download, keep_background=args.keep_background))

    elif args.cmd == "train":
        from .train import train

        best = train(args.data, model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                     device=args.device, name=args.name, **dict(args.overrides))
        print(best)

    elif args.cmd == "detect":
        from .config import DEFAULTS
        from .detect import YoloPredictor
        from .features import to_points, write_gpkg
        from .geo import raster_footprints
        from .pipeline import detect_step

        predictor = YoloPredictor(args.weights, conf=args.conf, imgsz=args.tile_px, device=args.device)
        detect_cfg = {**DEFAULTS["detect"], "tile_px": args.tile_px, "overlap_px": args.overlap_px,
                      "bands": args.bands, "batch_size": args.batch}
        dets = detect_step([Path(r) for r in args.raster], predictor, predictor.names,
                           {"gsd": args.model_gsd}, detect_cfg)
        fp = raster_footprints(args.raster)
        aoi = fp.dissolve().assign(name="aoi")[["name", "geometry"]]
        print(write_gpkg(args.out, {"aoi": aoi, "imagery_footprints": fp,
                                    "detections": dets, "detection_points": to_points(dets)}))
        print(dets["class_name"].value_counts().to_string() if not dets.empty else "no detections")

    elif args.cmd in ("tracts", "parking"):
        import geopandas as gpd

        from .features import write_gpkg

        dets = gpd.read_file(args.detections, layer="detections")
        aoi = gpd.read_file(args.detections, layer="aoi")
        if args.cmd == "tracts":
            from .pipeline import tracts_step

            layer = tracts_step(dets, aoi, {"year": args.year, "count_classes": args.count_classes,
                                            "footprint_classes": args.footprint_classes}, args.cache_dir)
            name = "tract_density"
        else:
            from .pipeline import parking_step

            layer = parking_step(dets, aoi, {"lots_path": args.lots, "vehicle_classes": args.vehicle_classes,
                                             "m2_per_stall": args.m2_per_stall})
            name = "parking_lots"
        out = write_gpkg(args.out, {name: layer})
        layer.drop(columns="geometry").to_csv(Path(out).with_suffix(".csv"), index=False)
        print(out)


if __name__ == "__main__":
    main()
