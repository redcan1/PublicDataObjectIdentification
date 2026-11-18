"""Convert a raw Overpass ``amenity=parking`` JSON (``out geom``) into a lots file for ``lots_path``.

Fallback for when osmnx can't reach Overpass. Fetch with, e.g.:

    curl -A "aerialdetect-research/0.1" --data-urlencode \
      'data=[out:json][timeout:60];(way["amenity"="parking"](S,W,N,E);relation["amenity"="parking"](S,W,N,E););out geom;' \
      https://overpass-api.de/api/interpreter -o lots.json
    python scripts/osm_lots_to_gpkg.py lots.json data/lots/<name>.gpkg

Output columns match ``load_osm_parking``: lot_id, osm_id, name, capacity, parking, access, operator.
"""

from __future__ import annotations

import json
import sys

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

TAGS = ("name", "capacity", "parking", "access", "operator")


def ring(geom):
    return [(p["lon"], p["lat"]) for p in geom]


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    els = json.load(open(src, encoding="utf-8"))["elements"]
    rows = []
    for e in els:
        t = e.get("tags", {})
        if e["type"] == "way":
            pts = ring(e.get("geometry", []))
            if len(pts) < 4 or pts[0] != pts[-1]:
                continue
            geom = Polygon(pts)
        elif e["type"] == "relation":
            outer = [LineString(ring(m["geometry"])) for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
            inner = [LineString(ring(m["geometry"])) for m in e.get("members", []) if m.get("role") == "inner" and m.get("geometry")]
            if not outer:
                continue
            geom = unary_union(list(polygonize(unary_union(outer))))
            if inner:
                geom = geom.difference(unary_union(list(polygonize(unary_union(inner)))))
        else:
            continue
        if geom.is_empty or not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        rows.append({"osm_id": f"{e['type']}/{e['id']}", **{k: t.get(k) for k in TAGS}, "geometry": geom})
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    g = g[g["parking"] != "underground"].reset_index(drop=True)  # invisible from the air
    g.insert(0, "lot_id", range(len(g)))
    g.to_file(dst, driver="GPKG")
    print(f"{len(g)} lots -> {dst}; parking types: {g['parking'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
