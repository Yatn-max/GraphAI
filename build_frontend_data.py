"""
Build the data file for the map frontend
=========================================
Reads
    - the SUMO net file (road shapes)                 e.g. osm.net.xml.gz
    - the scenario folders made by run_disruption_scenarios.py   e.g. disruptions/scenario_001/...
and writes
    frontend/data.js        <- everything the web page needs (road shapes in lat/lon, traffic, scenarios)

Then open frontend/index.html (see --serve below).

Usage
-----
    python build_frontend_data.py --net osm.net.xml.gz --scenarios disruptions --out frontend
    python build_frontend_data.py --net osm.net.xml.gz --scenarios disruptions --out frontend --serve

Needs only pandas and numpy (no PyTorch, no openpyxl).
"""

import argparse
import gzip
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# SUMO coordinates -> latitude / longitude
# --------------------------------------------------------------------------------------

def utm_to_latlon(easting, northing, zone, south=False):
    """Inverse UTM projection (WGS84), standard series formulas. Returns (lat, lon) in degrees."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)

    x = easting - 500000.0
    y = northing - (10000000.0 if south else 0.0)
    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))

    s, c, t = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    C1 = ep2 * c ** 2
    T1 = t ** 2
    N1 = a / math.sqrt(1 - e2 * s ** 2)
    R1 = a * (1 - e2) / (1 - e2 * s ** 2) ** 1.5
    D = x / (N1 * k0)

    lat = phi1 - (N1 * t / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)               # central meridian of the zone
    lon = lon0 + (D
                  - (1 + 2 * T1 + C1) * D ** 3 / 6
                  + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2) * D ** 5 / 120) / c
    return math.degrees(lat), math.degrees(lon)


def read_network(net_path):
    """Returns (geo_info, {edge_id: (name, [(x, y), ...])}) from the SUMO net file."""
    opener = gzip.open if str(net_path).endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        root = ET.parse(fh).getroot()

    loc = root.find("location")
    offset = tuple(float(v) for v in loc.get("netOffset", "0,0").split(","))
    proj = loc.get("projParameter", "")
    zone_match = re.search(r"\+zone=(\d+)", proj)
    geo = None
    if "+proj=utm" in proj and zone_match:
        geo = {"zone": int(zone_match.group(1)), "south": "+south" in proj, "offset": offset}

    shapes = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        shape = edge.get("shape")
        if not shape:                                           # fall back to the first lane's shape
            lane = edge.find("lane")
            shape = lane.get("shape") if lane is not None else None
        if not shape:
            continue
        pts = [tuple(float(v) for v in p.split(",")[:2]) for p in shape.split()]
        shapes[edge.get("id")] = (edge.get("name", ""), pts)
    return geo, shapes


def convert_points(pts, geo):
    if geo is None:                                             # no known projection: keep SUMO metres
        return [[round(x, 2), round(y, 2)] for x, y in pts]
    ox, oy = geo["offset"]
    out = []
    for x, y in pts:
        lat, lon = utm_to_latlon(x - ox, y - oy, geo["zone"], geo["south"])
        out.append([round(lat, 6), round(lon, 6)])
    return out


# --------------------------------------------------------------------------------------
# Scenario folders -> compact JSON
# --------------------------------------------------------------------------------------

def build(net_path, scen_root, out_dir):
    scen_root, out_dir = Path(scen_root), Path(out_dir)
    folders = sorted(p for p in scen_root.glob("scenario_*") if (p / "traffic.csv").exists())
    if not folders:
        raise SystemExit(f"No scenario_XXX folders with traffic.csv found in '{scen_root}'. "
                         f"Run run_disruption_scenarios.py first.")

    geo, shapes = read_network(net_path)
    print(f"Net file: {len(shapes)} edges | projection: " +
          (f"UTM zone {geo['zone']}" if geo else "unknown (map tiles disabled, plain drawing)"))

    # Edge order, time steps and the normal traffic come from the first scenario (identical in all of them).
    first = pd.read_csv(folders[0] / "traffic.csv", dtype={"edge_id": str},
                        usecols=["time", "edge_id", "traffic_before"])
    edge_ids = list(dict.fromkeys(first["edge_id"]))            # order of first appearance
    times = sorted(first["time"].unique().tolist())
    t_index = {t: i for i, t in enumerate(times)}
    e_index = {e: i for i, e in enumerate(edge_ids)}

    missing = [e for e in edge_ids if e not in shapes]
    if missing:
        raise SystemExit(f"{len(missing)} edges in the scenarios are missing from the net file, e.g. {missing[:3]}")

    baseline = (first.pivot(index="time", columns="edge_id", values="traffic_before")
                .reindex(index=times, columns=edge_ids).round().astype(int).values.tolist())

    edges = [{"id": e, "name": shapes[e][0], "pts": convert_points(shapes[e][1], geo)} for e in edge_ids]

    scenarios = []
    for folder in folders:
        with open(folder / "disruption.json") as f:
            dis = json.load(f)
        with open(folder / "summary.json") as f:
            summ = json.load(f)
        tr = pd.read_csv(folder / "traffic.csv", dtype={"edge_id": str}, usecols=["time", "edge_id", "added"])
        moved = tr[tr["added"] > 0]
        added = [[t_index[t], e_index[e], round(float(v), 2)]
                 for t, e, v in zip(moved["time"], moved["edge_id"], moved["added"])]
        w = (pd.read_csv(folder / "weights_used.csv", dtype={"edge_id": str})
             .set_index("edge_id").reindex(edge_ids)["weight"].round(3).tolist())
        scenarios.append({
            "id": int(dis["scenario_id"]),
            "edge": e_index[dis["edge_id"]],
            "start": int(dis["start_step"]), "end": int(dis["end_step"]),
            "diverted": round(float(summ["vehicles_diverted"]), 2),
            "unrouted": round(float(summ["vehicles_unrouted"]), 2),
            "top": [[e_index[x["edge_id"]], round(float(x["extra_vehicles"]), 2)]
                    for x in summ.get("top_receiving_edges", []) if x["edge_id"] in e_index],
            "weights": w,
            "added": added,
        })
        print(f"  {folder.name}: blocks {dis['edge_id']}, steps {dis['start_step']}-{dis['end_step']}, "
              f"{len(added)} traffic changes")

    data = {"geo": geo is not None, "edges": edges, "times": [int(t) for t in times],
            "baseline": baseline, "scenarios": scenarios}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "data.js"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("window.TRAFFIC_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"\nWrote {out_file} ({out_file.stat().st_size / 1e6:.1f} MB, {len(scenarios)} scenarios)")
    return out_dir


def serve(folder, port=8000):
    """Open the page through a tiny local web server (map tiles load more reliably than from file://)."""
    import functools
    import http.server
    import webbrowser
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        url = f"http://localhost:{port}/index.html"
        print(f"\nOpen {url}   (press Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="SUMO net file (.net.xml or .net.xml.gz)")
    ap.add_argument("--scenarios", default="disruptions", help="folder that contains scenario_XXX folders")
    ap.add_argument("--out", default="frontend", help="folder that contains index.html (data.js is written here)")
    ap.add_argument("--serve", action="store_true", help="after building, start a local server and open the page")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    out = build(a.net, a.scenarios, a.out)
    if a.serve:
        serve(out, a.port)
