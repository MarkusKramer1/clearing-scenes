#!/usr/bin/env python3
"""Pack each scene into one file the browser can read, for `web/`.

The viewer needs the survey cloud, the walkable surface, the graph and one
clearing schedule -- and it has to load them without a server, straight off the
file system, so `fetch()` is out. Each scene is written as a small JavaScript
file that assigns into `window.SCENES`, and `web/index.html` pulls in the one
the user picks.

Positions are quantised to uint16 over the scene's own extent. A 250 m site
quantises to 4 mm, against a voxel of 0.2 m: invisible, and it halves the
payload.

    python scripts/build_web.py
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from clearing import scene as scene_mod  # noqa: E402


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def quantise(xyz: np.ndarray, centre: np.ndarray) -> dict:
    """uint16 per axis over the scene's own extent, relative to `centre`.

    Any number of axes: the geometry here is (n, 3), the boundary line is
    (n, 2), and `centre` is sliced to match.
    """
    rel = xyz - centre
    lo, hi = rel.min(axis=0), rel.max(axis=0)
    scale = np.maximum(hi - lo, 1e-6) / 65535.0
    q = np.clip(np.rint((rel - lo) / scale), 0, 65535).astype(np.uint16)
    return {"q": b64(q.ravel()), "scale": scale.tolist(), "offset": lo.tolist()}


def fence(s, centre: np.ndarray) -> dict:
    """The scenario boundary, packed as a low fence for the viewer.

    The line is extruded into a low fence whose foot is exactly the line; why,
    and where the two heights come from, is on `Scene.boundary_fence`, which
    both this and the GIF renderer read so the two cannot disagree.
    """
    seg = np.asarray(s.boundary_seg, dtype=np.float64)
    if not len(seg):
        return None
    z0, z1 = s.boundary_fence

    b = s.doc.get("boundary") or {}
    return {
        **quantise(seg.reshape(-1, 2), centre[:2]),
        "fence": [z0 - float(centre[2]), z1 - float(centre[2])],
        "cellSize": s.boundary_cell_size,
        "stamp": b.get("stamp", ""),
        "cuts": b.get("cuts", 0),
        "heldLineM": b.get("held_line_m", 0.0),
        "excludedM2": b.get("walkable_dropped_m2", 0.0),
        "rimLengthM": b.get("rim_length_m", 0.0),
    }


def rle(mask: np.ndarray) -> list[int]:
    """Run lengths of a boolean mask, starting with a run of False."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    change = np.flatnonzero(np.diff(m)) + 1
    bounds = np.concatenate([[0], change, [m.size]])
    runs = np.diff(bounds).tolist()
    return ([0] + runs) if m[0] else runs


def pack(name: str, cloud_stride: int, surface_stride: int,
         schedule_key: str) -> dict:
    s = scene_mod.load(name, with_cloud=True)
    centre = s.cell_xyz.mean(axis=0)

    surf = s.cell_xyz[::surface_stride]
    cloud = s.cloud_xyz[::cloud_stride]

    out = {
        "name": s.name,
        "title": s.title,
        "centre": centre.tolist(),
        "cellSize": s.cell_size,
        "stats": {
            "cells": s.n_cells,
            "areaM2": round(s.doc["surface"]["area_m2"], 1),
            "vertices": s.n_nodes,
            "edges": int(s.edge_ij.shape[0]),
            "edgesShady": int(s.edge_shady.sum()),
            "routePoints": int(sum(len(r) for r in s.routes)),
            "range": s.doc["robot"]["detection_range_m"],
            "uncoverableM2": round(s.area(s.uncoverable), 1),
            # The surveyed ground this scenario gives up, measured on THIS
            # surface. E1's manifest carries its own figure for the same cut,
            # taken on its 0.4 m boundary raster; quoting that one next to a
            # walkable area computed at 0.2 m would be two rulers in one row.
            "excludedM2": round(float((s.doc.get("boundary") or {})
                                      .get("walkable_dropped_m2", 0.0)), 1),
        },
        "surface": {**quantise(surf, centre), "stride": surface_stride,
                    "height": [float(surf[:, 2].min()), float(surf[:, 2].max())]},
        "cloud": {**quantise(cloud, centre), "stride": cloud_stride,
                  "height": [float(cloud[:, 2].min()), float(cloud[:, 2].max())]},
        "nodes": b64((s.node_xyz - centre).astype(np.float32).ravel()),
        "edges": b64(s.edge_ij.astype(np.uint16).ravel()),
        "edgeShady": b64(s.edge_shady.astype(np.uint8)),
    }
    bnd = fence(s, centre)
    if bnd is not None:
        out["boundary"] = bnd

    # the path the robot walks along each edge -- one polyline per edge, packed
    # end to end with an offset table, the same shape as the CSR arrays above
    off = np.zeros(len(s.routes) + 1, dtype=np.uint32)
    off[1:] = np.cumsum([len(r) for r in s.routes])
    pts = np.concatenate([r for r in s.routes if len(r)]) if s.routes else None
    out["routes"] = {
        **quantise(pts, centre),
        "offsets": b64(off),
        "seconds": b64(np.array([s.travel_seconds[i, j]
                                 for i, j in s.edge_ij], np.float32)),
        "metres": b64(np.array([s.travel_metres[i, j]
                                for i, j in s.edge_ij], np.float32)),
    }

    # the detection sets, as run lengths over the drawn surface points -- what
    # the viewer paints when a vertex is selected or a schedule step is shown
    drawn = np.zeros(s.n_cells, dtype=bool)
    drawn[::surface_stride] = True
    index = -np.ones(s.n_cells, dtype=np.int64)
    index[drawn] = np.arange(drawn.sum())
    dsets = []
    for d in s.detection:
        m = np.zeros(drawn.sum(), dtype=bool)
        k = index[d]
        m[k[k >= 0]] = True
        dsets.append(rle(m))
    out["detection"] = dsets

    ex = REPO / "examples" / f"{name}.clearing.yaml"
    if ex.exists():
        doc = yaml.safe_load(ex.read_text())
        run = next((r for r in doc["runs"] if r["key"] == schedule_key), None)
        if run is not None:
            # The per-step contamination is re-derived by the verifier rather
            # than by the viewer: a flood over 300 000 cells is not something to
            # ask a browser for, and shipping what actually got verified is the
            # only way the picture cannot disagree with the metrics beside it.
            from clearing import verify

            steps = [tuple(st["vertices"]) for st in run["steps_list"]]
            sched = verify.Schedule(steps, run["key"])
            res = verify.propagate(s, sched)
            out["schedule"] = {
                "key": run["key"],
                "steps": [list(st) for st in steps],
                "contaminated": [rle(c[drawn]) for c in res.contaminated],
                "metrics": {k: run[k] for k in
                            ("steps", "team_peak", "team_median", "fleet",
                             "cleared", "cleared_except_uncoverable",
                             "residual_m2", "recontamination_events",
                             "makespan_s")
                            if k in run},
                "roster": roster_payload(s, sched, centre),
            }
    return out


def roster_payload(s, schedule, centre: np.ndarray) -> dict:
    """The same schedule with a fleet on it: who stands where, and what it walks.

    A step is a SET of vertices, and a set cannot be animated: it says which
    places are held, not which machine went where. `clearing.roster` matches a
    fleet across the steps and hands back one leg per occupied vertex; this
    ships those legs, plus the polyline each one is walked along, so the viewer
    can follow robot 7 through the plan rather than watch dots blink.

    Legs are deduplicated by their (from, to) pair -- a fleet walks the same
    stretch again and again -- and the polylines are quantised like every other
    geometry here.
    """
    from clearing import roster as roster_mod

    ros = roster_mod.assign(s, schedule)
    pairs = ros.pairs()
    paths, worst = roster_mod.walked_paths(s, pairs)
    index = {ij: k for k, ij in enumerate(pairs)}

    poly = [paths.get(ij, np.zeros((0, 3), np.float32)) for ij in pairs]
    off = np.zeros(len(poly) + 1, dtype=np.uint32)
    off[1:] = np.cumsum([len(p) for p in poly])
    pts = (np.concatenate([p for p in poly if len(p)])
           if any(len(p) for p in poly) else np.zeros((1, 3), np.float32))

    def secs(x):
        return round(float(x), 2) if np.isfinite(x) else None

    moves = [[{"r": m.robot, "v": m.to, "f": m.frm,
               "p": -1 if m.stays else index[(m.frm, m.to)],
               "s": secs(m.seconds), "m": secs(m.metres),
               **({"e": 1} if m.entered else {})}
              for m in step] for step in ros.steps]

    print(f"      roster: {ros.n_robots} robots, {sum(len(x) for x in moves)} legs, "
          f"{len(pairs)} distinct paths, {int(off[-1]):,} points "
          f"(length error <= {worst:.3f} m)", flush=True)

    return {
        "nRobots": ros.n_robots,
        "entry": ros.entry,
        "stepSeconds": [round(x, 2) for x in ros.step_seconds],
        "moves": moves,
        "positions": ros.positions,
        "paths": {**quantise(pts, centre), "offsets": b64(off)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--cloud-stride", type=int, default=2)
    ap.add_argument("--surface-stride", type=int, default=3)
    ap.add_argument("--schedule", default="kolling")
    a = ap.parse_args()

    data = REPO / "web" / "data"
    data.mkdir(parents=True, exist_ok=True)
    index = []
    for name in (a.scenes or scene_mod.available()):
        print(f"[web] {name}", flush=True)
        payload = pack(name, a.cloud_stride, a.surface_stride, a.schedule)
        path = data / f"{name}.js"
        path.write_text("window.SCENES=window.SCENES||{};window.SCENES["
                        + json.dumps(name) + "]=" + json.dumps(payload) + ";\n")
        mb = path.stat().st_size / 1e6
        index.append({"name": name, "title": payload["title"],
                      "file": f"data/{name}.js", "mb": round(mb, 2),
                      "stats": payload["stats"]})
        print(f"      {mb:.1f} MB", flush=True)

    (data / "index.js").write_text(
        "window.SCENE_INDEX=" + json.dumps(index, indent=1) + ";\n")
    print(f"[web] {len(index)} scenes, "
          f"{sum(i['mb'] for i in index):.1f} MB total")


if __name__ == "__main__":
    main()
