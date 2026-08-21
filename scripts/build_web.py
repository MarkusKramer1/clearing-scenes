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
    """uint16 per axis over the scene's own extent, relative to `centre`."""
    rel = xyz - centre
    lo, hi = rel.min(axis=0), rel.max(axis=0)
    scale = np.maximum(hi - lo, 1e-6) / 65535.0
    q = np.clip(np.rint((rel - lo) / scale), 0, 65535).astype(np.uint16)
    return {"q": b64(q.ravel()), "scale": scale.tolist(), "offset": lo.tolist()}


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
            "range": s.doc["robot"]["detection_range_m"],
            "uncoverableM2": round(s.area(s.uncoverable), 1),
        },
        "surface": {**quantise(surf, centre), "stride": surface_stride,
                    "height": [float(surf[:, 2].min()), float(surf[:, 2].max())]},
        "cloud": {**quantise(cloud, centre), "stride": cloud_stride,
                  "height": [float(cloud[:, 2].min()), float(cloud[:, 2].max())]},
        "nodes": b64((s.node_xyz - centre).astype(np.float32).ravel()),
        "edges": b64(s.edge_ij.astype(np.uint16).ravel()),
        "edgeShady": b64(s.edge_shady.astype(np.uint8)),
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
            res = verify.propagate(s, verify.Schedule(steps, run["key"]))
            out["schedule"] = {
                "key": run["key"],
                "steps": [list(st) for st in steps],
                "contaminated": [rle(c[drawn]) for c in res.contaminated],
                "metrics": {k: run[k] for k in
                            ("steps", "team_peak", "team_median", "cleared",
                             "cleared_except_uncoverable", "residual_m2",
                             "recontamination_events", "makespan_s")
                            if k in run},
            }
    return out


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
