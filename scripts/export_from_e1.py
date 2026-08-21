#!/usr/bin/env python3
"""Export one scenario per scene out of the `e1-clearing-graph` pipeline.

This is the ONLY script in this repository that needs the heavy machinery:
the survey clouds, the occupancy volumes and the raycaster all live in
`e1-clearing-graph` and on its external SSD. Everything else here reads the
files this script writes.

Per scene it writes two files:

    scenes/<scene>.yaml            the scenario -- readable, hand-editable
    scenes/geometry/<scene>.npz    the bulk arrays the YAML points at

The split is deliberate. A detection set is a list of ten thousand cell
indices; ninety of them do not belong in a file a person is meant to read.
The YAML carries the graph (vertices, guard-region edges, travel times) and
every number needed to understand the scenario; the npz carries the cell-level
payload that the verifier and the viewer consume.

GROUND ONLY
-----------
E1's graph is bipartite over two platform families. This export keeps the
ground half: ground vertices, ground-ground guard-region edges, and the ground
roadmap. That subgraph is exactly the setting of Kolling et al. 2010 -- one
kind of searcher, detection sets from line of sight at a fixed sensor height,
guard regions as the edges -- which is why the baselines in `clearing/kolling.py`
apply to it unchanged. The ground half does not depend on the UAV altitude:
the two families are sampled independently, and `carved_huav10/15/20` carry
byte-identical ground vertex sets. `--h-uav` therefore only picks which of the
identical files to read from.

Usage
-----
    python scripts/export_from_e1.py --e1-root ~/e1-clearing-graph
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: Human-readable names; the slugs are the Oxford Spires sequence names.
TITLES = {
    "blenheim-palace": "Blenheim Palace",
    "bodleian-library": "Bodleian Library",
    "christ-church": "Christ Church",
    "hb-allen-centre": "HB Allen Centre",
    "keble-college": "Keble College",
    "observatory-quarter": "Observatory Quarter",
}


# ---------------------------------------------------------------------------
# reading the E1 side
# ---------------------------------------------------------------------------


def e1_imports(e1_root: Path):
    """Put E1's `src` on the path and hand back what this script uses."""
    src = str(e1_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from e1.traversal import _lattice_edges  # noqa: PLC0415
    return _lattice_edges


def evader_edges(cells, height, d_v, h_climb, lattice_edges):
    """8-connected, height-aware adjacency over the walkable cells.

    The same lattice E1 moves its ground robot on. Height-awareness matters
    because the surface is per-voxel: a column under an arcade carries the
    ground and the terrace above it, and plain (i, j) adjacency would let an
    evader step through the vault.
    """
    a, b, _ = lattice_edges(cells[:, 0].astype(np.int64),
                            cells[:, 1].astype(np.int64),
                            np.asarray(height, dtype=np.float64),
                            float(d_v), float(h_climb))
    return a.astype(np.int32), b.astype(np.int32)


def slices(offsets, values, keep):
    """Re-pack a CSR-style (offsets, values) pair, keeping rows `keep`."""
    out = [values[offsets[i]:offsets[i + 1]] for i in keep]
    off = np.zeros(len(out) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(v) for v in out])
    flat = np.concatenate(out) if out else np.zeros(0, dtype=np.int32)
    return off, flat.astype(np.int32)


# ---------------------------------------------------------------------------
# writing the YAML by hand
# ---------------------------------------------------------------------------


def fmt(x, nd=3):
    """A float without the trailing noise of repr()."""
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def write_yaml(path: Path, scene: dict) -> None:
    L: list[str] = []
    add = L.append

    add(f"# {scene['title']} -- ground-robot clearing scenario")
    add("#")
    add("# Vertices are sampled sensor positions on the walkable surface. An edge")
    add("# (i, j) exists where a robot at j can watch part of the rim of what a")
    add("# robot at i sees -- the guard region of Kolling et al. 2010. `routes`")
    add("# gives the ground robot's travel time between every pair of vertices.")
    add("#")
    add("# Regenerate with scripts/export_from_e1.py; see docs/scene-format.md.")
    add("")
    add(f"scene: {scene['scene']}")
    add(f"title: {scene['title']}")
    add("")
    add("source:")
    for k, v in scene["source"].items():
        add(f"  {k}: {v}")
    add("")
    add("robot:")
    for k, v in scene["robot"].items():
        add(f"  {k}: {v}")
    add("")
    add("target:")
    for k, v in scene["target"].items():
        add(f"  {k}: {v}")
    add("")

    s = scene["surface"]
    add("surface:")
    add(f"  geometry_file: {s['geometry_file']}")
    add(f"  cell_size_m: {fmt(s['cell_size_m'])}")
    add(f"  cells: {s['cells']}")
    add(f"  area_m2: {fmt(s['area_m2'], 2)}")
    add(f"  bbox_min: [{', '.join(fmt(v, 2) for v in s['bbox_min'])}]")
    add(f"  bbox_max: [{', '.join(fmt(v, 2) for v in s['bbox_max'])}]")
    add(f"  max_step_m: {fmt(s['max_step_m'])}          "
        "# height a neighbouring cell may differ by")
    add(f"  evader_edges: {s['evader_edges']}       "
        "# 8-connected under that rule; recomputed, not stored")
    add(f"  uncoverable_cells: {s['uncoverable_cells']}   "
        "# seen by no vertex, at any team size")
    add(f"  uncoverable_area_m2: {fmt(s['uncoverable_area_m2'], 2)}")
    add("")

    g = scene["graph"]
    add("graph:")
    add(f"  vertices: {len(g['nodes'])}")
    add(f"  edges: {len(g['edges'])}")
    add(f"  edges_regular: {g['n_regular']}")
    add(f"  edges_shady: {g['n_shady']}")
    add("")
    add("  # id, position of the sensor [m], area the vertex sees, size of the rim")
    add("  vertices_list:")
    for n in g["nodes"]:
        add(f"    - {{id: {n['id']:>3}, "
            f"xyz: [{fmt(n['xyz'][0], 2):>9}, {fmt(n['xyz'][1], 2):>9}, "
            f"{fmt(n['xyz'][2], 2):>7}], "
            f"detection_m2: {fmt(n['detection_m2'], 1):>8}, "
            f"boundary_cells: {n['boundary_cells']:>6}}}")
    add("")
    add("  # i, j, whether the guard region is contained in another (shady) or")
    add("  # not (regular), and how big it is")
    add("  edges_list:")
    for e in g["edges"]:
        add(f"    - {{i: {e['i']:>3}, j: {e['j']:>3}, type: {e['type']:<7}, "
            f"guard_cells: {e['guard_cells']:>6}, "
            f"guard_m2: {fmt(e['guard_m2'], 2):>8}}}")
    add("")

    r = scene["routes"]
    add("routes:")
    add("  # Shortest path for the ground robot over the walkable surface,")
    add("  # between every pair of vertices it can reach. Lattice paths at the")
    add("  # cell size, so they overestimate a smoothed path by a few per cent.")
    add(f"  reachable_pairs: {len(r['values'])}")
    add(f"  unreachable_pairs: {r['unreachable_pairs']}")
    add(f"  speed_m_s: {fmt(r['speed_m_s'])}")
    add("  columns: [i, j, distance_m, time_s]")
    add("  values:")
    for i, j, d, t in r["values"]:
        add(f"    - [{i:>3}, {j:>3}, {fmt(d, 1):>7}, {fmt(t, 1):>7}]")
    add("")

    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------
# one scene
# ---------------------------------------------------------------------------


def export_scene(name: str, e1_root: Path, data_root: Path, out: Path,
                 variant: str, h_uav: float, R: float, seed: int,
                 cloud_points: int, lattice_edges) -> dict:
    derived = data_root / "derived" / name
    key = f"{variant}_huav{h_uav:g}_R{R:g}_seed{seed}"
    gpath = derived / "graphs" / f"{key}.npz"
    tpath = derived / "graphs" / f"{key}_traversal.npz"
    wpath = derived / f"walkable_{variant}.npz"
    for p in (gpath, tpath, wpath):
        if not p.exists():
            raise FileNotFoundError(p)

    params = json.loads((e1_root / "outputs" / "scenes" / name
                         / "stage6_result.json").read_text())["params"]
    d_v = float(params["d_v"])
    h_climb = max(float(params.get("h_climb", 0.25)), float(params["h_step"]))

    w = np.load(wpath)
    cells, height = w["cells"], w["height"]
    min_corner, n_cells = w["min_corner"], int(cells.shape[0])
    xyz = np.column_stack([min_corner[0] + (cells[:, 0] + 0.5) * d_v,
                           min_corner[1] + (cells[:, 1] + 0.5) * d_v,
                           height]).astype(np.float32)

    z = np.load(gpath)
    fam = z["node_family"]
    ground = np.flatnonzero(fam == 0)            # 0 = ground, 1 = air
    remap = -np.ones(fam.size, dtype=np.int64)
    remap[ground] = np.arange(ground.size)

    dset_off, dset_cells = slices(z["dset_offsets"], z["dset_cells"], ground)
    bnd_off, bnd_cells = slices(z["bnd_offsets"], z["bnd_cells"], ground)
    node_pos = z["node_pos"][ground].astype(np.float32)
    node_row = z["node_row"][ground].astype(np.int32)

    eij = z["edge_ij"]
    both = (fam[eij[:, 0]] == 0) & (fam[eij[:, 1]] == 0)
    esel = np.flatnonzero(both)
    edge_ij = remap[eij[esel]].astype(np.int32)
    edge_kind = z["edge_kind"][esel]              # 0 = regular, 1 = shady
    guard_off, guard_cells = slices(z["guard_offsets"], z["guard_cells"], esel)

    # what no ground vertex sees, at any team size
    coverable = np.zeros(n_cells, dtype=bool)
    coverable[dset_cells] = True
    uncoverable = np.flatnonzero(~coverable).astype(np.int32)

    # The roadmap indexes its matrices by clearing-graph node id. Asserted
    # rather than assumed: a silent mismatch here would attach every travel
    # time to the wrong pair of vertices and nothing downstream would notice.
    t = np.load(tpath)
    if not np.array_equal(t["ugv_node_ids"], ground.astype(t["ugv_node_ids"].dtype)):
        raise ValueError(f"{tpath.name}: roadmap vertex order does not match "
                         "the ground vertices of the graph")
    sec = t["ugv_seconds"].astype(np.float32)
    met = t["ugv_metres"].astype(np.float32)
    comp = t["ugv_component"].astype(np.int32)

    ea, eb = evader_edges(cells, height, d_v, h_climb, lattice_edges)

    cloud = np.load(e1_root / "outputs" / "scenes" / name
                    / "cloud_decimated.npy", mmap_mode="r")
    stride = max(1, cloud.shape[0] // cloud_points)
    cloud = np.ascontiguousarray(cloud[::stride]).astype(np.float32)

    np.savez_compressed(
        out / "scenes" / "geometry" / f"{name}.npz",
        cell_xyz=xyz, cell_ij=cells[:, :2].astype(np.int32),
        cell_size=np.float32(d_v),
        node_xyz=node_pos, node_cell=node_row,
        detection_offsets=dset_off, detection_cells=dset_cells,
        rim_offsets=bnd_off, rim_cells=bnd_cells,
        edge_ij=edge_ij, edge_shady=edge_kind.astype(np.uint8),
        guard_offsets=guard_off, guard_cells=guard_cells,
        travel_seconds=sec, travel_metres=met, travel_component=comp,
        route_points=t["ugv_path_pts"].astype(np.float32),
        route_offsets=t["ugv_path_off"].astype(np.int64),
        route_ij=remap[t["ugv_path_ab"]].astype(np.int32),
        uncoverable=uncoverable,
        cloud_xyz=cloud,
    )

    cell_area = d_v * d_v
    finite = np.isfinite(sec)
    iu = np.triu_indices(sec.shape[0], k=1)
    pairs = [(int(i), int(j), float(met[i, j]), float(sec[i, j]))
             for i, j in zip(*iu) if finite[i, j]]

    doc = {
        "scene": name,
        "title": TITLES.get(name, name),
        "source": {
            "dataset": "Oxford Spires Dataset (Tao et al., IJRR 2025)",
            "survey": "terrestrial laser scan, merged cloud + individual E57 setups",
            "pipeline": "e1-clearing-graph, stages 2-6",
            "occupancy": f"O-{variant} (unobserved space blocks)",
            "graph_file": key,
            "note": ("the ground half of E1's two-platform graph; identical for "
                     "every h_uav, which only selects the file"),
        },
        "robot": {
            "kind": "ground",
            "sensor_height_m": params["h_ugv"],
            "detection_range_m": R,
            "speed_m_s": params["v_ugv"],
        },
        "target": {
            "height_m": params["h_t"],
            "radius_m": params["r_t"],
            "visibility": "all 5 rays to the target cylinder must pass",
        },
        "surface": {
            "geometry_file": f"geometry/{name}.npz",
            "cell_size_m": d_v,
            "cells": n_cells,
            "area_m2": n_cells * cell_area,
            "bbox_min": xyz.min(axis=0).tolist(),
            "bbox_max": xyz.max(axis=0).tolist(),
            "max_step_m": h_climb,
            "evader_edges": int(ea.size),
            "uncoverable_cells": int(uncoverable.size),
            "uncoverable_area_m2": uncoverable.size * cell_area,
        },
        "graph": {
            "nodes": [
                {"id": i, "xyz": node_pos[i].tolist(),
                 "detection_m2": (dset_off[i + 1] - dset_off[i]) * cell_area,
                 "boundary_cells": int(bnd_off[i + 1] - bnd_off[i])}
                for i in range(ground.size)],
            "edges": [
                {"i": int(edge_ij[k, 0]), "j": int(edge_ij[k, 1]),
                 "type": "shady" if edge_kind[k] else "regular",
                 "guard_cells": int(guard_off[k + 1] - guard_off[k]),
                 "guard_m2": (guard_off[k + 1] - guard_off[k]) * cell_area}
                for k in range(edge_ij.shape[0])],
            "n_regular": int((edge_kind == 0).sum()),
            "n_shady": int((edge_kind != 0).sum()),
        },
        "routes": {
            "values": pairs,
            "unreachable_pairs": int(len(iu[0]) - len(pairs)),
            "speed_m_s": params["v_ugv"],
        },
    }
    write_yaml(out / "scenes" / f"{name}.yaml", doc)
    return {
        "scene": name, "title": doc["title"],
        "cells": n_cells, "area_m2": round(n_cells * cell_area, 1),
        "vertices": int(ground.size), "edges": int(edge_ij.shape[0]),
        "cloud_points": int(cloud.shape[0]),
        "uncoverable_m2": round(uncoverable.size * cell_area, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-root", type=Path, default=Path.home() / "e1-clearing-graph")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/media/kramer/Extreme SSD/e1-data"))
    ap.add_argument("--out", type=Path, default=REPO)
    ap.add_argument("--scenes", nargs="*", default=sorted(TITLES))
    ap.add_argument("--variant", default="carved")
    ap.add_argument("--h-uav", type=float, default=20.0)
    ap.add_argument("--R", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cloud-points", type=int, default=250_000)
    a = ap.parse_args()

    lattice_edges = e1_imports(a.e1_root)
    index = []
    for name in a.scenes:
        print(f"[export] {name}", flush=True)
        info = export_scene(name, a.e1_root, a.data_root, a.out, a.variant,
                            a.h_uav, a.R, a.seed, a.cloud_points, lattice_edges)
        index.append(info)
        print(f"         {info['vertices']} vertices, {info['edges']} edges, "
              f"{info['area_m2']:,.0f} m2", flush=True)

    (a.out / "scenes" / "index.json").write_text(
        json.dumps({"config": {"variant": a.variant, "R": a.R, "seed": a.seed},
                    "scenes": index}, indent=2) + "\n")
    print(f"[export] wrote {len(index)} scenes")


if __name__ == "__main__":
    main()
