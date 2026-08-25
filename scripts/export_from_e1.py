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

THE SCENARIO BOUNDARY
---------------------
A survey is not a scenario. The Oxford Spires clouds run off down service
avenues and out through gateways, and a search problem posed on all of it is
posed on ground nobody was asked to clear -- and, worse, on ground an evader
can walk in from. E1 answers that with a hand-drawn boundary per scene
(`configs/site_boundary.yaml` there): the operator draws the lines where the
site stops, and everything from Stage 4 on runs on the surface those lines
retain. The graphs this script reads were built that way, so the vertices,
the detection sets, the guard regions and the roadmap are already bounded.

WHAT THIS SCRIPT HAS TO DO ABOUT IT IS NOT OPTIONAL. Every array in the graph
npz -- `node_row`, `dset_cells`, `bnd_cells`, `guard_cells` -- is a ROW INDEX
into the retained surface. `walkable_<variant>.npz` on the SSD is still the
FULL one, because the boundary cuts it at read time and never rewrites it. So
reading the graph against the full walkable file does not fail: it silently
reads the wrong cells, and every detection set lands somewhere else on the
site. The surface here is therefore cut with the same call the E1 stages use,
`boundary_apply.walkable_surface`/`inside`, and the run refuses to start if
E1's own Stage 6 result disagrees with the boundary that is approved now.

The boundary line itself is exported alongside the surface, so the viewer can
draw where the scenario stops rather than leave the cut looking like the edge
of the data. The survey cloud is NOT cut: it is context, the boundary runs
along and through the buildings in it, and a cloud clipped to the line would
hide the very structure the line was drawn against.

THE CUT IS E1'S, EXACTLY, INCLUDING WHERE IT IS A CELL OFF. `boundary_apply`
decides a walkable cell by `floor(i * d_v / d_a)`, and at an exact face of the
0.4 m boundary raster that is a floating-point coin flip -- `728 * 0.2 / 0.4`
is 363.999999999999994 in double, so that cell is kept by the row next door.
Across the six scenes it moves 383 cells, 0.16 to 6.72 m² per scene against
1 280 to 11 469 m² retained. They are kept as E1 keeps them and not corrected
here: this script's claim is that the scenario is the approved one, and a
scenario that is nearly the approved one is worth less than one that is
exactly it. `tests/test_boundary.py` measures the seam and bounds it to the
line. The fix belongs in `boundary_apply.inside`, upstream, where correcting
it would move all six stamps and rerun Stages 4-6.

`--no-boundary` exports the full surveyed surface instead. It is only correct
against a graph built the same way (E1's `--no-boundary`), and the guard below
will say so.

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
    from e1 import boundary_apply  # noqa: PLC0415
    from e1.traversal import _lattice_edges  # noqa: PLC0415
    return _lattice_edges, boundary_apply


def retained(name: str, ba, result6: dict, cells, min_corner, d_v: float,
             *, bounded: bool):
    """Which walkable cells the approved boundary keeps, and the note about it.

    Returns `(keep, block)`. `keep` is a boolean mask over the rows of the full
    walkable surface; `block` is what goes in the YAML.

    THE GUARD IS THE POINT OF THE FUNCTION. `keep` decides how the surface is
    renumbered, and every row index in the graph npz was written against one
    particular renumbering. `boundary_apply.check` compares the boundary E1's
    Stage 6 recorded against the one approved now, and anything but agreement
    stops the export -- because the failure it prevents is not a crash but a
    scenario whose detection sets are attached to the wrong ground.
    """
    disagreement = ba.check(result6, name)
    stamp = ba.stamp(name)

    if not bounded:
        if not disagreement:
            raise SystemExit(
                f"{name}: --no-boundary, but E1's Stage 6 ran against the "
                f"approved boundary {stamp}. Its row indices are into the cut "
                f"surface; exporting them against the full one would attach "
                f"every detection set to the wrong cells. Rerun E1 Stage 6 "
                f"with --no-boundary, or drop --no-boundary here.")
        return np.ones(len(cells), dtype=bool), {
            "bounded": False, "excluded_m2": 0.0,
            "note": "the whole surveyed surface; no scenario boundary applied",
        }

    if disagreement:
        raise SystemExit(
            f"{name}: the graph on the SSD was {disagreement}.\n"
            f"Rerun E1 Stages 4-6 against the approved boundary, or pass "
            f"--no-boundary to both. Row indices into the walkable surface are "
            f"not comparable across a boundary change, and mixing them reads "
            f"the wrong cells without failing.")

    keep = ba.inside(name, cells.astype(np.int64), min_corner, d_v)
    if keep is None:
        raise SystemExit(
            f"{name}: no approved scenario boundary. E1 publishes one per "
            f"scene under outputs/site_boundary/approved/; approve it there, "
            f"or pass --no-boundary to export the full surveyed surface.")

    rim = ba.rim_segments(name)
    st = ba.status(name)
    return keep, {
        "bounded": True,
        "stamp": stamp,
        "source": "e1-clearing-graph configs/site_boundary.yaml, drawn by hand",
        "approved": st.get("approved", ""),
        "cuts": int(st.get("n_cuts", 0)),
        "held_line_m": float(st.get("held_line_m", 0.0)),
        "retained_m2": float(st.get("retained_m2", 0.0)),
        "excluded_m2": float(st.get("excluded_m2", 0.0)),
        "rim_segments": int(len(rim["seg"])) if rim else 0,
        "rim_length_m": float(len(rim["seg"]) * rim["d_a"]) if rim else 0.0,
        "_rim": rim,
    }


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


def simplify(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iteratively. Returns the indices to keep.

    A lattice path is mostly long straight runs with a corner every so often,
    so this takes a 500-point walk down to a couple of dozen points. At an
    epsilon below the cell size it is visually lossless, and the lengths quoted
    in the YAML come from the FULL path, never from the simplified one.
    """
    n = len(pts)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = pts[j] - pts[i]
        length = float(np.linalg.norm(seg))
        rel = pts[i + 1:j] - pts[i]
        if length < 1e-9:
            d = np.linalg.norm(rel, axis=1)
        else:
            u = seg / length
            d = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        m = int(np.argmax(d))
        if d[m] > eps:
            k = i + 1 + m
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return np.flatnonzero(keep)


def edge_routes(cells, height, d_v, h_climb, xyz, node_cell, edge_ij,
                travel_metres, lattice_edges, eps=0.12, chunk=16):
    """The path the robot actually walks along every edge of the graph.

    This is the point of the layer: an edge of the visibility graph is a
    statement about what two positions can see, and the ground between them may
    be a wall. Drawing the edge as a straight segment says the opposite of what
    the travel time means. So each edge gets its true shortest path over the
    walkable lattice -- the same lattice, the same step rule -- reconstructed
    from a Dijkstra predecessor tree and then simplified for drawing.

    The reconstructed length is checked against the travel matrix E1 wrote. A
    mismatch would mean this lattice and E1's have drifted apart, and it fails
    loudly rather than shipping routes that disagree with their own times.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra

    a, b, w = lattice_edges(cells[:, 0].astype(np.int64),
                            cells[:, 1].astype(np.int64),
                            np.asarray(height, dtype=np.float64),
                            float(d_v), float(h_climb))
    n = cells.shape[0]
    g = sparse.coo_matrix((w, (a, b)), shape=(n, n)).tocsr()
    g = g + g.T

    by_src: dict[int, list[int]] = {}
    for k, (i, _j) in enumerate(edge_ij):
        by_src.setdefault(int(i), []).append(k)

    routes: list[np.ndarray] = [np.zeros((0, 3), np.float32)] * edge_ij.shape[0]
    worst = 0.0
    srcs = sorted(by_src)
    for c0 in range(0, len(srcs), chunk):
        block = srcs[c0:c0 + chunk]
        _, pred = dijkstra(g, indices=node_cell[block], directed=False,
                           return_predecessors=True)
        for row, src in enumerate(block):
            p = pred[row]
            for k in by_src[src]:
                j = int(edge_ij[k, 1])
                path, q = [int(node_cell[j])], int(node_cell[j])
                while q != int(node_cell[src]):
                    q = int(p[q])
                    if q < 0:
                        path = []
                        break
                    path.append(q)
                if not path:
                    continue
                pts = xyz[path[::-1]].astype(np.float64)
                walked = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                worst = max(worst, abs(walked - float(travel_metres[src, j])))
                routes[k] = pts[simplify(pts, eps)].astype(np.float32)

    if worst > 0.05:
        raise ValueError(f"route lengths disagree with the travel matrix by up "
                         f"to {worst:.3f} m; the lattices have drifted apart")

    off = np.zeros(len(routes) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(r) for r in routes])
    pts = (np.concatenate(routes) if len(routes)
           else np.zeros((0, 3), np.float32))
    return off, pts.astype(np.float32), worst


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
    add("# gives the ground robot's travel time between every pair of vertices;")
    add("# the path it actually walks along each edge is in the npz beside it.")
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

    b = scene["boundary"]
    add("boundary:")
    add("  # Where the scenario stops. Drawn by hand in e1-clearing-graph")
    add("  # (configs/site_boundary.yaml) and approved there; the surface,")
    add("  # the vertices, the guard regions and the routes below are all")
    add("  # what survives it. The evader cannot leave it and no robot has to")
    add("  # watch across it: the drawn lines are held, not open frontier.")
    if not b["bounded"]:
        add("  bounded: false")
        add(f"  note: {b['note']}")
    else:
        add(f"  bounded: true")
        add(f"  stamp: {b['stamp']}            "
            "# hash of the drawn lines; matches E1's approval")
        add(f"  approved: {b['approved']}")
        add(f"  source: {b['source']}")
        add(f"  cuts: {b['cuts']}                  "
            "# hand-drawn lines across the openings")
        add(f"  held_line_m: {fmt(b['held_line_m'], 1)}          "
            "# how much of the perimeter those lines are")
        add(f"  cells_dropped: {b['cells_dropped']}       "
            "# walkable cells outside it, removed before anything was indexed")
        add(f"  walkable_dropped_m2: {fmt(b['walkable_dropped_m2'], 1)}   "
            "# what that is in area: surveyed ground this scenario gives up")
        add(f"  excluded_m2: {fmt(b['excluded_m2'], 1)}           "
            "# E1's own figure, measured on its 0.4 m boundary raster rather")
        add("                              "
            "# than on this 0.2 m walkable surface, so the two differ slightly")
        add(f"  rim_segments: {b['rim_segments']}")
        add(f"  rim_length_m: {fmt(b['rim_length_m'], 1)}       "
            "# the whole outline, in the npz as boundary_seg")
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
    add(f"  polylines_for_graph_edges: {r['polylines']}   "
        "# the walked path per edge, in the npz")
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
                 cloud_points: int, lattice_edges, ba, bounded: bool) -> dict:
    derived = data_root / "derived" / name
    key = f"{variant}_huav{h_uav:g}_R{R:g}_seed{seed}"
    gpath = derived / "graphs" / f"{key}.npz"
    tpath = derived / "graphs" / f"{key}_traversal.npz"
    wpath = derived / f"walkable_{variant}.npz"
    for p in (gpath, tpath, wpath):
        if not p.exists():
            raise FileNotFoundError(p)

    result6 = json.loads((e1_root / "outputs" / "scenes" / name
                          / "stage6_result.json").read_text())
    params = result6["params"]
    d_v = float(params["d_v"])
    h_climb = max(float(params.get("h_climb", 0.25)), float(params["h_step"]))

    w = np.load(wpath)
    cells, height = w["cells"], w["height"]
    min_corner = w["min_corner"]

    # The scenario boundary, before anything is indexed. Cutting the surface
    # renumbers its rows, and every row index read below -- node_row, the
    # detection sets, the rims, the guard regions -- was written against the
    # cut numbering.
    keep, boundary = retained(name, ba, result6, cells, min_corner, d_v,
                              bounded=bounded)
    cells, height = cells[keep], height[keep]
    n_cells = int(cells.shape[0])
    boundary["cells_dropped"] = int((~keep).sum())
    boundary["walkable_dropped_m2"] = boundary["cells_dropped"] * d_v * d_v

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

    # THE CHECK THAT A BOUNDARY MISMATCH CANNOT SURVIVE. `node_pos` is a world
    # position, written by E1 beside `node_row` and not derived from it, so the
    # two agree only if this surface is the surface the graph was built on. A
    # graph cut differently would still index in range and still load; it would
    # just put every vertex somewhere else on the site, and nothing downstream
    # would notice. Verified here, once, against the cheapest possible witness.
    if node_row.size:
        if int(node_row.max()) >= n_cells:
            raise ValueError(
                f"{gpath.name}: vertex row {int(node_row.max())} is past the "
                f"end of the {n_cells:,}-cell surface it is meant to index -- "
                f"the graph and the surface were cut differently")
        off = np.abs(node_pos[:, :2] - xyz[node_row][:, :2]).max()
        if off > 0.5 * d_v:
            raise ValueError(
                f"{gpath.name}: vertices sit up to {off:.2f} m from the cells "
                f"they index. The graph and the walkable surface were cut to "
                f"different boundaries; rerun E1 Stages 4-6, or re-approve.")

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
    route_off, route_pts, route_err = edge_routes(
        cells, height, d_v, h_climb, xyz, node_row, edge_ij, met, lattice_edges)

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
        edge_route_offsets=route_off, edge_route_points=route_pts,
        uncoverable=uncoverable,
        cloud_xyz=cloud,
        # The line the scenario stops at, as the faces between a retained cell
        # and one that is not -- E1's own rendering of it, at the boundary
        # raster's resolution rather than a smoothed polygon, so it meets the
        # surface it was cut from instead of floating a few centimetres off it.
        boundary_seg=(np.asarray(boundary["_rim"]["seg"], np.float32)
                      if boundary.get("_rim") else np.zeros((0, 2, 2), np.float32)),
        boundary_cell_size=np.float32(boundary["_rim"]["d_a"]
                                      if boundary.get("_rim") else 0.0),
    )
    boundary.pop("_rim", None)

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
        "boundary": boundary,
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
            "polylines": int(sum(1 for k in range(edge_ij.shape[0])
                                 if route_off[k + 1] > route_off[k])),
        },
    }
    write_yaml(out / "scenes" / f"{name}.yaml", doc)
    return {
        "scene": name, "title": doc["title"], "boundary": boundary,
        "cells": n_cells, "area_m2": round(n_cells * cell_area, 1),
        "vertices": int(ground.size), "edges": int(edge_ij.shape[0]),
        "cloud_points": int(cloud.shape[0]),
        "route_points": int(route_pts.shape[0]),
        "route_length_error_m": round(route_err, 4),
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
    ap.add_argument("--no-boundary", action="store_true",
                    help="export the full surveyed surface. Only correct "
                         "against graphs E1 also built unbounded.")
    a = ap.parse_args()

    lattice_edges, ba = e1_imports(a.e1_root)
    index = []
    for name in a.scenes:
        print(f"[export] {name}", flush=True)
        info = export_scene(name, a.e1_root, a.data_root, a.out, a.variant,
                            a.h_uav, a.R, a.seed, a.cloud_points, lattice_edges,
                            ba, not a.no_boundary)
        index.append(info)
        b = info["boundary"]
        if b["bounded"]:
            print(f"         boundary {b['stamp']}: {b['cells_dropped']:,} "
                  f"cells outside it dropped, "
                  f"{b['walkable_dropped_m2']:,.0f} m2 of surveyed ground",
                  flush=True)
        print(f"         {info['vertices']} vertices, {info['edges']} edges, "
              f"{info['area_m2']:,.0f} m2; {info['route_points']:,} route points "
              f"(length error <= {info['route_length_error_m']} m)", flush=True)

    (a.out / "scenes" / "index.json").write_text(
        json.dumps({"config": {"variant": a.variant, "R": a.R, "seed": a.seed,
                               "bounded": not a.no_boundary},
                    "scenes": index}, indent=2) + "\n")
    print(f"[export] wrote {len(index)} scenes")


if __name__ == "__main__":
    main()
