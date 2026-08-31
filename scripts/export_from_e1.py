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
guard regions as the edges -- which is why `clearing/gsst.py` applies to it
unchanged. The ground half does not depend on the UAV altitude: the two
families are sampled independently, and `carved_huav10/15/20` carry
byte-identical ground vertex sets. `--h-uav` therefore only picks which of the
identical files to read from.

WHICH VERTEX SET, AND WHY IT IS NOT E1'S SAMPLE
------------------------------------------------
`--vertex-set` selects it, and the default is `repaired-ground`.

E1's sampler places sensor positions until free space is COVERED. That is the
art-gallery objective the 2010 formulation inherits, and it is not the property
a clearing argument needs. Two things survive into the raw sample, both
measured upstream and neither visible in a coverage figure:

  * fragments that SEVER the walkable surface are watched by nobody, so the
    ground guard graph comes apart where the site does not. Christ Church's
    surface is one connected piece of 8 133 m2 and its ground guard graph was
    in three, the whole split carried by a 0.40 m2 gate and a 0.32 m2 step-over
    (`e1-clearing-graph/docs/guard_graph_components.md`). Clearing components
    "in sequence" was then a sound argument about the wrong object.
  * 16 to 116 m2 per scene is ground that no GROUND vertex can see -- and every
    square metre of it is seen by the air family. For a homogeneous ground team
    that is a permanent contamination source, a floor no schedule can go under,
    and it is a property of a sample drawn for a MIXED team.

`repaired-ground` is E1's sample plus one ground sensor per severing threshold
(`scripts/run_graph_connect.py`) and one per ground-blind cluster
(`scripts/run_ground_closure.py`). Ground vertices, released -> repaired-ground:
hb-allen 20 -> 32, keble 24 -> 34, blenheim 36 -> 47, observatory 39 -> 60,
christ-church 62 -> 81, bodleian 112 -> 151. It is a CHANGE TO THE PROBLEM and
it is recorded as one, in `scenes/index.json` and in every scene YAML.

`released` is the raw sample, kept as the control. `repaired` (thresholds) and
`repaired-blind` are the intermediate levels; `repaired-blind` is the set
E1's own mixed-team planners are measured on, and it still leaves the
ground-blind holes, so a ground-only schedule cannot reach zero on it.

The detection predicate is untouched: all five rays to the target cylinder must
arrive. E1's `any-ray` is a weaker requirement and therefore a DIFFERENT
guarantee, not a cheaper computation of this one.

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


#: `--vertex-set` -> the repair level `e2.gsst.repair_scene` knows it by.
LEVELS = {"repaired-ground": "ground", "repaired-blind": "blind",
          "repaired": "thresholds"}


def released_ground(gpath: Path, tpath: Path) -> dict:
    """The ground half of E1's Stage-6 graph, exactly as it was written.

    This is the vertex set the first six releases of this repository shipped:
    E1's Stage-5 sample with nothing added to it. It is kept because a
    scenario nobody can reproduce is worth less than one whose history is
    readable, and because `--vertex-set released` is the control every claim
    about the repaired sets is a claim against.

    WHAT IS WRONG WITH IT, said once here rather than discovered downstream.
    The sampler places positions until free space is COVERED, which is the
    art-gallery objective and not the property a clearing argument needs. Two
    consequences survive into the data: fragments that SEVER the walkable
    surface are watched by nobody, so the ground guard graph comes apart where
    the site does not; and 16 to 116 m2 per scene is ground no GROUND vertex
    can see, all of it seen by the air family, which is a floor no homogeneous
    schedule can go under. `repaired-ground` closes both. See
    `e1-clearing-graph/docs/guard_graph_components.md` and `docs/kolling-loop.md`.
    """
    z = np.load(gpath)
    fam = z["node_family"]
    ground = np.flatnonzero(fam == 0)            # 0 = ground, 1 = air
    remap = -np.ones(fam.size, dtype=np.int64)
    remap[ground] = np.arange(ground.size)

    dset_off, dset_cells = slices(z["dset_offsets"], z["dset_cells"], ground)
    bnd_off, bnd_cells = slices(z["bnd_offsets"], z["bnd_cells"], ground)

    eij = z["edge_ij"]
    both = (fam[eij[:, 0]] == 0) & (fam[eij[:, 1]] == 0)
    esel = np.flatnonzero(both)
    guard_off, guard_cells = slices(z["guard_offsets"], z["guard_cells"], esel)

    # The roadmap indexes its matrices by clearing-graph node id. Asserted
    # rather than assumed: a silent mismatch here would attach every travel
    # time to the wrong pair of vertices and nothing downstream would notice.
    t = np.load(tpath)
    if not np.array_equal(t["ugv_node_ids"], ground.astype(t["ugv_node_ids"].dtype)):
        raise ValueError(f"{tpath.name}: roadmap vertex order does not match "
                         "the ground vertices of the graph")

    return {
        "vertex_set": "released",
        "node_pos": z["node_pos"][ground].astype(np.float32),
        "node_row": z["node_row"][ground].astype(np.int32),
        "dset_off": dset_off, "dset_cells": dset_cells,
        "bnd_off": bnd_off, "bnd_cells": bnd_cells,
        "edge_ij": remap[eij[esel]].astype(np.int32),
        "edge_kind": z["edge_kind"][esel],           # 0 = regular, 1 = shady
        "guard_off": guard_off, "guard_cells": guard_cells,
        "seconds": t["ugv_seconds"].astype(np.float32),
        "metres": t["ugv_metres"].astype(np.float32),
        "component": t["ugv_component"].astype(np.int32),
        "note": "E1 Stage 5/6 as released; no vertices added",
    }


def repaired_ground(name: str, e1_root: Path, level: str, want: dict,
                    cells: np.ndarray, verbose: bool = True) -> dict:
    """E1's REPAIRED ground vertex set, its guard graph and its roadmap.

    WHY THIS CANNOT JUST READ AN ARTEFACT
    --------------------------------------
    `outputs/e2/graph_connect/<scene>_vertices_<level>.npz` carries the added
    VERTICES -- rows, positions, detection sets -- and, above `thresholds`, no
    edges at all. E2 does not need them in that form; it rebuilds the ground
    guard graph with E1's own builder every time it wants one
    (`e2.gsst.ground_guide`). So does this, for the same reason and with the
    same code: the edge relation shipped here has to be E1's, not a second
    implementation of it that happens to agree today.

    What it costs is the heavy machinery -- the occupancy grid, the raycaster,
    the survey data on the SSD -- which is why this script and no other in this
    repository needs them.

    THE REBUILD IS CHECKED, NOT TRUSTED. `e2.gsst.ground_guide` is called on
    the same arguments and the two graphs must agree edge for edge, kind for
    kind. That check is worth having twice over: `ground_guide` ALREADY refuses
    a rebuild that is not the committed `thresholds` artefact on the scenes
    where the two vertex sets coincide, so agreeing with it inherits that
    guarantee instead of restating it.

    The roadmap is rebuilt too. `<key>_traversal.npz` was written for the
    released node set and has no rows for the vertices this adds; reading it
    would attach travel times to the wrong vertices without failing.
    """
    from e1 import boundary_apply as ba  # noqa: PLC0415
    from e1.config import load_params  # noqa: PLC0415
    from e1.detection import (UAV as E1_UAV, UGV as E1_UGV,  # noqa: PLC0415
                              DetectionContext)
    from e1.graph import build_graph, full_detection_sets  # noqa: PLC0415
    from e1.grid import Grid  # noqa: PLC0415
    from e1.paths import DataPaths  # noqa: PLC0415
    from e1.sampling import SampleResult  # noqa: PLC0415
    from e1.traversal import build_traversal, ugv_lattice  # noqa: PLC0415
    from e2.gsst import ground_guide, repair_scene  # noqa: PLC0415
    from e2.propagate import UGV  # noqa: PLC0415
    from e2.scene import find_graph, load_pilot  # noqa: PLC0415

    released = load_pilot(e1_root, name, which=find_graph(e1_root, name, want),
                          verbose=False)
    # `repair_scene` re-reads the artefact's identity block -- variant, h_uav,
    # R, seed, cell count, vertex count, edge counts -- and refuses anything
    # but an exact match with the scene just loaded.
    raw = repair_scene(e1_root, name, released, level=level)
    index = [int(q) for q in np.flatnonzero(np.asarray(raw.families) == UGV)]
    rows = np.asarray(raw.node_rows)[index].astype(np.int64)

    p = load_params(e1_root / "configs" / "params.yaml")
    paths = DataPaths()
    grid = Grid.load(paths.grid_npz(name, "carved")).finalise(raw.variant)
    surf, _prov = ba.walkable_surface(name, raw.variant, paths, grid.shape,
                                      bounded=True)
    # The surface E2 loaded and the one this script cut must be the same one,
    # because every row index below is into it. Cheap to check, silent to get
    # wrong.
    if not np.array_equal(np.asarray(surf.cells)[:, :2].astype(np.int64),
                          np.asarray(cells)[:, :2].astype(np.int64)):
        raise ValueError(
            f"{name}: E2's walkable surface ({len(surf.cells):,} cells) is not "
            f"the one this export cut ({len(cells):,}); the two applied "
            f"different boundaries and every row index would read wrong cells")

    gpu = None
    try:
        from e1.raycast_gpu import make_raycaster  # noqa: PLC0415
        gpu = make_raycaster(grid.data, verbose=False)
    except Exception:                                            # noqa: BLE001
        pass
    ctx = DetectionContext(grid=grid, surf=surf, p=p, R=raw.R,
                           h_uav=raw.h_uav, gpu=gpu)
    samples = {E1_UGV: SampleResult(family=E1_UGV,
                                    node_rows=[int(r) for r in rows],
                                    detection_sets=[np.zeros(0, np.int64)]
                                    * int(rows.size)),
               E1_UAV: SampleResult(family=E1_UAV)}
    samples = full_detection_sets(ctx, samples, verbose=False)
    pos = {(E1_UGV, int(r)): ctx.sensor_world(int(r), E1_UGV) for r in rows}
    G = build_graph(surf, samples, p.h_step, pos)

    # --- the check ---------------------------------------------------------
    guide, _gindex = ground_guide(e1_root, name, released, raw, level=level)
    kinds = {(min(e.i, e.j), max(e.i, e.j)):
             (0 if e.kind == "regular" else 1) for e in G.edges}
    want_kinds = {(min(i, j), max(i, j)): (0 if k == 0 else 1)
                  for (i, j), k in guide.kind_by_pair.items()}
    if kinds != want_kinds:
        only_here = sorted(set(kinds) - set(want_kinds))[:5]
        only_there = sorted(set(want_kinds) - set(kinds))[:5]
        raise ValueError(
            f"{name}: the guard graph rebuilt here ({len(kinds)} edges) is not "
            f"the one e2.gsst.ground_guide builds ({len(want_kinds)}). "
            f"Edges only here: {only_here}; only there: {only_there}")

    # --- pack --------------------------------------------------------------
    n = len(G.nodes)
    dset_off = np.zeros(n + 1, dtype=np.int64)
    bnd_off = np.zeros(n + 1, dtype=np.int64)
    dset_off[1:] = np.cumsum([nd.dset.size for nd in G.nodes])
    bnd_off[1:] = np.cumsum([nd.boundary.size for nd in G.nodes])
    guard_off = np.zeros(len(G.edges) + 1, dtype=np.int64)
    guard_off[1:] = np.cumsum([e.guard_cells.size for e in G.edges])

    def flat(arrays, total):
        return (np.concatenate(arrays).astype(np.int32) if total
                else np.zeros(0, dtype=np.int32))

    lat, ijz = ugv_lattice(surf, p)
    tg = build_traversal(
        E1_UGV, lat, ijz, np.asarray(surf.cells)[rows][:, :2].astype(np.int64),
        [nd.id for nd in G.nodes], grid.shape, surf.min_corner, surf.d_v,
        p.v_ugv, k_draw=0, verbose=verbose, src_rows=rows)

    return {
        "vertex_set": level,
        "node_pos": np.asarray([nd.position for nd in G.nodes],
                               dtype=np.float32).reshape(n, 3),
        "node_row": np.asarray([nd.row for nd in G.nodes], dtype=np.int32),
        "dset_off": dset_off,
        "dset_cells": flat([nd.dset for nd in G.nodes], int(dset_off[-1])),
        "bnd_off": bnd_off,
        "bnd_cells": flat([nd.boundary for nd in G.nodes], int(bnd_off[-1])),
        "edge_ij": np.asarray([(e.i, e.j) for e in G.edges],
                              dtype=np.int32).reshape(-1, 2),
        "edge_kind": np.asarray([0 if e.kind == "regular" else 1
                                 for e in G.edges], dtype=np.int8),
        "guard_off": guard_off,
        "guard_cells": flat([e.guard_cells for e in G.edges],
                            int(guard_off[-1])),
        "seconds": tg.seconds.astype(np.float32),
        "metres": tg.metres.astype(np.float32),
        "component": tg.component.astype(np.int32),
        "released_vertices": int((np.asarray(released.families) == UGV).sum()),
        "note": (f"E1 Stage 5/6 plus e2.gsst.repair_scene(level={level!r}); "
                 f"guard graph rebuilt with e1.graph.build_graph and checked "
                 f"against e2.gsst.ground_guide"),
    }


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
        "outline_segments": int(len(rim["seg"])) if rim else 0,
        "outline_length_m": float(len(rim["seg"]) * rim["d_a"]) if rim else 0.0,
        "_rim": rim,
    }


def evader_edges(cells, height, d_v, h_climb, lattice_edges):
    """8-connected, height-aware adjacency over the walkable cells.

    The same lattice E1 moves its ground platform on. Height-awareness matters
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


def edge_routes(cells, height, d_v, h_climb, xyz, vertex_cell, edge_ij,
                travel_metres, lattice_edges, eps=0.12, chunk=16):
    """The path the platform actually walks along every edge of the graph.

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
        _, pred = dijkstra(g, indices=vertex_cell[block], directed=False,
                           return_predecessors=True)
        for row, src in enumerate(block):
            p = pred[row]
            for k in by_src[src]:
                j = int(edge_ij[k, 1])
                path, q = [int(vertex_cell[j])], int(vertex_cell[j])
                while q != int(vertex_cell[src]):
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

    add(f"# {scene['title']} -- ground-platform clearing scenario")
    add("#")
    add("# Vertices are sampled sensor positions on the walkable surface. An edge")
    add("# (i, j) exists where a searcher at j can watch part of dD(p_i), the")
    add("# boundary of what a searcher at i sees: the guard region G_ij of")
    add("# Kolling et al. 2010. `routes` gives the platform's travel time")
    add("# between every pair of vertices;")
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
    add("platform:")
    for k, v in scene["platform"].items():
        add(f"  {k}: {v}")
    add("")
    add("target:")
    for k, v in scene["target"].items():
        add(f"  {k}: {v}")
    add("")

    b = scene["site_boundary"]
    add("site_boundary:")
    add("  # Where the scenario stops. Drawn by hand in e1-clearing-graph")
    add("  # (configs/site_boundary.yaml) and approved there; the surface,")
    add("  # the vertices, the guard regions and the routes below are all")
    add("  # what survives it. The evader cannot leave it and no searcher has")
    add("  # to watch across it: the drawn lines are held, not open frontier.")
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
        add(f"  outline_segments: {b['outline_segments']}")
        add(f"  outline_length_m: {fmt(b['outline_length_m'], 1)}    "
            "# the whole outline, in the npz as site_boundary_seg")
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
    add("  # id, sensor position [m], the area D(p) the vertex sees, and how")
    add("  # many cells its detection-set boundary dD(p) is")
    add("  vertices_list:")
    for n in g["nodes"]:
        add(f"    - {{id: {n['id']:>3}, "
            f"xyz: [{fmt(n['xyz'][0], 2):>9}, {fmt(n['xyz'][1], 2):>9}, "
            f"{fmt(n['xyz'][2], 2):>7}], "
            f"detection_m2: {fmt(n['detection_m2'], 1):>8}, "
            f"detection_boundary_cells: {n['boundary_cells']:>6}}}")
    add("")
    add("  # i, j, whether the guard region G_ij = dD(p_i) n D(p_j) is strictly")
    add("  # contained in another one (shady) or not (regular), and how big it is")
    add("  edges_list:")
    for e in g["edges"]:
        add(f"    - {{i: {e['i']:>3}, j: {e['j']:>3}, type: {e['type']:<7}, "
            f"guard_region_cells: {e['guard_region_cells']:>6}, "
            f"guard_region_m2: {fmt(e['guard_m2'], 2):>8}}}")
    add("")

    r = scene["routes"]
    add("routes:")
    add("  # Shortest path for the platform over the walkable surface,")
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
                 cloud_points: int, lattice_edges, ba, bounded: bool,
                 vertex_set: str) -> dict:
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

    if vertex_set == "released":
        g = released_ground(gpath, tpath)
    else:
        g = repaired_ground(name, e1_root, LEVELS[vertex_set],
                            {"variant": variant, "h_uav": h_uav, "R": R,
                             "seed": seed}, cells)
    node_pos, node_row = g["node_pos"], g["node_row"]
    dset_off, dset_cells = g["dset_off"], g["dset_cells"]
    bnd_off, bnd_cells = g["bnd_off"], g["bnd_cells"]
    edge_ij, edge_kind = g["edge_ij"], g["edge_kind"]
    guard_off, guard_cells = g["guard_off"], g["guard_cells"]
    sec, met, comp = g["seconds"], g["metres"], g["component"]
    n_vertices = int(node_pos.shape[0])

    # THE CHECK THAT A BOUNDARY MISMATCH CANNOT SURVIVE. `node_pos` is a world
    # position, written beside `node_row` and not derived from it, so the two
    # agree only if this surface is the surface the graph was built on. A graph
    # cut differently would still index in range and still load; it would just
    # put every vertex somewhere else on the site, and nothing downstream would
    # notice. Verified here, once, against the cheapest possible witness.
    if node_row.size:
        if int(node_row.max()) >= n_cells:
            raise ValueError(
                f"{name}: vertex row {int(node_row.max())} is past the end of "
                f"the {n_cells:,}-cell surface it is meant to index -- the "
                f"graph and the surface were cut differently")
        off = np.abs(node_pos[:, :2] - xyz[node_row][:, :2]).max()
        if off > 0.5 * d_v:
            raise ValueError(
                f"{name}: vertices sit up to {off:.2f} m from the cells they "
                f"index. The graph and the walkable surface were cut to "
                f"different boundaries; rerun E1 Stages 4-6, or re-approve.")

    # what no ground vertex sees, at any team size
    coverable = np.zeros(n_cells, dtype=bool)
    coverable[dset_cells] = True
    uncoverable = np.flatnonzero(~coverable).astype(np.int32)

    ea, eb = evader_edges(cells, height, d_v, h_climb, lattice_edges)
    route_off, route_pts, route_err = edge_routes(
        cells, height, d_v, h_climb, xyz, node_row, edge_ij, met, lattice_edges)

    cloud = np.load(e1_root / "outputs" / "scenes" / name
                    / "cloud_decimated.npy", mmap_mode="r")
    stride = max(1, cloud.shape[0] // cloud_points)
    cloud = np.ascontiguousarray(cloud[::stride]).astype(np.float32)

    np.savez_compressed(
        out / "scenes" / "geometry" / f"{name}.npz",
        # THE NAMES ARE THE PAPER'S where the paper has one. `D(p)` is a
        # detection set, its boundary is `dD(p)` and not a "rim", and
        # `G_ij = dD(p_i) n D(p_j)` is a guard region -- so a reader with
        # Kolling et al. (2010) open can match every array to a symbol in it.
        # `site_boundary_*` is the scenario cut and carries the longer name
        # precisely because `boundary` alone would now be ambiguous.
        cell_xyz=xyz, cell_ij=cells[:, :2].astype(np.int32),
        cell_size=np.float32(d_v),
        vertex_xyz=node_pos, vertex_cell=node_row,
        detection_offsets=dset_off, detection_cells=dset_cells,
        detection_boundary_offsets=bnd_off, detection_boundary_cells=bnd_cells,
        edge_ij=edge_ij, edge_shady=edge_kind.astype(np.uint8),
        guard_region_offsets=guard_off, guard_region_cells=guard_cells,
        travel_seconds=sec, travel_metres=met, travel_component=comp,
        edge_route_offsets=route_off, edge_route_points=route_pts,
        uncoverable=uncoverable,
        cloud_xyz=cloud,
        # The line the scenario stops at, as the faces between a retained cell
        # and one that is not -- E1's own rendering of it, at the boundary
        # raster's resolution rather than a smoothed polygon, so it meets the
        # surface it was cut from instead of floating a few centimetres off it.
        site_boundary_seg=(np.asarray(boundary["_rim"]["seg"], np.float32)
                           if boundary.get("_rim") else np.zeros((0, 2, 2), np.float32)),
        site_boundary_cell_size=np.float32(boundary["_rim"]["d_a"]
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
            "vertex_set": vertex_set,
            "vertex_set_note": g["note"],
            "note": ("the ground half of E1's two-platform graph; identical for "
                     "every h_uav, which only selects the file"),
        },
        "platform": {
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
        "site_boundary": boundary,
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
                for i in range(n_vertices)],
            "edges": [
                {"i": int(edge_ij[k, 0]), "j": int(edge_ij[k, 1]),
                 "type": "shady" if edge_kind[k] else "regular",
                 "guard_region_cells": int(guard_off[k + 1] - guard_off[k]),
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
        "scene": name, "title": doc["title"], "site_boundary": boundary,
        "vertex_set": vertex_set,
        "released_vertices": g.get("released_vertices"),
        "cells": n_cells, "area_m2": round(n_cells * cell_area, 1),
        "vertices": n_vertices, "edges": int(edge_ij.shape[0]),
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
    ap.add_argument("--vertex-set", default="repaired-ground",
                    choices=["repaired-ground", "repaired-blind", "repaired",
                             "released"],
                    help="which vertex set to ship. `released` is E1's raw "
                         "Stage-5 sample and is the control; the three "
                         "`repaired*` levels are E2's, and `repaired-ground` "
                         "is the only one with no ground a ground sensor "
                         "cannot see.")
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
                            ba, not a.no_boundary, a.vertex_set)
        index.append(info)
        b = info["site_boundary"]
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
                               "bounded": not a.no_boundary,
                               "vertex_set": a.vertex_set},
                    "scenes": index}, indent=2) + "\n")
    print(f"[export] wrote {len(index)} scenes")


if __name__ == "__main__":
    main()
