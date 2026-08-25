"""The scenario boundary: that it is there, and that the scene is inside it.

The cut is made upstream, in `scripts/export_from_e1.py`, against a mask this
repository never sees -- so the only thing that ships is the line and the
surface that survived it. These check the one thing that matters about that
pair: that they are the same cut. A surface exported against the wrong
boundary would still load, still plot and still clear; it would just be
somewhere else on the site.
"""

from __future__ import annotations

import numpy as np
import pytest

from clearing import scene as scene_mod

NAMES = scene_mod.available()


def inside(s, xy: np.ndarray) -> np.ndarray:
    """Point-in-boundary by crossing parity, exactly.

    `boundary_seg` is not a traced outline: it is the set of grid faces between
    a retained cell of the boundary raster and one that is not. So the faces
    perpendicular to x, counted along a ray in -x, give the retained mask back
    exactly rather than approximately -- which is what makes this a check on
    the export and not a second, softer opinion about where the line runs.
    """
    seg = np.asarray(s.boundary_seg, dtype=np.float64)
    d_a = s.boundary_cell_size
    vert = seg[np.abs(seg[:, 0, 0] - seg[:, 1, 0]) < 1e-9]     # constant x
    fx = vert[:, 0, 0]
    ylo = np.minimum(vert[:, 0, 1], vert[:, 1, 1])

    out = np.zeros(len(xy), dtype=bool)
    for k, (px, py) in enumerate(xy):
        band = (ylo <= py) & (py < ylo + d_a)
        out[k] = bool(np.count_nonzero(fx[band] < px) % 2)
    return out


@pytest.mark.parametrize("name", NAMES)
def test_the_scene_carries_the_boundary_it_was_cut_to(name):
    s = scene_mod.load(name)
    b = s.doc["boundary"]
    assert b["bounded"], "these scenes are exported inside a boundary"
    assert b["stamp"], "an approved boundary is identified by its stamp"
    assert b["rim_segments"] == len(s.boundary_seg)
    assert b["rim_length_m"] == pytest.approx(
        len(s.boundary_seg) * s.boundary_cell_size, abs=0.1)
    assert b["walkable_dropped_m2"] == pytest.approx(
        b["cells_dropped"] * s.cell_area, abs=0.1)
    assert b["cells_dropped"] > 0, "every one of these scenes gives up ground"


@pytest.mark.parametrize("name", NAMES)
def test_every_cell_of_the_surface_is_inside_the_line(name):
    """The surface and the line are the same cut, checked against the line.

    Sampled rather than exhaustive: parity is O(segments) per point and these
    surfaces run to 300 000 cells. A surface cut to a different boundary does
    not disagree in a corner, it disagrees over hundreds of square metres, so a
    thousand cells is far more than enough to catch it.

    THE SEAM IS ALLOWED, AND ONLY THE SEAM. The two grids differ -- the
    surface is 0.2 m, the boundary raster 0.4 m -- and E1 decides a cell by
    the arithmetic `floor(i * d_v / d_a)`, which at an exact face is a
    floating-point coin flip: `728 * 0.2 / 0.4` is 363.999999999999994 in
    double, so that cell lands one row short of where it geometrically sits.
    Measured against the mask itself, that moves 383 cells across all six
    scenes -- 4 at HB Allen, 168 at the Bodleian, between 0.16 and 6.72 m² per
    scene, against 1 280 to 11 469 m² retained.

    Those cells are kept anyway, and this test allows them, because the thing
    worth protecting is that this repository's cut IS E1's cut. Recomputing
    the mapping here to put the seam right would make the two disagree by the
    same 383 cells in the other direction, and a scenario that is nearly the
    approved one is worth less than one that is exactly it. So the tolerance
    is a distance, not a count: an outlier has to be ON the line.
    """
    s = scene_mod.load(name)
    rng = np.random.default_rng(0)
    pick = rng.choice(s.n_cells, size=min(2000, s.n_cells), replace=False)
    xy = s.cell_xyz[pick, :2]
    got = inside(s, xy)
    if got.all():
        return

    from scipy.spatial import cKDTree

    mid = np.asarray(s.boundary_seg, np.float64).mean(axis=1)
    d, _ = cKDTree(mid).query(xy[~got])
    assert (d <= 1.5 * s.boundary_cell_size).all(), (
        f"{int((d > 1.5 * s.boundary_cell_size).sum())} sampled cells sit well "
        f"inside the excluded region, up to {d.max():.1f} m from the line -- "
        f"this surface was cut to a different boundary")
    assert (~got).sum() <= 0.01 * len(pick), (
        f"{int((~got).sum())} of {len(pick)} sampled cells are outside the "
        f"line; the seam accounts for a handful, not for this")


@pytest.mark.parametrize("name", NAMES)
def test_every_vertex_stands_inside_the_line(name):
    """And so does every place a robot is asked to stand."""
    s = scene_mod.load(name)
    got = inside(s, s.node_xyz[:, :2])
    assert got.all(), (f"vertices {np.flatnonzero(~got).tolist()} stand "
                       f"outside the scenario boundary")


@pytest.mark.parametrize("name", NAMES)
def test_the_line_closes(name):
    """Each end of each segment meets an odd number of others: no loose ends.

    A boundary with a gap in it is not a boundary, and a gap is exactly what a
    dropped face looks like -- invisible in a picture, fatal to the parity test
    above and to any claim that the evader cannot leave.
    """
    s = scene_mod.load(name)
    pts = np.round(np.asarray(s.boundary_seg, np.float64).reshape(-1, 2), 4)
    _, counts = np.unique(pts, axis=0, return_counts=True)
    assert (counts % 2 == 0).all(), "the boundary has loose ends"
