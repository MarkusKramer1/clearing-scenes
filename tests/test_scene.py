"""The scenario files say what they contain; these check that they do."""

from __future__ import annotations

import numpy as np
import pytest

from clearing import scene as scene_mod
from clearing.scene import lattice_edges

NAMES = scene_mod.available()
SMALL = "hb-allen-centre"


@pytest.mark.parametrize("name", NAMES)
def test_yaml_agrees_with_geometry(name):
    """Every count in the YAML is recomputed from the arrays beside it."""
    s = scene_mod.load(name, speck_max_area_m2=0.0)
    d = s.doc

    assert d["surface"]["cells"] == s.n_cells
    assert d["surface"]["area_m2"] == pytest.approx(s.n_cells * s.cell_area)
    assert d["graph"]["vertices"] == s.n_vertices == len(d["graph"]["vertices_list"])
    assert d["graph"]["edges"] == s.edge_ij.shape[0] == len(d["graph"]["edges_list"])
    assert d["graph"]["edges_shady"] == int(s.edge_shady.sum())
    assert d["graph"]["edges_regular"] == int((~s.edge_shady).sum())
    assert d["surface"]["uncoverable_cells"] == s.uncoverable.size

    a, _ = s.evader_edges
    assert d["surface"]["evader_edges"] == a.size


@pytest.mark.parametrize("name", NAMES)
def test_graph_is_well_formed(name):
    s = scene_mod.load(name)
    assert s.edge_ij.min() >= 0
    assert s.edge_ij.max() < s.n_vertices
    assert (s.edge_ij[:, 0] != s.edge_ij[:, 1]).all(), "no self-loops"
    assert len(s.detection) == len(s.detection_boundary) == s.n_vertices
    for d in s.detection:
        assert d.size == 0 or (d.min() >= 0 and d.max() < s.n_cells)
    # a guard region lies inside the detection set of one of its endpoints
    for k in range(min(30, s.edge_ij.shape[0])):
        i, j = s.edge_ij[k]
        g = s.guard_region[k]
        seen = np.isin(g, s.detection[i]) | np.isin(g, s.detection[j])
        assert seen.all()


@pytest.mark.parametrize("name", NAMES)
def test_travel_times_are_distance_over_speed(name):
    s = scene_mod.load(name)
    v = s.doc["platform"]["speed_m_s"]
    T, D = s.travel_seconds, s.travel_metres
    ok = np.isfinite(T) & np.isfinite(D)
    assert np.allclose(T[ok], D[ok] / v, rtol=1e-4)
    assert np.allclose(np.diag(T), 0)
    assert np.allclose(T[ok], T.T[ok], rtol=1e-4), "travel is symmetric"

    # every reachable pair listed in the YAML matches the matrix
    for i, j, dist, secs in s.doc["routes"]["values"][:200]:
        assert D[i, j] == pytest.approx(dist, abs=0.06)
        assert T[i, j] == pytest.approx(secs, abs=0.06)


@pytest.mark.parametrize("name", NAMES)
def test_every_edge_carries_the_path_the_robot_walks(name):
    """The polyline runs from one vertex's cell to the other's, on the surface.

    Its measured length is shorter than the travel matrix by up to the
    simplification tolerance and never longer: Ramer-Douglas-Peucker only ever
    cuts corners off a lattice walk.
    """
    s = scene_mod.load(name)
    assert len(s.routes) == s.edge_ij.shape[0]
    assert s.doc["routes"]["polylines_for_graph_edges"] == \
        sum(1 for r in s.routes if len(r))

    ends = s.cell_xyz[s.vertex_cell]
    for k in range(min(60, s.edge_ij.shape[0])):
        i, j = (int(v) for v in s.edge_ij[k])
        r = s.routes[k]
        assert len(r) >= 2
        assert np.allclose(r[0], ends[i], atol=1e-3)
        assert np.allclose(r[-1], ends[j], atol=1e-3)

        walked = float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())
        true = float(s.travel_metres[i, j])
        assert walked <= true + 1e-3
        assert walked >= 0.9 * true

    # ... and it is at least as long as flying there in a straight line
    straight = np.linalg.norm(ends[s.edge_ij[:, 0]] - ends[s.edge_ij[:, 1]], axis=1)
    walked = np.array([s.travel_metres[i, j] for i, j in s.edge_ij])
    assert (walked >= straight - 1e-3).all()


def test_route_lookup_is_symmetric():
    s = scene_mod.load(SMALL)
    i, j = (int(v) for v in s.edge_ij[3])
    assert np.array_equal(s.route(i, j), s.route(j, i))
    assert s.route(0, 0).shape == (0, 3)


def test_lattice_respects_the_step_threshold():
    """Two cells in the same column, a metre apart, are not neighbours."""
    ij = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    z = np.array([0.0, 0.0, 0.0, 1.0])
    a, b, w = lattice_edges(ij, z, 0.2, 0.25)
    pairs = {tuple(sorted(p)) for p in zip(a.tolist(), b.tolist())}
    assert (0, 1) in pairs and (0, 2) in pairs
    assert (0, 3) not in pairs and (1, 3) not in pairs and (2, 3) not in pairs
    assert w.min() == pytest.approx(0.2)


def test_a_column_with_two_layers_keeps_both():
    """The ground under an arcade and the terrace over it both get neighbours."""
    ij = np.array([[0, 0], [1, 0], [1, 0]], dtype=np.int32)
    z = np.array([0.0, 0.0, 4.0])          # column 1 carries two walkable cells
    a, b, _ = lattice_edges(ij, z, 0.2, 0.25)
    pairs = {tuple(sorted(p)) for p in zip(a.tolist(), b.tolist())}
    assert (0, 1) in pairs, "ground joins ground"
    assert (0, 2) not in pairs, "ground does not step 4 m up to the terrace"


def test_specks_leave_the_evader_space():
    loose = scene_mod.load(SMALL, speck_max_area_m2=0.0)
    tight = scene_mod.load(SMALL, speck_max_area_m2=4.0)
    assert loose.excluded.sum() == 0
    assert tight.excluded.sum() > 0
    assert tight.evader_edges[0].size < loose.evader_edges[0].size
    # only uncoverable cells are ever dropped
    assert not tight.excluded[tight.coverable].any()


def test_the_speck_filter_may_not_cut_the_site_up():
    """A speck that is holding two pieces of surface together stays.

    THE FAILURE THIS PREVENTS is not a wrong area, it is a wrong verdict. A
    fragment is dropped because nothing can SEE it, which says nothing about
    whether it is load-bearing; some of them are thresholds -- a gate, a
    step-over -- and taking one out stops the propagator being able to walk an
    evader through the doorway. A strategy that leaves a leak on the far side
    is then recorded as clearing the site. On Blenheim Palace that was the
    difference between 11 055 m2 contaminated and `cleared`.

    THE INVARIANT, stated exactly: no piece of the whole walkable surface may
    be SPLIT by the removal. Whatever survives of one connected piece of the
    lattice must still be connected to itself in the evader space. A piece may
    vanish entirely -- that is a fragment that was nothing but speck -- and it
    may lose cells at its fringe, but it may not come apart.
    """
    import numpy as np
    from scipy.sparse.csgraph import connected_components

    from clearing.scene import _csr

    for name in scene_mod.available():
        s = scene_mod.load(name)
        la, lb = s.lattice
        ea, eb = s.evader_edges
        whole = connected_components(_csr(la, lb, s.n_cells), directed=False)[1]
        left = connected_components(_csr(ea, eb, s.n_cells), directed=False)[1]
        keep = ~s.excluded
        # one label of the reduced graph per label of the whole one
        order = np.argsort(whole[keep], kind="stable")
        w, q = whole[keep][order], left[keep][order]
        cut = np.flatnonzero(np.r_[True, w[1:] != w[:-1], True])
        for lo, hi in zip(cut[:-1], cut[1:]):
            assert np.unique(q[lo:hi]).size == 1, (
                f"{name}: surface piece {w[lo]} was split into "
                f"{np.unique(q[lo:hi]).size} by the speck filter")


def test_only_uncoverable_cells_are_ever_excluded():
    for name in scene_mod.available():
        s = scene_mod.load(name)
        assert not s.excluded[s.coverable].any(), name
