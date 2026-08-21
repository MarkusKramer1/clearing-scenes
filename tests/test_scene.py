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
    assert d["graph"]["vertices"] == s.n_nodes == len(d["graph"]["vertices_list"])
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
    assert s.edge_ij.max() < s.n_nodes
    assert (s.edge_ij[:, 0] != s.edge_ij[:, 1]).all(), "no self-loops"
    assert len(s.detection) == len(s.rim) == s.n_nodes
    for d in s.detection:
        assert d.size == 0 or (d.min() >= 0 and d.max() < s.n_cells)
    # a guard region lies inside the detection set of one of its endpoints
    for k in range(min(30, s.edge_ij.shape[0])):
        i, j = s.edge_ij[k]
        g = s.guard[k]
        seen = np.isin(g, s.detection[i]) | np.isin(g, s.detection[j])
        assert seen.all()


@pytest.mark.parametrize("name", NAMES)
def test_travel_times_are_distance_over_speed(name):
    s = scene_mod.load(name)
    v = s.doc["robot"]["speed_m_s"]
    T, D = s.travel_seconds, s.travel_metres
    ok = np.isfinite(T) & np.isfinite(D)
    assert np.allclose(T[ok], D[ok] / v, rtol=1e-4)
    assert np.allclose(np.diag(T), 0)
    assert np.allclose(T[ok], T.T[ok], rtol=1e-4), "travel is symmetric"

    # every reachable pair listed in the YAML matches the matrix
    for i, j, dist, secs in s.doc["routes"]["values"][:200]:
        assert D[i, j] == pytest.approx(dist, abs=0.06)
        assert T[i, j] == pytest.approx(secs, abs=0.06)


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
