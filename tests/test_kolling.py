"""The baseline and the verifier, on a toy graph and on the smallest real scene."""

from __future__ import annotations

import numpy as np
import pytest

from clearing import kolling, scene as scene_mod, verify

SMALL = "hb-allen-centre"


# ---------------------------------------------------------------------------
# a corridor of n cells, with one vertex per block of `span` cells
# ---------------------------------------------------------------------------


def corridor(n_cells: int = 60, span: int = 10) -> scene_mod.Scene:
    """A 1-cell-wide corridor. Every vertex sees its own block and nothing else.

    Small enough to reason about by hand: k blocks in a row means a schedule
    has to sweep from one end, and one robot behind the front is enough.
    """
    ij = np.stack([np.arange(n_cells), np.zeros(n_cells)], axis=1).astype(np.int32)
    xyz = np.column_stack([np.arange(n_cells) * 0.2, np.zeros(n_cells),
                           np.zeros(n_cells)]).astype(np.float32)
    n_nodes = n_cells // span
    detection = [np.arange(k * span, (k + 1) * span, dtype=np.int32)
                 for k in range(n_nodes)]
    node_cell = np.array([k * span + span // 2 for k in range(n_nodes)], np.int32)
    edges = np.array([[k, k + 1] for k in range(n_nodes - 1)], np.int32)
    T = np.abs(node_cell[:, None] - node_cell[None, :]).astype(np.float32) * 0.2
    return scene_mod.Scene(
        name="corridor", title="corridor",
        doc={"surface": {"max_step_m": 0.25}, "robot": {"speed_m_s": 1.0}},
        cell_xyz=xyz, cell_ij=ij, cell_size=0.2,
        detection=detection, rim=[d[[0, -1]] for d in detection],
        node_xyz=xyz[node_cell], node_cell=node_cell,
        edge_ij=edges, edge_shady=np.zeros(len(edges), bool),
        guard=[np.array([span * (k + 1) - 1], np.int32) for k in range(len(edges))],
        travel_seconds=T, travel_metres=T,
        uncoverable=np.zeros(0, np.int32), speck_max_area_m2=0.0)


# ---------------------------------------------------------------------------
# the label recursion
# ---------------------------------------------------------------------------


def test_label_of_a_path_is_two_robots():
    """A corridor: one robot sweeps, one holds the edge in front of it.

    An interior vertex entered from its parent has a single child, so
    L = max(rho + w, L_child) = 2 all the way down, and the label does not grow
    with the length of the corridor. Rooting at an end is what the recursion
    picks, and the order it hands back is the sweep from that end.
    """
    s = corridor(60, 10)                       # 6 vertices in a row, 5 edges
    tree, rest = kolling.spanning_tree(s)
    assert len(tree) == 5 and rest == []
    lab = kolling.label_tree(s, tree, rest)
    assert lab.robots == 2
    assert lab.root in (0, s.n_nodes - 1)
    assert lab.order in (list(range(s.n_nodes)), list(reversed(range(s.n_nodes))))


def test_label_of_a_star_is_the_arms_held_at_once():
    """Sweeping the centre means holding every still-dirty arm at once.

    Five arms round one centre. Entering at the centre costs six -- a sweeper
    and a blocker per arm. Entering along an arm costs five, because that arm
    is already clear by the time the centre is swept, and the recursion picks
    the cheaper root.
    """
    s = corridor(60, 10)                       # six vertices
    arms = s.n_nodes - 1
    star = [(0, k) for k in range(1, s.n_nodes)]
    lab = kolling.label_tree(s, star, [])
    assert lab.per_root[0] == arms + 1
    assert lab.robots == arms
    assert lab.root != 0


def test_non_tree_edges_make_the_label_larger():
    s = corridor(60, 10)
    tree, _ = kolling.spanning_tree(s)
    plain = kolling.label_tree(s, tree, []).robots
    charged = kolling.label_tree(s, tree, [(0, 3), (1, 4)]).robots
    assert charged > plain


# ---------------------------------------------------------------------------
# planning and verifying
# ---------------------------------------------------------------------------


def test_the_corridor_is_cleared_by_a_small_team():
    s = corridor(60, 10)
    p = kolling.plan(s, order="kolling")
    r = verify.propagate(s, p.schedule)
    assert r.cleared
    assert max(r.team) <= 3
    assert r.recontamination_events == 0


def test_an_empty_schedule_clears_nothing():
    s = corridor(20, 10)
    r = verify.propagate(s, verify.Schedule([]))
    assert not r.cleared
    assert r.residual.all()


def test_clearing_out_of_order_recontaminates():
    """Skip a block and the part behind it reconnects to what is still dirty.

    Sweeping the corridor in order clears it with one robot at a time, because
    each detection set separates what is behind it from what is ahead. Jumping
    the third block instead leaves blocks one and two joined, so the flood puts
    the cleared end straight back.
    """
    s = corridor(40, 10)                       # 4 vertices, one block each
    good = verify.propagate(s, verify.Schedule([(0,), (1,), (2,), (3,)]))
    assert good.cleared
    assert good.recontamination_events == 0

    bad = verify.propagate(s, verify.Schedule([(0,), (2,), (1,), (3,)]))
    assert bad.recontamination_events > 0


def test_a_schedule_with_an_unknown_vertex_is_rejected():
    s = corridor(20, 10)
    with pytest.raises(ValueError):
        verify.propagate(s, verify.Schedule([(99,)]))


def test_makespan_counts_the_walking():
    s = corridor(60, 10)
    far = verify.Schedule([(0,), (5,), (0,)])
    assert verify.makespan(s, far) == pytest.approx(2 * s.travel_seconds[0, 5])
    assert verify.makespan(s, verify.Schedule([(0,), (0,), (0,)])) == 0.0


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------


def test_all_at_once_is_the_ceiling_on_a_real_scene():
    """Whatever survives every vertex at once survives every schedule."""
    s = scene_mod.load(SMALL)
    r = verify.propagate(s, verify.all_at_once(s))
    assert r.cleared_except_uncoverable(s)
    assert not r.cleared, "these scenes all keep some uncoverable area"


def test_the_baseline_clears_the_smallest_scene():
    s = scene_mod.load(SMALL)
    p = kolling.plan(s, order="kolling")
    r = verify.propagate(s, p.schedule)
    assert r.cleared_except_uncoverable(s)
    assert max(r.team) <= s.n_nodes
    assert r.makespan_s > 0
