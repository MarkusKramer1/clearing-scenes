"""Identities on a strategy: the fleet, its legs, and the ground under them."""

from __future__ import annotations

import numpy as np
import pytest

from clearing import roster, scene as scene_mod, verify

from conftest import SMALL, corridor


# ---------------------------------------------------------------------------
# the fleet
# ---------------------------------------------------------------------------


def test_every_step_is_covered_by_one_machine_per_vertex():
    s = corridor(60, 10)
    strat = verify.Strategy([(0, 1), (1, 2), (2, 3, 4)])
    R = roster.assign(s, strat)
    for step, moves in zip(strat.steps, R.steps):
        assert sorted(m.to for m in moves) == sorted(step)
        assert len({m.machine for m in moves}) == len(step)


def test_the_fleet_is_the_searcher_count():
    """The claim the machine view makes: the ids stop at the largest step.

    A strategy that shrinks and grows again must not invent machines. The
    machines a step does not need park where they stand and are re-tasked.
    """
    s = corridor(60, 10)
    strat = verify.Strategy([(0, 1, 2), (0,), (0, 5)])
    R = roster.assign(s, strat)
    assert R.n_machines == max(len(st) for st in strat.steps) == 3
    assert [m.entered for st in R.steps[1:] for m in st] == [False] * 3


def test_a_re_tasked_machine_walks_from_where_it_parked():
    """And the makespan is what THAT walk costs, not a fresh machine's.

    Corridor of six vertices; the team is three, drops to one, and asks for a
    second one back at the far end. Charging a new machine from the entry would
    make the last step cost `T[0, 5]`; the machine parked at vertex 2 is nearer,
    so the assignment sends it and the step costs `T[2, 5]`.
    """
    s = corridor(60, 10)
    strat = verify.Strategy([(0, 1, 2), (0,), (0, 5)])
    R = roster.assign(s, strat)
    last = [m for m in R.steps[-1] if m.to == 5][0]
    assert last.frm == 2
    assert R.step_seconds[-1] == pytest.approx(s.travel_seconds[2, 5])
    assert R.step_seconds[-1] < s.travel_seconds[0, 5]
    # the first step is the team walking on to the site, and it still costs
    assert R.makespan_s == pytest.approx(s.travel_seconds[0, 2]
                                         + s.travel_seconds[2, 5])


def test_a_machine_always_leaves_from_where_it_stopped():
    s = scene_mod.load(SMALL)
    strat = _schedule_of(s)
    R = roster.assign(s, strat)
    at = {}
    for t, moves in enumerate(R.steps):
        for m in moves:
            assert m.frm == at.get(m.machine, R.entry)
            assert m.entered == (m.machine not in at)
            at[m.machine] = m.to
        assert R.positions[t][: len(at)] == [at[r] for r in range(len(at))]
        assert len(R.positions[t]) == R.n_machines


def test_the_makespan_is_the_slowest_machine_of_every_step():
    s = scene_mod.load(SMALL)
    strat = _schedule_of(s)
    R = roster.assign(s, strat)
    for moves, secs in zip(R.steps, R.step_seconds):
        assert secs == pytest.approx(max(m.seconds for m in moves))
    assert verify.makespan(s, strat) == pytest.approx(R.makespan_s)
    assert verify.fleet(s, strat) == R.n_machines


def test_an_empty_strategy_has_no_fleet():
    R = roster.assign(corridor(20, 10), verify.Strategy([]))
    assert R.n_machines == 0 and R.steps == [] and R.makespan_s == 0.0


# ---------------------------------------------------------------------------
# the ground under a leg
# ---------------------------------------------------------------------------


def test_legs_are_walked_over_the_surface():
    """Endpoints on the two vertices, length equal to the quoted travel.

    The viewer animates machines along these polylines, so a leg that cut through
    a building would be a picture of something the travel time never allowed.
    The vertices sit at sensor height above the ground the path runs on, hence
    the plan comparison.
    """
    s = scene_mod.load(SMALL)
    R = roster.assign(s, _schedule_of(s))
    paths, worst = roster.walked_paths(s, R.pairs())

    assert set(paths) == set(R.pairs())
    assert worst <= 0.05
    for (i, j), p in paths.items():
        assert p.shape[0] >= 2
        assert np.allclose(p[0, :2], s.vertex_xyz[i, :2], atol=s.cell_size)
        assert np.allclose(p[-1, :2], s.vertex_xyz[j, :2], atol=s.cell_size)
        walked = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        assert walked >= np.linalg.norm(s.vertex_xyz[i, :2] - s.vertex_xyz[j, :2]) - 1e-6
        assert walked <= s.travel_metres[i, j] + 1.0     # simplified for drawing


def test_a_leg_along_a_graph_edge_is_the_shipped_route():
    """One walk, one polyline. The viewer must not draw two different lines."""
    s = scene_mod.load(SMALL)
    i, j = (int(v) for v in s.edge_ij[0])
    paths, _ = roster.walked_paths(s, [(i, j), (j, i)])
    shipped = s.route(i, j)
    assert np.allclose(sorted(map(tuple, paths[(i, j)])), sorted(map(tuple, shipped)))
    assert np.allclose(paths[(i, j)], paths[(j, i)][::-1])


def _schedule_of(s) -> verify.Strategy:
    """A real strategy to put a fleet on: the published method's own.

    Was `clearing.kolling.plan` until that baseline was removed. The roster is
    indifferent to which planner produced a strategy -- it is handed a sequence
    of sets -- so the fixture only has to be a strategy that moves a team
    around this scene, and the one under test elsewhere is the cheapest to get.
    """
    from clearing import gsst

    return gsst.run(s, variant="regular", n_trees=8, seed=1).strategy
