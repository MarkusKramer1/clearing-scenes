"""Driving an atomic strategy, and judging it on a clock, with no sliding.

The two modules under test are the ones that make the difference between the
published method's atomic verdict and its driven one, so what is asserted here
is mostly that they agree with things that are known independently: the atomic
verifier where nobody moves, and each other where the rule says they must.
"""

from __future__ import annotations

import numpy as np
import pytest

from clearing import clock, execute, gsst, scene as scene_mod, verify

from conftest import SMALL, corridor


# ---------------------------------------------------------------------------
# the executor
# ---------------------------------------------------------------------------


def test_the_fleet_is_the_searcher_count_unless_spent_unless_a_robot_is_spent():
    """Sequentialising moves nobody: the same machines go to the same posts.

    The only thing that may add one is `spend_machines`, and then it says so.
    """
    s = corridor(60, 10)
    strat = verify.Strategy([(0,), (0, 1), (1, 2), (2, 3), (3, 4, 5)])
    rep = execute.sequentialise(s, strat, absorb_m2=0.0)
    assert rep.fleet == strat.n_searchers + rep.metrics()["machines_spent"]


def test_every_move_moves_exactly_one_robot():
    s = corridor(60, 10)
    strat = verify.Strategy([(0, 1), (2, 3), (4, 5)])
    rep = execute.sequentialise(s, strat, absorb_m2=0.0)
    for prev, cur in zip(rep.strategy.steps, rep.strategy.steps[1:]):
        assert len(set(cur) - set(prev)) <= 1
        assert len(set(prev) - set(cur)) <= 1


def test_the_atomic_steps_are_all_reached():
    """Every set the atomic strategy asked for is occupied at some move.

    This is what makes the driven strategy the SAME plan rather than another
    one: the executor reorders the walking, it does not choose posts.
    """
    s = corridor(60, 10)
    strat = verify.Strategy([(0, 1), (1, 2), (2, 3, 4)])
    rep = execute.sequentialise(s, strat, absorb_m2=0.0)
    seen = {frozenset(st) for st in rep.strategy.steps}
    for want in strat.steps:
        assert any(set(want) <= q for q in seen)


def test_positions_agree_with_the_moves_and_the_occupied_set():
    s = scene_mod.load(SMALL)
    r = gsst.run(s, variant="regular", n_trees=8, seed=1)
    rep = execute.sequentialise(s, r.strategy)
    ros = rep.roster
    for moves, pos, occ in zip(ros.steps, ros.positions, rep.strategy.steps):
        assert sorted(v for v in pos if v >= 0) == sorted(occ)
        for m in moves:
            assert pos[m.machine] == m.to


def test_a_drive_nobody_can_cover_spends_a_robot_rather_than_conceding():
    """The paper's own fallback, and the alternative it is chosen over.

    A corridor cleared by a single machine leapfrogging forward has no standing
    team at all, so every drive opens the front. With `spend_machines` a second
    searcher is placed and nothing is given back; without it the least-leaking
    move is committed anyway and the concession is recorded.

    Absorption is off because the corridor is 2.4 m2 end to end and the 1 m2
    default would swallow the very fragments this is about.
    """
    s = corridor(40, 10)
    strat = verify.Strategy([(0,), (1,), (2,), (3,)])
    spent = execute.sequentialise(s, strat, spend_machines=True, absorb_m2=0.0)
    kept = execute.sequentialise(s, strat, spend_machines=False, absorb_m2=0.0)
    assert spent.metrics()["machines_spent"] == 1
    assert spent.fleet == 2 and spent.metrics()["legs_conceding"] == 0
    assert kept.metrics()["machines_spent"] == 0
    assert kept.fleet == 1 and kept.metrics()["legs_conceding"] == 3


# ---------------------------------------------------------------------------
# the clock
# ---------------------------------------------------------------------------


def test_a_strategy_nobody_drives_is_the_atomic_verdict():
    """With no transitions the clock has nothing to charge for.

    The clock is only ever stricter than `verify.propagate` because of what
    happens BETWEEN two steps. Hand it a strategy in which nothing moves and
    the two must give the same residual, or one of them is not modelling the
    instant the same way.

    Absorption off on BOTH sides. It is a real tolerance rather than a rounding
    one -- on this scene it is the difference between 34 cells left and none --
    so a comparison of two verifiers has to hold it fixed instead of leaving
    each to its own default.
    """
    s = scene_mod.load(SMALL)
    strat = verify.all_at_once(s)
    rep = execute.sequentialise(s, strat, absorb_m2=0.0)
    tl = clock.timeline_propagate(s, rep.strategy, rep.roster, exact=True,
                                  absorb_m2=0.0)
    atomic = verify.propagate(s, strat, absorb_m2=0.0)
    assert tl["residual_cells"] == int(atomic.residual.sum())
    assert tl["residual_cells"] > 0              # and it is not vacuous


def test_the_exact_rule_and_the_sampled_one_agree():
    """`exact=True` is not an approximation of `dt = 1 s, sub = 8`; it is it.

    Under the no-sliding rule `O(t)` changes only when a mover arrives, so its
    intersection over any interval is its value at the interval's left end.
    Flooding once per arrival must therefore reach the same fixed point as
    flooding nine times a second, and this is what says so on real geometry.
    """
    s = scene_mod.load(SMALL)
    r = gsst.run(s, variant="regular", n_trees=100, seed=1)
    rep = execute.sequentialise(s, r.strategy)
    a = clock.timeline_propagate(s, rep.strategy, rep.roster, exact=True)
    b = clock.timeline_propagate(s, rep.strategy, rep.roster, dt=1.0, sub=8)
    for k in ("cleared", "residual_cells", "residual_m2",
              "residual_avoidable_m2", "cleared_within_tolerance",
              "mission_seconds"):
        assert a[k] == b[k], k
    assert a["substeps"] < b["substeps"]        # and it is very much cheaper


def test_refining_dt_never_reports_less_contamination():
    """The strict rule is a refinement ladder, and this is which way it runs.

    A finer `dt` looks at more instants, so `M_I` can only shrink and the
    residual can only grow. A number that got BETTER when the clock was
    refined would mean the rule is not the one documented.
    """
    s = scene_mod.load(SMALL)
    r = gsst.run(s, variant="regular", n_trees=100, seed=1)
    rep = execute.sequentialise(s, r.strategy)
    coarse = clock.timeline_propagate(s, rep.strategy, rep.roster, dt=8.0)
    fine = clock.timeline_propagate(s, rep.strategy, rep.roster, dt=1.0)
    assert fine["residual_cells"] >= coarse["residual_cells"]


def test_the_instant_rule_is_the_optimistic_one():
    s = scene_mod.load(SMALL)
    r = gsst.run(s, variant="regular", n_trees=100, seed=1)
    rep = execute.sequentialise(s, r.strategy)
    strict = clock.timeline_propagate(s, rep.strategy, rep.roster, rule="strict")
    instant = clock.timeline_propagate(s, rep.strategy, rep.roster, rule="instant")
    assert instant["residual_cells"] <= strict["residual_cells"]


def test_a_roster_of_the_wrong_length_is_refused():
    s = corridor(60, 10)
    strat = verify.Strategy([(0, 1), (1, 2)])
    rep = execute.sequentialise(s, strat)
    with pytest.raises(ValueError, match="steps"):
        clock.timeline_propagate(s, verify.Strategy([(0,)]), rep.roster)


def test_the_driver_is_credited_with_nothing():
    """The 2010 prohibition, asserted rather than assumed.

    Two blocks, one machine, and the strategy that clears them in the atomic
    model: stand on the first, then stand on the second. Nobody is ever in
    transit there, so it clears. Driven, the machine is blind between the two
    posts, the whole corridor floods back while it walks, and the block it
    started in is contaminated when it arrives.

    This is the entire difference between the two verdicts, on the smallest
    instance that shows it -- and no `dt` closes it, because the credit for a
    driving sensor is a modelling choice and not a sampling rate.
    """
    s = corridor(20, 10)
    strat = verify.Strategy([(0,), (1,)])
    assert verify.propagate(s, strat, absorb_m2=0.0).cleared   # atomically
    rep = execute.sequentialise(s, strat, spend_machines=False, absorb_m2=0.0)
    tl = clock.timeline_propagate(s, rep.strategy, rep.roster, exact=True,
                                  absorb_m2=0.0)
    assert not tl["cleared"]                         # and the driven one
    assert tl["residual_cells"] == 10                # the block it came from
    assert tl["exposure"][0]["peak_contaminated"] == s.n_cells


# ---------------------------------------------------------------------------
# end to end, on the smallest real scene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", gsst.VARIANTS)
def test_the_published_method_clears_the_smallest_scene_when_driven(variant):
    """What the repaired vertex set bought, on the scene cheap enough to test.

    The same strategy, atomically, leaves over a thousand square metres. That
    gap is the result, so both halves of it are asserted.
    """
    s = scene_mod.load(SMALL)
    r = gsst.run(s, variant=variant, n_trees=100, seed=1)
    rep = execute.sequentialise(s, r.strategy)
    tl = clock.timeline_propagate(s, rep.strategy, rep.roster, exact=True)
    assert tl["residual_avoidable_m2"] == 0.0
    assert tl["cleared_within_tolerance"]
    assert rep.fleet == r.best.searchers              # no machine had to be spent


#: The published method's cheapest CLEARING fleet per scene, driven one machine
#: at a time with no sliding and judged on the strict clock. Pinned because these
#: are the numbers the README and `docs/gsst.md` quote, and because every one
#: of them moved when the vertex set was repaired and when the speck filter was
#: stopped from cutting the site up. The three larger scenes are measured the
#: same way and recorded in `examples/*.gsst.yaml`; they are left out here only
#: because a Bodleian run is ten thousand seconds of simulated mission and
#: minutes of wall clock.
DRIVEN = {
    "hb-allen-centre": ("regular_biased", 13),
    "keble-college": ("regular", 8),
    "blenheim-palace": ("regular", 10),
}


@pytest.mark.parametrize("name", sorted(DRIVEN))
def test_the_driven_team_has_not_moved(name):
    variant, fleet = DRIVEN[name]
    s = scene_mod.load(name)
    r = gsst.run(s, variant=variant, n_trees=100, seed=1)
    rep = execute.sequentialise(s, r.strategy)
    tl = clock.timeline_propagate(s, rep.strategy, rep.roster, exact=True)
    assert rep.fleet == fleet
    assert tl["residual_avoidable_m2"] == 0.0
    assert tl["cleared_within_tolerance"]
