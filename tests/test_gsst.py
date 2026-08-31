"""The published Kolling et al. (2010) strategy layer: label, moves, conversion.

The label recursion and the move sequence are checked against each other on
trees small enough to count by hand, because that pairing is the whole claim:
`label_no_slide` says what a tree costs and `tree_strategy` has to spend
exactly that and no more.
"""

from __future__ import annotations

import numpy as np
import pytest

from clearing import gsst, scene as scene_mod, verify
from conftest import corridor

SMALL = "hb-allen-centre"


def children_of(edges: list[tuple[int, int]], n: int, root: int):
    """Root an explicit edge list, for the hand-checkable trees below."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    children: list[list[int]] = [[] for _ in range(n)]
    seen = [False] * n
    seen[root] = True
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                children[u].append(v)
                stack.append(v)
    return children, adj


# ---------------------------------------------------------------------------
# the label, and the searcher the missing slide costs
# ---------------------------------------------------------------------------


def test_a_single_vertex_costs_one():
    children, _ = children_of([], 1, 0)
    assert gsst.label_no_slide(children, 0)[0] == 1


@pytest.mark.parametrize("k", [2, 3, 5, 12, 40])
def test_a_path_costs_two_without_sliding(k):
    """One machine walks a corridor only if it may sweep while it walks.

    Classical edge search gives max{rho_1, rho_2 + 1} = 1 for a path: the
    searcher slides from one end to the other and the corridor is clear behind
    it. Forbidding the slide means the guard has to stand still until somebody
    else is placed ahead of it, which is the second machine -- and it is two for
    any length, not two per junction, because the pair leapfrogs.
    """
    edges = [(i, i + 1) for i in range(k - 1)]
    children, _ = children_of(edges, k, 0)
    assert gsst.label_no_slide(children, 0)[0] == 2


def test_a_star_costs_two():
    """rho_1 = 1, so the rule adds one: leaves are cheap but not free."""
    edges = [(0, 1), (0, 2), (0, 3)]
    children, _ = children_of(edges, 4, 0)
    assert gsst.label_no_slide(children, 0)[0] == 2


def test_a_junction_of_three_corridors_costs_three():
    """Three subtrees of label 2 meeting: rho_1 = rho_2 = 2, so max{2, 3} = 3.

    Two of the three arms have to be held while the third is worked on, and
    the arm being worked on already costs two by itself.
    """
    edges = []
    n = 1
    for _arm in range(3):
        edges += [(0, n), (n, n + 1), (n + 1, n + 2)]
        n += 3
    children, _ = children_of(edges, n, 0)
    assert gsst.label_no_slide(children, 0)[0] == 3


# ---------------------------------------------------------------------------
# the moves spend exactly the label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edges,n", [
    ([(i, i + 1) for i in range(9)], 10),                       # a path
    ([(0, 1), (0, 2), (0, 3)], 4),                              # a star
    ([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (0, 6), (6, 7)], 8),
])
def test_the_strategy_costs_the_label_and_clears_the_tree(edges, n):
    for root in range(n):
        children, adj = children_of(edges, n, root)
        lab = gsst.label_no_slide(children, root)
        steps = gsst.tree_strategy(children, lab, root)
        peak = max(len(s) for s in steps)
        assert peak == lab[root], f"root {root}: spent {peak}, label {lab[root]}"
        ok, residual = gsst.graph_clears(
            n, [tuple(sorted(s)) for s in steps], adj)
        assert ok, f"root {root}: {residual} vertices left contaminated"


def test_every_vertex_is_swept_at_least_once():
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]
    children, _ = children_of(edges, 6, 0)
    lab = gsst.label_no_slide(children, 0)
    swept = set().union(*gsst.tree_strategy(children, lab, 0))
    assert swept == set(range(6))


# ---------------------------------------------------------------------------
# cycle edges
# ---------------------------------------------------------------------------


def test_a_cycle_edge_buys_a_guard():
    """A path plus the edge that closes it into a ring.

    On the path alone the pair leapfrogs to the end and nothing is left
    standing. Close the ring and the far end is a door back into the cleared
    start, so the conversion stations a machine on it -- the strategy gets
    strictly more expensive, and that is the term GSST is minimising over
    trees.
    """
    n = 6
    ring = [(i, (i + 1) % n) for i in range(n)]
    path = ring[:-1]
    children, path_adj = children_of(path, n, 0)
    lab = gsst.label_no_slide(children, 0)
    steps = gsst.tree_strategy(children, lab, 0)

    ring_adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in ring:
        ring_adj[i].append(j)
        ring_adj[j].append(i)
    cycle_adj: list[list[int]] = [[] for _ in range(n)]
    cycle_adj[0].append(n - 1)
    cycle_adj[n - 1].append(0)

    on_path = [tuple(sorted(s)) for s in steps]
    on_ring, dirty, left = gsst.to_graph_strategy(n, steps, path_adj, cycle_adj)

    # the tree state the conversion read the doors off, and where it ended:
    # a contiguous strategy on the tree leaves nothing behind, so the last
    # frame is empty and there is one frame per step
    assert len(dirty) == len(steps)
    assert not left.any()

    assert gsst.graph_clears(n, on_path, path_adj)[0]
    assert not gsst.graph_clears(n, on_path, ring_adj)[0]     # the door is open
    assert gsst.graph_clears(n, on_ring, ring_adj)[0]         # the guard shuts it
    assert max(len(s) for s in on_ring) > max(len(s) for s in on_path)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_gsst_clears_the_corridor_scene():
    s = corridor(60, 10)
    r = gsst.run(s, variant="naive", n_trees=5, seed=1)
    assert r.best.cleared
    assert r.best.searchers == 2                       # a corridor, without sliding


@pytest.mark.parametrize("variant", gsst.VARIANTS)
def test_gsst_clears_the_graph_on_a_real_scene(variant):
    s = scene_mod.load(SMALL)
    r = gsst.run(s, variant=variant, n_trees=8, seed=1)
    assert r.best.cleared, f"{variant}: {r.best.residual_vertices} vertices left"
    assert r.best.searchers <= s.n_vertices
    assert len(r.searchers_per_draw) == 8


def test_the_best_draw_is_reproducible():
    s = scene_mod.load(SMALL)
    a = gsst.run(s, variant="regular", n_trees=8, seed=7)
    b = gsst.run(s, variant="regular", n_trees=8, seed=7)
    assert a.best.steps == b.best.steps
