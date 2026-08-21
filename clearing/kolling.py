"""The 2.5D team-visibility baseline of Kolling et al. (2010), on these scenes.

WHAT THE PAPER DOES
-------------------
Kolling, Kleiner, Lewis & Sycara, *Pursuit-Evasion in 2.5D Based on
Team-Visibility*, IROS 2010, place guaranteed clearing on real terrain:

  * sample sensor positions on the surface until free space is covered;
  * for each position p compute its detection set D(p) by line of sight, given
    a sensor height and a minimum target height;
  * make each sampled position a vertex, and put an edge (i, j) wherever the
    rim of D(p_i) meets D(p_j) -- the guard region G_ij, the part of i's
    boundary a robot standing at j can watch;
  * an edge is *shady* when its guard region is strictly contained in another
    one and *regular* otherwise;
  * clear the graph with the GRAPH-CLEAR machinery of Kolling & Carpin: a
    spanning tree, a label computed bottom-up, and a strategy read off it.

Steps one to four happen in `e1-clearing-graph` and are what
`scripts/export_from_e1.py` ships in `scenes/`. This module is step five, and
it is the only part a user of this repository is expected to replace.

TWO THINGS ARE COMPUTED, AND THEY ANSWER DIFFERENT QUESTIONS
-------------------------------------------------------------
`label_tree` gives the GRAPH-CLEAR number: how many robots the tree recursion
says a spanning tree of this graph needs. It is a bound in a model where every
blocked edge costs its own robot.

`plan` gives an executable schedule. It does not charge per edge, because one
robot standing at p_j watches every guard region that lies inside D(p_j) at
once -- so the frontier is covered by a set cover over vertices, not by a sum
over edges. The schedule is what `clearing.verify` then checks at cell level,
and it needs materially fewer robots than the label. Both numbers are reported,
because the gap between them is a property of the geometry worth seeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import Scene
from .verify import Schedule, flood


# ---------------------------------------------------------------------------
# the spanning tree
# ---------------------------------------------------------------------------


def spanning_tree(scene: Scene) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """A maximum spanning forest by guard-region size, and what it leaves out.

    Maximum rather than minimum: an edge inside the tree is handled by the
    recursion, an edge outside it has to be blocked for as long as either of
    its endpoints is being worked on. Keeping the big guard regions in the tree
    is therefore the cheaper side to err on.

    A forest, not a tree: the guard graph of a real site need not be connected,
    and pretending otherwise would silently drop components.
    """
    size = np.array([g.size for g in scene.guard])
    order = np.argsort(-size)
    parent = list(range(scene.n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree, rest = [], []
    for k in order:
        i, j = (int(v) for v in scene.edge_ij[k])
        ri, rj = find(i), find(j)
        if ri == rj:
            rest.append((i, j))
        else:
            parent[ri] = rj
            tree.append((i, j))
    return tree, rest


# ---------------------------------------------------------------------------
# the GRAPH-CLEAR label on a tree
# ---------------------------------------------------------------------------


@dataclass
class TreeLabel:
    robots: int                     # the label at the best root
    root: int
    order: list[int]                # the vertex order the recursion clears in
    per_root: dict[int, int] = field(default_factory=dict)
    components: int = 1


def label_tree(scene: Scene, tree: list[tuple[int, int]],
               rest: list[tuple[int, int]] | None = None) -> TreeLabel:
    """The GRAPH-CLEAR label recursion of Kolling & Carpin (2007).

    With `rho(v)` robots to sweep a vertex and `w(e)` to block an edge, the
    label of the subtree at `v`, entered from its parent `p`, is

        L(v, p) = max{ rho(v) + sum_c w(v,c),
                       max_i [ L(c_i, v) + sum_{j>i} w(v,c_j) ] }

    -- either you are sweeping `v` itself, which needs a blocker on every edge
    to a child, or you are down in child `i`, which needs that child's own
    label plus a blocker on every child not yet cleared.

    There is no term for the parent edge. You came from `p`, so that side is
    already clear and nothing has to hold it: blocking is needed against
    contamination, not against the past. Charging it is the easy mistake here,
    and it makes the label grow along a corridor that one robot walks down.

    `v` is swept first and the children follow in increasing order of
    `L(c_i,v) - w(v,c_i)`, which the usual exchange argument shows is optimal:
    the expensive subtree is entered last, when the fewest siblings are still
    contaminated and holding blockers.

    Here `rho = 1` and `w = 1`: one robot standing at `p_i` sweeps `D(p_i)`,
    and one robot standing at `p_j` holds the guard region `G_ij`.

    Edges outside the spanning tree are charged to both their endpoints. That
    is the conservative reading -- such an edge has to stay blocked while
    either end is worked on -- and it is why a dense guard graph makes this
    number large.

    The root is chosen by trying every vertex, which is O(n^2) and irrelevant
    at these sizes.
    """
    n = scene.n_nodes
    rho = np.ones(n, dtype=np.int64)
    for i, j in (rest or []):
        rho[i] += 1
        rho[j] += 1

    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in tree:
        adj[i].append(j)
        adj[j].append(i)

    # which forest component each vertex is in; the answer is the max over them
    comp = -np.ones(n, dtype=np.int64)
    n_comp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = n_comp
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = n_comp
                    stack.append(v)
        n_comp += 1

    def label(v: int, parent: int) -> tuple[int, list[int]]:
        """Label of the subtree at v, and the order it gets cleared in."""
        kids = [c for c in adj[v] if c != parent]
        if not kids:
            return int(rho[v]), [v]

        sub = [label(c, v) for c in kids]
        # decreasing L - w(v,c); w is 1 throughout, so decreasing L. Indexing
        # from the front then counts the siblings still to come, which is the
        # sum_{j>i} of the formula written the other way round.
        sub.sort(key=lambda t: -t[0])

        best = int(rho[v] + len(kids))
        for i, (lc, _oc) in enumerate(sub):
            best = max(best, lc + i)
        # v first, then the cheap subtrees, then the expensive one
        order = [v]
        for _lc, oc in reversed(sub):
            order.extend(oc)
        return best, order

    per_root: dict[int, int] = {}
    best_total, best_root, best_order = None, 0, []
    for r in range(n):
        total, order = 0, []
        for c in range(n_comp):
            # a component that does not contain r is entered at its own first
            # vertex; the components are cleared one after another, so the team
            # is the largest of them and not their sum
            start = r if comp[r] == c else int(np.flatnonzero(comp == c)[0])
            lab, o = label(start, -1)
            total = max(total, lab)
            order.extend(o)
        per_root[r] = total
        if best_total is None or total < best_total:
            best_total, best_root, best_order = total, r, order

    return TreeLabel(robots=int(best_total), root=int(best_root),
                     order=best_order, per_root=per_root, components=n_comp)


# ---------------------------------------------------------------------------
# an executable schedule
# ---------------------------------------------------------------------------


def frontier(scene: Scene, cleared: np.ndarray) -> np.ndarray:
    """Cells of `cleared` with a walkable neighbour outside it.

    The only thing that has to be watched. A cell deep inside the cleared
    region cannot be reached without crossing the frontier; a cell on the
    frontier is where the flood comes back in.
    """
    a, b = scene.evader_edges
    out = np.zeros(scene.n_cells, dtype=bool)
    ca, cb = cleared[a], cleared[b]
    out[a[ca & ~cb]] = True
    out[b[cb & ~ca]] = True
    return out


def cover(scene: Scene, target: np.ndarray, must: list[int]) -> tuple[list[int], int]:
    """Greedy set cover of `target` by detection sets, seeded with `must`.

    Greedy rather than exact: set cover is NP-hard and the ln(n) guarantee is
    the standard thing to reach for. The cover is recomputed from scratch every
    step -- a guard held because the frontier ran past it four steps ago is a
    robot the schedule pays for and does not use.
    """
    need = target.copy()
    chosen = list(must)
    for v in chosen:
        need[scene.detection[v]] = False
    while need.any():
        best, gain = -1, 0
        for v in range(scene.n_nodes):
            if v in chosen:
                continue
            g = int(need[scene.detection[v]].sum())
            if g > gain:
                best, gain = v, g
        if best < 0:
            break
        chosen.append(best)
        need[scene.detection[best]] = False
    return chosen, int(need.sum())


def sweep_order(scene: Scene) -> list[int]:
    """Vertices ordered along the principal axis of the walkable surface.

    Sweeping keeps the cleared region a single advancing band, so there is one
    front and it is a line. Taking whichever vertex adds the most new area
    instead sends the region jumping across the site, and then several
    disconnected patches are open at once and every one has a perimeter to
    hold.
    """
    xy = scene.cell_xyz[:, :2]
    centre = xy.mean(axis=0)
    _, V = np.linalg.eigh(np.cov((xy - centre).T))
    axis = V[:, -1]
    proj = (scene.node_xyz[:, :2] - centre) @ axis
    return [int(v) for v in np.argsort(proj)]


@dataclass
class Plan:
    schedule: Schedule
    order: str
    open_frontier: list[int] = field(default_factory=list)

    def notes(self) -> dict:
        return {
            "order": self.order,
            "steps_with_an_open_frontier":
                int(sum(1 for q in self.open_frontier if q)),
            "frontier_cells_left_open": int(sum(self.open_frontier)),
        }


def plan(scene: Scene, order: str = "kolling", verbose: bool = False) -> Plan:
    """Grow a cleared region and hold its frontier, vertex by vertex.

    `order` picks the sequence vertices are taken in: `kolling` follows the
    spanning-tree recursion above, `sweep` follows the principal axis.

    The planner tracks the state the verifier will be in, step by step, by
    running the same flood. Planning against a cleared region it does not
    actually hold is the one way this can be quietly wrong.

    It does NOT decide whether the schedule clears. That is `verify.propagate`,
    which is given the schedule and believes only the cells.
    """
    if order == "kolling":
        tree, rest = spanning_tree(scene)
        seq = label_tree(scene, tree, rest).order
    elif order == "sweep":
        seq = sweep_order(scene)
    else:
        raise ValueError(f"unknown order {order!r}")

    C = np.ones(scene.n_cells, dtype=bool)      # contaminated
    steps: list[tuple[int, ...]] = []
    open_frontier: list[int] = []

    for k, v in enumerate(seq):
        if not C[scene.detection[v]].any():
            continue                            # nothing left for this vertex
        want = ~C
        want[scene.detection[v]] = True
        guards, missed = cover(scene, frontier(scene, want), [v])
        steps.append(tuple(sorted(guards)))
        open_frontier.append(missed)

        O = scene.observed(guards)
        C = flood(scene, C & ~O, ~O)
        if verbose:
            print(f"    step {len(steps):>3} (vertex {v:>3}): "
                  f"team {len(guards):>2}, contaminated {int(C.sum()):>8,}"
                  + (f", frontier open on {missed:,} cells" if missed else ""),
                  flush=True)

    return Plan(Schedule(steps, name=f"kolling/{order}"), order, open_frontier)
