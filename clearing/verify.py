"""Does a schedule actually clear the scene? The contamination-set verifier.

THE MODEL
---------
State is the set of cells that could still hold the evader. It starts as the
whole walkable surface. A schedule is a sequence of steps; each step names the
vertices that are occupied. Per step:

    O := union of D(v) over the occupied vertices
    C := C \\ O                    what is seen right now is clear
    C := flood(C, allowed = ~O)   the evader runs anywhere it can still reach

and the schedule clears the scene iff C is empty at the end.

THE EVADER IS NEVER SIMULATED. There is no agent and no random seed. The
guarantee is the emptiness of C, which quantifies over every arbitrarily fast,
omniscient evader at once.

WHY THE FLOOD IS CONNECTED COMPONENTS AND NOT ITERATED DILATION
---------------------------------------------------------------
"Arbitrarily fast" means the evader crosses any distance between two steps, so
contamination is not dilated by one cell per step -- it fills every cell of ~O
reachable from a contaminated one. That is exactly a connected-component
labelling of the subgraph induced on ~O: one pass, and it cannot stop early the
way a capped dilation loop can.

The verifier is deliberately thinner than the graph. It knows cells, adjacency
and detection sets -- not guard regions, not the regular/shady classification.
That is the point: if a graph-level strategy claims to clear and this reports a
residual, the guard-region abstraction is wrong for that geometry, and the
verifier has to be able to say so without assuming the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .scene import Scene


@dataclass
class Schedule:
    """A sequence of steps, each naming the occupied vertices."""

    steps: list[tuple[int, ...]]
    name: str = ""

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def team_peak(self) -> int:
        return max((len(s) for s in self.steps), default=0)

    def validate(self, scene: Scene) -> None:
        for t, step in enumerate(self.steps):
            for v in step:
                if not 0 <= int(v) < scene.n_nodes:
                    raise ValueError(
                        f"step {t}: vertex {v} is not in the graph "
                        f"(0..{scene.n_nodes - 1})")


@dataclass
class Result:
    contaminated: list[np.ndarray] = field(default_factory=list)  # per step
    residual: np.ndarray = None
    team: list[int] = field(default_factory=list)
    recontaminated: list[int] = field(default_factory=list)
    makespan_s: float = 0.0

    @property
    def cleared(self) -> bool:
        return not bool(self.residual.any())

    def cleared_except_uncoverable(self, scene: Scene) -> bool:
        """Cleared apart from cells no vertex of this graph can ever see.

        The honest weaker claim. `uncoverable` is a property of the sampled
        vertex set, not a theorem about the geometry -- a denser sampling could
        shrink it -- so it is reported as a tolerance and never folded silently
        into `cleared`.
        """
        return not bool((self.residual & scene.coverable).any())

    @property
    def recontamination_events(self) -> int:
        return int(sum(1 for c in self.recontaminated if c))

    def metrics(self, scene: Scene) -> dict:
        return {
            "steps": len(self.team),
            "team_peak": int(max(self.team)) if self.team else 0,
            "team_median": int(np.median(self.team)) if self.team else 0,
            "cleared": self.cleared,
            "cleared_except_uncoverable": self.cleared_except_uncoverable(scene),
            "residual_cells": int(self.residual.sum()),
            "residual_m2": round(scene.area(self.residual), 2),
            "recontamination_events": self.recontamination_events,
            "recontaminated_cells": int(sum(self.recontaminated)),
            "makespan_s": round(self.makespan_s, 1),
        }


# ---------------------------------------------------------------------------
# the flood
# ---------------------------------------------------------------------------


def _graph(scene: Scene) -> sparse.csr_matrix:
    a, b = scene.evader_edges
    n = scene.n_cells
    g = sparse.coo_matrix((np.ones(a.size, np.uint8), (a, b)), shape=(n, n)).tocsr()
    return g + g.T


def flood(scene: Scene, seed: np.ndarray, allowed: np.ndarray,
          graph: sparse.csr_matrix | None = None) -> np.ndarray:
    """Every cell of `allowed` reachable from a `seed` cell, to a fixed point."""
    g = _graph(scene) if graph is None else graph
    idx = np.flatnonzero(allowed)
    if idx.size == 0:
        return np.zeros(scene.n_cells, dtype=bool)
    n_comp, lab = csgraph.connected_components(g[idx][:, idx], directed=False)
    hot = np.zeros(n_comp, dtype=bool)
    hot[lab[seed[idx]]] = True
    out = np.zeros(scene.n_cells, dtype=bool)
    out[idx[hot[lab]]] = True
    return out


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------


def propagate(scene: Scene, schedule: Schedule, verbose: bool = False) -> Result:
    schedule.validate(scene)
    g = _graph(scene)

    C = ~scene.excluded          # excluded cells are never contaminated
    res = Result()
    prev = C.copy()
    for t, step in enumerate(schedule.steps):
        O = scene.observed(step)
        C = flood(scene, C & ~O, ~O, graph=g)
        res.contaminated.append(C.copy())
        res.team.append(len(step))
        res.recontaminated.append(int((C & ~prev).sum()))
        prev = C
        if verbose and (t + 1) % 10 == 0:
            print(f"    step {t + 1}: team {len(step)}, "
                  f"contaminated {int(C.sum()):,}", flush=True)
    res.residual = C
    res.makespan_s = makespan(scene, schedule)
    return res


# ---------------------------------------------------------------------------
# how long it takes
# ---------------------------------------------------------------------------


def makespan(scene: Scene, schedule: Schedule) -> float:
    """Wall-clock time of the schedule, from the travel times in the YAML.

    Robots already on site are matched to the vertices of the next step by
    minimum total travel (`scipy.optimize.linear_sum_assignment`); a robot that
    keeps its vertex travels nothing. When a step needs more robots than are on
    site, the extra ones walk on from the entry point -- the first vertex of the
    first step -- which is what deploying a team onto a site looks like and what
    keeps a schedule whose team only grows from costing nothing at all. A step
    takes as long as its slowest robot.

    This is the schedule's SECONDARY cost. It does not enter the guarantee at
    all: the evader is arbitrarily fast, so whether the scene is cleared is a
    combinatorial statement about the detection sets and carries no time.
    """
    from scipy.optimize import linear_sum_assignment

    if not schedule.steps:
        return 0.0
    T = np.asarray(scene.travel_seconds, dtype=float)
    BIG = 1e9
    entry = int(schedule.steps[0][0])       # where the team walks on to the site

    def cost_to(v: int, frm: int | None) -> float:
        c = T[entry if frm is None else frm, v]
        return BIG if not np.isfinite(c) else float(c)

    total, held = 0.0, []
    for step in schedule.steps:
        want = list(step)
        # Robots already on site are matched to this step's vertices by least
        # total travel; whatever is left over is a robot walking on to the site
        # at `entry`. A robot that keeps its vertex travels nothing.
        n, m = len(held), len(want)
        size = max(n, m)
        cost = np.zeros((size, size))
        for j, v in enumerate(want):
            for i in range(size):
                cost[i, j] = cost_to(v, held[i] if i < n else None)
        r, c = linear_sum_assignment(cost)
        moved = [cost[i, j] for i, j in zip(r, c) if j < m and cost[i, j] < BIG]
        total += max(moved) if moved else 0.0
        held = want
    return float(total)


def all_at_once(scene: Scene) -> Schedule:
    """Every vertex occupied at once, forever. The feasibility ceiling.

    Not a comparison: an unlimited team held simultaneously. A residual that
    survives this survives every schedule over this vertex set, of any size.
    """
    return Schedule([tuple(range(scene.n_nodes))], name="all vertices at once")
