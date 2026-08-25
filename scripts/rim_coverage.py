#!/usr/bin/env python3
"""How much of a detection set's rim can its neighbours actually watch?

An edge of the guard graph says that a robot at `j` can watch SOME of the rim
of `D(p_i)` -- the guard region `G_ij = dD(p_i) ∩ D(p_j)` is non-empty. Every
graph-level clearing argument then treats "i is clear and a neighbour is
occupied" as enough to keep it clear.

This script measures the step that argument skips: what FRACTION of dD(p_i) a
neighbour covers. If the best single neighbour covers half the rim, a graph
strategy that guards one edge at a time is guarding half a door.

    python scripts/rim_coverage.py [--scenes christ-church ...]

The quantity is a property of the exported geometry alone. No strategy, no
schedule and no random seed enter it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np                                  # noqa: E402

from clearing import scene as scene_mod             # noqa: E402


def coverage(s) -> tuple[np.ndarray, np.ndarray]:
    """Per vertex: the fraction of its rim covered by the best single
    neighbour, by all of its GRAPH NEIGHBOURS at once, and by the whole vertex
    set at once.

    The third column is the control, and it decides who is to blame. If the
    whole vertex set covers a rim that the neighbours do not, the sampling is
    sound and the EDGE RELATION is what fails: a set cover over vertices can
    hold a boundary that no set of edges describes. If even the vertex set
    falls short, no schedule over it could have held that rim and the planner
    is exonerated instead.
    """
    nb: list[list[int]] = [[] for _ in range(s.n_nodes)]
    for k in range(len(s.edge_ij)):
        i, j = (int(v) for v in s.edge_ij[k])
        nb[i].append(j)
        nb[j].append(i)

    seen = np.zeros(s.n_cells, dtype=bool)
    best = np.zeros(s.n_nodes)
    allof = np.zeros(s.n_nodes)
    for i in range(s.n_nodes):
        rim = s.rim[i]
        if rim.size == 0:
            continue
        union = np.zeros(rim.size, dtype=bool)
        top = 0
        for j in nb[i]:
            seen[s.detection[j]] = True
            hit = seen[rim]
            top = max(top, int(hit.sum()))
            union |= hit
            seen[s.detection[j]] = False
        best[i] = top / rim.size
        allof[i] = int(union.sum()) / rim.size
    return best, allof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    a = ap.parse_args()

    print(f"{'scene':>22} {'n':>4} | {'best single neighbour':>19} | "
          f"{'EVERY neighbour at once = every other vertex':>45}")
    print(f"{'':>22} {'':>4} | {'median':>8} {'mean':>9} | "
          f"{'median':>8} {'mean':>9} {'min':>8} {'closed':>10}")
    rows = []
    for name in (a.scenes or scene_mod.available()):
        s = scene_mod.load(name)
        b, al = coverage(s)
        rows.append((b, al))
        print(f"{name:>22} {s.n_nodes:>4} | {np.median(b):>8.2f} {b.mean():>9.2f} | "
              f"{np.median(al):>8.2f} {al.mean():>9.2f} {al.min():>8.2f} "
              f"{int((al > 0.999).sum()):>4}/{len(al):<5}")
    b = np.concatenate([r[0] for r in rows])
    al = np.concatenate([r[1] for r in rows])
    print(f"{'all scenes':>22} {len(b):>4} | {np.median(b):>8.2f} {b.mean():>9.2f} | "
          f"{np.median(al):>8.2f} {al.mean():>9.2f} {al.min():>8.2f} "
          f"{int((al > 0.999).sum()):>4}/{len(al):<5}")
    print()
    print("The second block is the ceiling, not just a neighbour statistic: a "
          "vertex that\nwatches any part of dD(i) has a non-empty guard region "
          "with i and is therefore\na neighbour of i. So no set of vertices "
          "OTHER THAN i closes i's rim -- which is\nwhy D(i) cannot be held "
          "once the robot standing on i leaves.")


if __name__ == "__main__":
    main()
