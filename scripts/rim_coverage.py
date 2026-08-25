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


def coverage(s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    everything = np.zeros(s.n_cells, dtype=bool)
    for d in s.detection:
        everything[d] = True
    best = np.zeros(s.n_nodes)
    allof = np.zeros(s.n_nodes)
    ceil = np.zeros(s.n_nodes)
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
        ceil[i] = int(everything[rim].sum()) / rim.size
    return best, allof, ceil


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    a = ap.parse_args()

    hdr = (f"{'scene':>22} {'n':>4} | {'best neighbour':>16} | "
           f"{'all neighbours':>16} {'complete':>9} | "
           f"{'every vertex':>14} {'complete':>9}")
    print(hdr)
    print(f"{'':>22} {'':>4} | {'median':>7} {'mean':>8} | "
          f"{'median':>7} {'mean':>8} {'':>9} | {'median':>6} {'min':>7} {'':>9}")
    rows = []
    for name in (a.scenes or scene_mod.available()):
        s = scene_mod.load(name)
        b, al, ce = coverage(s)
        rows.append((b, al, ce))
        print(f"{name:>22} {s.n_nodes:>4} | {np.median(b):>7.2f} {b.mean():>8.2f} | "
              f"{np.median(al):>7.2f} {al.mean():>8.2f} "
              f"{int((al > 0.999).sum()):>4}/{len(al):<4} | "
              f"{np.median(ce):>6.2f} {ce.min():>7.2f} "
              f"{int((ce > 0.999).sum()):>4}/{len(ce):<4}")
    b = np.concatenate([r[0] for r in rows])
    al = np.concatenate([r[1] for r in rows])
    ce = np.concatenate([r[2] for r in rows])
    print(f"{'all scenes':>22} {len(b):>4} | {np.median(b):>7.2f} {b.mean():>8.2f} | "
          f"{np.median(al):>7.2f} {al.mean():>8.2f} "
          f"{int((al > 0.999).sum()):>4}/{len(al):<4} | "
          f"{np.median(ce):>6.2f} {ce.min():>7.2f} "
          f"{int((ce > 0.999).sum()):>4}/{len(ce):<4}")


if __name__ == "__main__":
    main()
