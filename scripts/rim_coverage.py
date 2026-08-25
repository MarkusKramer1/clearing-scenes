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
    neighbour, and by all of its neighbours at once."""
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

    print(f"{'scene':>22} {'n':>4} | {'best single neighbour':>28} | "
          f"{'all neighbours at once':>28} | {'rims':>9}")
    print(f"{'':>22} {'':>4} | {'median':>8} {'mean':>8} {'max':>10} | "
          f"{'median':>8} {'mean':>8} {'min':>10} | {'complete':>9}")
    rows = []
    for name in (a.scenes or scene_mod.available()):
        s = scene_mod.load(name)
        b, al = coverage(s)
        rows.append((b, al))
        print(f"{name:>22} {s.n_nodes:>4} | {np.median(b):>8.2f} {b.mean():>8.2f} "
              f"{b.max():>10.2f} | {np.median(al):>8.2f} {al.mean():>8.2f} "
              f"{al.min():>10.2f} | {int((al > 0.999).sum()):>4}/{len(al):<4}")
    b = np.concatenate([r[0] for r in rows])
    al = np.concatenate([r[1] for r in rows])
    print(f"{'all scenes':>22} {len(b):>4} | {np.median(b):>8.2f} {b.mean():>8.2f} "
          f"{b.max():>10.2f} | {np.median(al):>8.2f} {al.mean():>8.2f} "
          f"{al.min():>10.2f} | {int((al > 0.999).sum()):>4}/{len(al):<4}")


if __name__ == "__main__":
    main()
