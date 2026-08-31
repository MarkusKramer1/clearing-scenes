#!/usr/bin/env python3
"""Kolling et al. (2010) as published, on every scene, in all three variants.

Per scene and per cycle-edge variant it draws `--trees` random depth-first
spanning forests, solves each with the no-sliding label recursion, converts it
to a graph strategy, and keeps the cheapest. The best strategy of each variant
is then handed to the cell-level verifier, which knows nothing about guard
regions and judges it on the surface.

    python scripts/run_gsst.py [--scenes christ-church ...] [--trees 100]

Writes `examples/<scene>.gsst.yaml` per scene and prints the comparison table
the paper's first finding is about: does dropping shady edges save searchers?

TWO NUMBERS PER RUN, AND THEY ANSWER DIFFERENT QUESTIONS. `team` is the peak of
the graph strategy -- the paper's own output, a statement about the guard graph
under vertex contamination. `n_searchers` in the verification block is the same
strategy's peak as the cell verifier counts it, and `cleared` is whether the
surface is actually clear at the end. The paper never computes the second, and
the gap between the two is a property of the abstraction, not of the planner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np                                      # noqa: E402

from clearing import (clock as clock_mod, execute,       # noqa: E402
                      gsst, scene as scene_mod, verify)


def fmt(x, nd=2):
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def run_one(name: str, trees: int, seed: int, cells: bool, verbose: bool,
            drive: bool = False, dt: float = 1.0, sub: int = 8,
            absorb_m2: float = 1.0, exact: bool = False) -> dict:
    s = scene_mod.load(name)
    n_shady = int(s.edge_shady.sum())
    print(f"[gsst] {name}: {s.n_vertices} vertices, {len(s.edge_ij)} edges "
          f"({n_shady} shady)", flush=True)

    runs = []
    for variant in gsst.VARIANTS:
        r = gsst.run(s, variant=variant, n_trees=trees, seed=seed, verbose=verbose)
        summary = r.summary()
        line = (f"       {variant:>15}: min {summary['searchers_min']:>3}  "
                f"mean {summary['searchers_mean']:>6}  sd {summary['searchers_sd']:>5}  "
                f"({summary['steps']} steps)")
        entry = {"variant": variant, "result": r, "summary": summary}
        if cells:
            v = verify.propagate(s, r.strategy)
            entry["cells"] = v.metrics(s)
            line += (f"   cells: peak {entry['cells']['n_searchers']}, "
                     f"{'cleared' if entry['cells']['cleared'] else 'residual '
                        + fmt(entry['cells']['residual_m2']) + ' m2'}")
        print(line, flush=True)
        if drive:
            rep = execute.sequentialise(s, r.strategy, absorb_m2=absorb_m2,
                                        verbose=verbose)
            tl = clock_mod.timeline_propagate(
                s, rep.strategy, rep.roster, dt=dt, sub=sub, rule="strict",
                absorb_m2=absorb_m2, exact=exact, verbose=verbose)
            entry["driven"] = rep.metrics()
            entry["driven_clock"] = tl
            print(f"       {'':>15}  driven: {entry['driven']['legs']} legs, "
                  f"fleet {entry['driven']['fleet']}, "
                  f"{entry['driven']['legs_holding']} holding, "
                  f"{entry['driven']['legs_conceding']} conceding, "
                  f"{entry['driven']['machines_spent']} spent"
                  f"   |  strict clock: residual {fmt(tl['residual_m2'])} m2, "
                  f"avoidable {fmt(tl['residual_avoidable_m2'])} m2, "
                  f"mission {tl['mission_seconds']:,.0f} s", flush=True)
        runs.append(entry)
    return {"scene": s, "runs": runs, "trees": trees, "seed": seed,
            "driven": drive}


def write_yaml(path: Path, out: dict) -> None:
    s = out["scene"]
    L: list[str] = []
    add = L.append

    add(f"# {s.title} -- Kolling et al. (2010) as published")
    add("#")
    add("# GSST over random depth-first spanning forests of the guard graph:")
    add("# the Barriere et al. label recursion with sliding forbidden, then the")
    add("# tree strategy converted back to a graph strategy by stationing a")
    add("# searcher wherever a cycle edge leads into a contaminated vertex.")
    add("# Produced by scripts/run_gsst.py.")
    add("")
    add(f"scene: {s.name}")
    add(f"scenario_file: ../scenes/{s.name}.yaml")
    add(f"trees_per_variant: {out['trees']}")
    add(f"seed: {out['seed']}")
    add("")
    add("graph:")
    add(f"  vertices: {s.n_vertices}")
    add(f"  edges: {len(s.edge_ij)}")
    add(f"  edges_regular: {int((~s.edge_shady.astype(bool)).sum())}")
    add(f"  edges_shady: {int(s.edge_shady.sum())}")
    add("")
    add("variants:")
    add("  # searchers_* are the peak of the GRAPH strategy over the forests")
    add("  # drawn -- the paper's own output.")
    add("  # The paper's own output; a claim about the guard graph under vertex")
    add("  # contamination, not about the surface.")
    for run in out["runs"]:
        m = run["summary"]
        add(f"  - variant: {run['variant']}")
        for k, v in m.items():
            if k == "variant":
                continue
            add(f"    {k}: {v}")
        if "cells" in run:
            add("    verified_on_cells:")
            add("      # the same strategy, judged by clearing.verify, which is")
            add("      # told nothing about guard regions or shadiness. Steps")
            add("      # are instantaneous here: nobody is ever in transit.")
            for k, v in run["cells"].items():
                add(f"      {k}: {v}")
        if "driven" in run:
            add("    driven_no_sliding:")
            add("      # the same strategy executed one MACHINE at a time, with")
            add("      # the 2010 prohibition kept: a mover in transit is")
            add("      # credited with nothing, so a drive is admissible only")
            add("      # when the standing team covers the frontier by itself.")
            for k, v in run["driven"].items():
                add(f"      {k}: {v}")
            add("    driven_on_the_clock:")
            add("      # clearing.clock, strict rule: a cell survives an")
            add("      # interval only if it is watched at every instant of it.")
            for k, v in run["driven_clock"].items():
                if k in ("contaminated_after_step", "exposure"):
                    continue
                add(f"      {k}: {v}")
        add("    steps_list:")
        add("      # one line per step: the vertices held")
        for t, step in enumerate(run["result"].best.steps):
            add(f"      - {{t: {t:>3}, machines: {len(step):>3}, "
                f"vertices: [{', '.join(str(v) for v in step)}]}}")
        add("")
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--trees", type=int, default=100,
                    help="random spanning forests per variant (paper: 100)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-cells", action="store_true",
                    help="skip cell-level verification of the best strategies")
    ap.add_argument("--drive", action="store_true",
                    help="also execute each best strategy one machine at a time "
                         "under the no-sliding rule, and judge it on a clock")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--sub", type=int, default=8)
    ap.add_argument("--absorb-m2", type=float, default=1.0,
                    help="contaminated fragments smaller than this are "
                         "absorbed at every step; 0 turns it off")
    ap.add_argument("--exact", action="store_true",
                    help="flood once per arrival instead of every dt. Exact "
                         "under the no-sliding rule, where O(t) only changes "
                         "when a mover arrives.")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    names = a.scenes or scene_mod.available()
    table = []
    for name in names:
        out = run_one(name, a.trees, a.seed, not a.no_cells, a.verbose,
                      drive=a.drive, dt=a.dt, sub=a.sub,
                      absorb_m2=a.absorb_m2, exact=a.exact)
        write_yaml(REPO / "examples" / f"{name}.gsst.yaml", out)
        row = {"scene": name, "vertices": out["scene"].n_vertices}
        for run in out["runs"]:
            row[run["variant"]] = run["summary"]["searchers_min"]
            row[run["variant"] + "_mean"] = run["summary"]["searchers_mean"]
            row[run["variant"] + "_sd"] = run["summary"]["searchers_sd"]
            if "cells" in run:
                row[run["variant"] + "_cells"] = run["cells"]
            if "driven" in run:
                row[run["variant"] + "_driven"] = (run["driven"],
                                                   run["driven_clock"])
        table.append(row)

    print()
    print("fewest searchers over the drawn forests, by cycle-edge variant")
    print(f"{'scene':>22} {'n':>4} {'naive':>7} {'regular':>9} "
          f"{'reg+bias':>10} {'saved':>7}")
    for r in table:
        saved = r["naive"] - min(r["regular"], r["regular_biased"])
        print(f"{r['scene']:>22} {r['vertices']:>4} {r['naive']:>7} "
              f"{r['regular']:>9} {r['regular_biased']:>10} {saved:>+7}")
    saved = [r["naive"] - min(r["regular"], r["regular_biased"]) for r in table]
    print(f"{'':>22} {'':>4} {'':>7} {'':>9} {'mean saved':>10} "
          f"{np.mean(saved):>+7.1f}")

    if a.drive:
        print()
        print("driven one machine at a time, no sliding, on the strict clock")
        print(f"{'scene':>22} {'variant':>15} {'fleet':>6} {'spent':>6} "
              f"{'conceding':>10} {'residual m2':>12} {'avoidable':>10} "
              f"{'tol':>4} {'mission s':>10}")
        for r in table:
            for variant in gsst.VARIANTS:
                d = r.get(variant + "_driven")
                if d is None:
                    continue
                dr, tl = d
                print(f"{r['scene']:>22} {variant:>15} {dr['fleet']:>6} "
                      f"{dr['machines_spent']:>6} {dr['legs_conceding']:>10} "
                      f"{tl['residual_m2']:>12,.2f} "
                      f"{tl['residual_avoidable_m2']:>10,.2f} "
                      f"{'yes' if tl['cleared_within_tolerance'] else 'NO':>4} "
                      f"{tl['mission_seconds']:>10,.0f}")


if __name__ == "__main__":
    main()
