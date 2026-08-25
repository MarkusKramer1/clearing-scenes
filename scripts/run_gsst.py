#!/usr/bin/env python3
"""Kolling et al. (2010) as published, on every scene, in all three variants.

Per scene and per cycle-edge variant it draws `--trees` random depth-first
spanning forests, solves each with the no-sliding label recursion, converts it
to a graph strategy, and keeps the cheapest. The best schedule of each variant
is then handed to the cell-level verifier, which knows nothing about guard
regions and judges it on the surface.

    python scripts/run_gsst.py [--scenes christ-church ...] [--trees 100]

Writes `examples/<scene>.gsst.yaml` per scene and prints the comparison table
the paper's first finding is about: does dropping shady edges save robots?

TWO NUMBERS PER RUN, AND THEY ANSWER DIFFERENT QUESTIONS. `team` is the peak of
the graph strategy -- the paper's own output, a statement about the guard graph
under vertex contamination. `team_peak` in the verification block is the same
schedule's peak as the cell verifier counts it, and `cleared` is whether the
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

from clearing import gsst, scene as scene_mod, verify    # noqa: E402


def fmt(x, nd=2):
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def run_one(name: str, trees: int, seed: int, cells: bool, verbose: bool) -> dict:
    s = scene_mod.load(name)
    n_shady = int(s.edge_shady.sum())
    print(f"[gsst] {name}: {s.n_nodes} vertices, {len(s.edge_ij)} edges "
          f"({n_shady} shady)", flush=True)

    runs = []
    for variant in gsst.VARIANTS:
        r = gsst.run(s, variant=variant, n_trees=trees, seed=seed, verbose=verbose)
        summary = r.summary()
        line = (f"       {variant:>15}: min {summary['team_min']:>3}  "
                f"mean {summary['team_mean']:>6}  sd {summary['team_sd']:>5}  "
                f"({summary['steps']} steps)")
        entry = {"variant": variant, "result": r, "summary": summary}
        if cells:
            v = verify.propagate(s, r.schedule)
            entry["cells"] = v.metrics(s)
            line += (f"   cells: peak {entry['cells']['team_peak']}, "
                     f"{'cleared' if entry['cells']['cleared'] else 'residual '
                        + fmt(entry['cells']['residual_m2']) + ' m2'}")
        print(line, flush=True)
        runs.append(entry)
    return {"scene": s, "runs": runs, "trees": trees, "seed": seed}


def write_yaml(path: Path, out: dict) -> None:
    s = out["scene"]
    L: list[str] = []
    add = L.append

    add(f"# {s.title} -- Kolling et al. (2010) as published")
    add("#")
    add("# GSST over random depth-first spanning forests of the guard graph:")
    add("# the Barriere et al. label recursion with sliding forbidden, then the")
    add("# tree strategy converted back to a graph strategy by stationing a")
    add("# robot wherever a cycle edge leads into a contaminated vertex.")
    add("# Produced by scripts/run_gsst.py.")
    add("")
    add(f"scene: {s.name}")
    add(f"scenario_file: ../scenes/{s.name}.yaml")
    add(f"trees_per_variant: {out['trees']}")
    add(f"seed: {out['seed']}")
    add("")
    add("graph:")
    add(f"  vertices: {s.n_nodes}")
    add(f"  edges: {len(s.edge_ij)}")
    add(f"  edges_regular: {int((~s.edge_shady.astype(bool)).sum())}")
    add(f"  edges_shady: {int(s.edge_shady.sum())}")
    add("")
    add("variants:")
    add("  # team_* are the peak of the GRAPH strategy over the drawn forests.")
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
            add("      # the same schedule, judged by clearing.verify, which is")
            add("      # told nothing about guard regions or shadiness")
            for k, v in run["cells"].items():
                add(f"      {k}: {v}")
        add("    steps_list:")
        add("      # one line per step: the vertices held")
        for t, step in enumerate(run["result"].best.steps):
            add(f"      - {{t: {t:>3}, robots: {len(step):>3}, "
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
                    help="skip cell-level verification of the best schedules")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    names = a.scenes or scene_mod.available()
    table = []
    for name in names:
        out = run_one(name, a.trees, a.seed, not a.no_cells, a.verbose)
        write_yaml(REPO / "examples" / f"{name}.gsst.yaml", out)
        row = {"scene": name, "vertices": out["scene"].n_nodes}
        for run in out["runs"]:
            row[run["variant"]] = run["summary"]["team_min"]
            row[run["variant"] + "_mean"] = run["summary"]["team_mean"]
            row[run["variant"] + "_sd"] = run["summary"]["team_sd"]
            if "cells" in run:
                row[run["variant"] + "_cells"] = run["cells"]
        table.append(row)

    print()
    print("minimum team over the drawn forests, by cycle-edge variant")
    print(f"{'scene':>22} {'n':>4} {'naive':>7} {'regular':>9} "
          f"{'reg+bias':>10} {'saved':>7}")
    for r in table:
        saved = r["naive"] - min(r["regular"], r["regular_biased"])
        print(f"{r['scene']:>22} {r['vertices']:>4} {r['naive']:>7} "
              f"{r['regular']:>9} {r['regular_biased']:>10} {saved:>+7}")
    saved = [r["naive"] - min(r["regular"], r["regular_biased"]) for r in table]
    print(f"{'':>22} {'':>4} {'':>7} {'':>9} {'mean saved':>10} "
          f"{np.mean(saved):>+7.1f}")


if __name__ == "__main__":
    main()
