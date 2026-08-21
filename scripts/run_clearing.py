#!/usr/bin/env python3
"""Run the Kolling baseline on every scene and write the example clearings.

Per scene it writes `examples/<scene>.clearing.yaml`: the schedule itself --
which vertices are occupied at every step -- plus what the cell-level verifier
made of it. Three runs are recorded:

    kolling   vertices in the order the spanning-tree recursion clears them
    sweep     vertices in the order of the surface's principal axis
    all       every vertex occupied at once, as the feasibility ceiling

`all` is not a comparison. It is an unlimited team held simultaneously, so a
residual that survives it survives every schedule over this vertex set.

    python scripts/run_clearing.py [--scenes christ-church ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from clearing import kolling, scene as scene_mod, verify  # noqa: E402


def fmt(x, nd=2):
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def run_one(name: str, verbose: bool) -> dict:
    s = scene_mod.load(name)
    tree, rest = kolling.spanning_tree(s)
    label = kolling.label_tree(s, tree, rest)

    runs = []
    for order in ("kolling", "sweep"):
        p = kolling.plan(s, order=order, verbose=verbose)
        r = verify.propagate(s, p.schedule)
        runs.append({"key": order, "schedule": p.schedule,
                     "metrics": {**r.metrics(s), **p.notes()}})

    sched = verify.all_at_once(s)
    r = verify.propagate(s, sched)
    runs.append({"key": "all", "schedule": sched, "metrics": r.metrics(s)})

    return {"scene": s, "label": label, "tree": tree, "rest": rest, "runs": runs}


def write_yaml(path: Path, out: dict) -> None:
    s, label = out["scene"], out["label"]
    L: list[str] = []
    add = L.append

    add(f"# {s.title} -- clearing examples")
    add("#")
    add("# Produced by scripts/run_clearing.py and checked cell by cell with")
    add("# clearing.verify.propagate. Every step lists the vertices a robot")
    add("# stands on; a robot sees D(v) from there and the evader may be")
    add("# anywhere it is not seen.")
    add("")
    add(f"scene: {s.name}")
    add(f"scenario_file: ../scenes/{s.name}.yaml")
    add("")
    add("graph_clear_label:")
    add("  # Kolling & Carpin's tree recursion on a maximum spanning forest of")
    add("  # the guard graph, one robot per swept vertex and per blocked edge.")
    add("  # An upper bound in a model that charges every edge separately; the")
    add("  # schedules below cover the frontier with a set cover over vertices")
    add("  # instead, and need fewer.")
    add(f"  robots: {label.robots}")
    add(f"  root_vertex: {label.root}")
    add(f"  tree_edges: {len(out['tree'])}")
    add(f"  non_tree_edges: {len(out['rest'])}")
    add(f"  forest_components: {label.components}")
    add("")
    add("tolerance:")
    add("  # Cells no vertex sees. Small connected fragments of it are taken")
    add("  # out of the evader space (see clearing.scene.Scene.excluded); the")
    add("  # rest stays in and shows up as residual.")
    add(f"  uncoverable_m2: {fmt(s.area(s.uncoverable))}")
    add(f"  excluded_as_specks_m2: {fmt(s.area(s.excluded))}")
    add(f"  speck_max_area_m2: {fmt(s.speck_max_area_m2)}")
    add("")
    add("runs:")
    for run in out["runs"]:
        m = run["metrics"]
        add(f"  - key: {run['key']}")
        add(f"    name: {run['schedule'].name or run['key']}")
        for k, v in m.items():
            add(f"    {k}: {v}")
        add("    steps_list:")
        add("      # one line per step: the vertices held")
        for t, step in enumerate(run["schedule"].steps):
            add(f"      - {{t: {t:>3}, robots: {len(step):>3}, "
                f"vertices: [{', '.join(str(v) for v in step)}]}}")
        add("")
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    names = a.scenes or scene_mod.available()
    summary = []
    for name in names:
        print(f"[clearing] {name}", flush=True)
        out = run_one(name, a.verbose)
        write_yaml(REPO / "examples" / f"{name}.clearing.yaml", out)
        best = next(r for r in out["runs"] if r["key"] == "kolling")
        m = best["metrics"]
        print(f"           label {out['label'].robots} robots; schedule "
              f"{m['steps']} steps, peak {m['team_peak']}, "
              f"cleared_except_uncoverable={m['cleared_except_uncoverable']}, "
              f"residual {m['residual_m2']} m2", flush=True)
        summary.append((name, out["label"].robots, m))

    print()
    print(f"{'scene':22} {'label':>6} {'steps':>6} {'peak':>5} {'median':>7} "
          f"{'residual m2':>12} {'makespan h':>11}")
    for name, lab, m in summary:
        print(f"{name:22} {lab:>6} {m['steps']:>6} {m['team_peak']:>5} "
              f"{m['team_median']:>7} {m['residual_m2']:>12} "
              f"{m['makespan_s'] / 3600:>11.2f}")


if __name__ == "__main__":
    main()
