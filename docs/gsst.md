# Kolling et al. (2010) as published, and what it does on these scenes

`docs/kolling.md` describes the planner this repository shipped first: a
GRAPH-CLEAR label for a bound, and a greedy cover of the **cell** frontier for
a schedule. It is a good planner and it is not the 2010 paper's.

This page is the paper's own machinery, implemented in `clearing/gsst.py` and
run by `scripts/run_gsst.py`. It exists to answer one question — *how far does
the published method get on real sites* — and the answer is not the one the
team sizes suggest.

---

## What was implemented

The paper's architecture is deliberate: **all the terrain realism is absorbed
before the graph exists, and the clearing theory is then reused almost
unchanged.** Steps 1–3 already live in `e1-clearing-graph` and ship in
`scenes/`; this is step 4.

| step | status |
|---|---|
| sample positions greedily until free space is covered | upstream, unchanged |
| detection sets `D(p)` from sensor height, target height, range | upstream, unchanged |
| guard regions `G_ij = dD(p_i) ∩ D(p_j)`; regular vs shady | upstream, unchanged |
| **strategy on the graph** | **`clearing/gsst.py`** |

The strategy layer is the other lineage from GRAPH-CLEAR, exactly as the paper
specifies it:

* **contamination on vertices**, following Hollinger et al.'s robotics-adapted
  edge search rather than classical Parsons edge search;
* the **label-based tree recursion of Barrière et al.**, which yields contiguous
  strategies without recontamination;
* **sliding forbidden.** A robot driving between two strategic locations cannot
  promise its path covers the detection-set boundaries on the way, so only
  `place` and `remove` survive. The label pays for it in the leaf case:

  ```
  classical   lambda = max{rho_1, rho_2 + 1}
  2.5D        lambda = rho_1 + 1              if rho_1 = 1
                       max{rho_1, rho_2 + 1}  otherwise
  ```

* the **GSST anytime layer**: draw many random depth-first spanning forests,
  solve each, convert back to a graph strategy by stationing a robot wherever a
  cycle edge leads into a contaminated vertex, keep the cheapest.

`tests/test_gsst.py` checks the pairing that carries the whole thing: the move
sequence `tree_strategy` emits spends *exactly* the label `label_no_slide`
predicts, on every rooting of every hand-checkable tree, and clears it.

### Where the guard rule reads its contamination

The conversion has one subtlety worth stating, because getting it wrong is
silent. Contamination for the cycle-edge rule is read off the **tree**, not off
the graph. On the graph a door vertex is already contaminated by the time you
look at it — the flood came through the door. The tree state is the strategy's
*intention*, and the guards are what make the graph honour it.

It is also what makes the conversion sound rather than merely plausible. Any
edge from a tree-contaminated vertex to a tree-clear one is either a tree edge,
and the tree flood would have crossed it unless the far end is occupied by the
step itself, or a cycle edge, and the rule has just put a robot on the far end.
Either way the boundary is occupied and the containment survives the step.

---

## Finding 1: dropping shady edges helps, and by the published amount

The paper compares three treatments of cycle edges and reports that using only
regular ones is worth **2–3 robots in the minimum across all tested
conditions**. Over 100 random forests per variant per scene:

| scene | n | naive | regular | regular + biased tree | saved |
|---|--:|--:|--:|--:|--:|
| blenheim-palace | 36 | 11 | 9 | 9 | +2 |
| bodleian-library | 112 | 20 | 17 | 17 | +3 |
| christ-church | 62 | 10 | 8 | 8 | +2 |
| hb-allen-centre | 20 | 9 | 8 | 8 | +1 |
| keble-college | 24 | 7 | 6 | **5** | +2 |
| observatory-quarter | 39 | 14 | 11 | 13 | +3 |
| | | | | **mean** | **+2.2** |

**Reproduced.** +1 to +3, mean +2.2, on geometry the authors never saw.

The **variance** claim reproduces only partly. Standard deviation over the 100
draws falls from naive to regular on four scenes (4.47→3.46, 1.92→1.59,
2.82→2.10, 2.24→2.38 is a rise) and rises on two. Lower variance is the trend,
not the rule, at this sample size.

**The biased tree does not pay.** The paper's third variant wins once
(keble-college, 6→5), loses once (observatory-quarter, 11→13), ties four times,
and has a *worse* mean team on five of six scenes. On this scene class,
biasing the depth-first construction toward regular edges is not worth
implementing; dropping shady cycle edges is.

---

## Finding 2: the graph clears, and the ground does not

Every schedule above provably clears the **guard graph** under vertex
contamination — `gsst.graph_clears` is an independent check and it passes for
every draw reported. The paper stops here. This repository has a cell-level
verifier that was told nothing about guard regions, so it does not have to.

| scene | GSST naive | GSST regular | GSST reg+bias | cell-frontier planner (`kolling.py`) |
|---|--:|--:|--:|--:|
| blenheim-palace | 11 **clears** | 9 · 8 247 m² left | 9 · 8.8 m² left | 8 **clears** |
| bodleian-library | 20 · 10 620 m² | 17 · 10 699 m² | 17 · 10 717 m² | 22 · 17.6 m² |
| christ-church | 10 · 7 129 m² | 8 · 7 692 m² | 8 · 7 692 m² | 12 · 6.3 m² |
| hb-allen-centre | 9 · 1 078 m² | 8 · 940 m² | 8 · 1 131 m² | 10 · 4.2 m² |
| keble-college | 7 **clears** | 6 · 5 982 m² | 5 · 4 449 m² | 5 **clears** |
| observatory-quarter | 14 · 2 702 m² | 11 · 2 702 m² | 13 · 2 702 m² | 13 · 4.5 m² |

Christ Church is 8 133 m² of walkable surface and the graph strategy leaves
7 692 m² of it contaminated. This is **not** a data artefact: `all vertices
occupied at once` leaves only 4–18 m² on these scenes, all of it cells no
vertex can see, so the surface is clearable over this vertex set and this
strategy is what fails to clear it.

<!-- the honest reading of the two tables together -->
**The team sizes in the two halves of that table are not comparable, and it
would be easy to read them as if they were.** GSST needs 8 robots at Christ
Church against the cell planner's 12 — but 8 robots that do not clear the site
are not cheaper than 12 that do. Finding 1 is a real reproduction of a real
result *about the graph*. Finding 2 is why the number it improves is not yet a
number about the ground.

---

## Why: an edge is not a door, it is part of one

The measurement is `scripts/rim_coverage.py`, and it is a property of the
exported geometry alone — no strategy, no schedule, no seed.

An edge exists when `G_ij = dD(p_i) ∩ D(p_j)` is non-empty: a robot at `j` can
watch **some** of the rim of `D(p_i)`. Every graph-level clearing argument then
treats *"i is clear and a neighbour is occupied"* as enough to keep it clear.
So: how much of the rim is *some*?

| scene | n | best neighbour (median) | all neighbours at once | complete | **every vertex at once** | complete |
|---|--:|--:|--:|--:|--:|--:|
| blenheim-palace | 36 | 0.63 | 0.94 | 0/36 | **1.00** | 36/36 |
| bodleian-library | 112 | 0.51 | 0.95 | 0/112 | **1.00** | 112/112 |
| christ-church | 62 | 0.54 | 0.85 | 0/62 | **1.00** | 62/62 |
| hb-allen-centre | 20 | 0.63 | 0.86 | 0/20 | **1.00** | 20/20 |
| keble-college | 24 | 0.77 | 0.93 | 0/24 | **1.00** | 24/24 |
| observatory-quarter | 39 | 0.57 | 0.88 | 0/39 | **1.00** | 39/39 |
| **all** | **293** | **0.54** | **0.92** | **0/293** | **1.00** | **293/293** |

The median neighbour watches **54%** of the rim it is supposed to guard, and
occupying *every* neighbour of a vertex simultaneously **still** leaves a gap —
on all 293 vertices of all six sites. Not one rim in the corpus is closed by
the graph.

**The last column is the control, and it decides who is to blame.** Every one
of those 293 rims *is* closed by the vertex set — median 1.00, minimum 1.00,
293 of 293 complete. So the sampling is sound and there is nothing wrong with
where the vertices are. What fails is the **edge relation**: a rim needs
several watchers at once, an edge names them one at a time, and no amount of
discharging edge obligations assembles the set. A set cover over vertices can
hold a boundary that no set of edges describes.

And this is exactly where the tree strategy dies. `dD(p_i)` is by definition a
subset of `D(p_i)`, so **a robot standing at `i` watches the whole of its own
rim** — while it is there, `D(p_i)` cannot be re-entered at all. The gap opens
the moment it leaves, which is precisely what a tree strategy does: `clear(v)`
returns with no robot left anywhere in the subtree, on the argument that the
parent edge is the only way back in. On the graph that argument is sound. On
the surface the vertex was holding a rim that its neighbours cannot between
them close, and nobody is holding it now.

So the guard graph records **who can watch a piece of a boundary**, never
**whether the boundary is closed**. A strategy that discharges every edge
obligation has still left a hole in every detection set it cleared, and an
arbitrarily fast evader needs one hole.

The gap is therefore not a tuning failure of GSST, and it is not a sampling
failure either. It is the abstraction: the graph is a faithful record of *who
can see part of what*, and clearing needs *what it takes to close a boundary*.
Those are different questions, and the second one is a set cover.

This is the structural reason `clearing/kolling.py` covers the **cell**
frontier with a set cover over vertices instead of discharging edges: a set
cover can hold a boundary that no single edge, and no set of edges, describes.

---

## What was not run

**The sensing-range sweep.** The paper's second finding — that on complex
terrain a *longer* sensing range needs *more* robots, because it complicates
the detection-set boundary rather than enlarging it — needs the same scenes
exported at several values of `s_r`. These ship at `R = 30 m` only, and
re-exporting requires the survey clouds and the raycaster in
`e1-clearing-graph` (`scripts/export_from_e1.py --R ...`), whose `data/` is
empty here. It is the most interesting thing left undone, and the rim-coverage
table above predicts its mechanism would show up here too.

**Heterogeneity.** No UAV, as in the rest of this repository. The 2010 paper is
homogeneous and this is exactly its setting.

**Cell-level components.** Guard-graph components are cleared in sequence, which
is sound on the graph, where a component has no edges leaving it. Two graph
components can still be adjacent on the walkable surface. Where that happens it
is one more contributor to the residual above, and it is not separated out.

---

## Reproduction

```bash
python scripts/run_gsst.py --trees 100          # both tables, ~6 min
python scripts/rim_coverage.py                  # the third, seconds
python -m pytest tests/test_gsst.py
```

Per-scene schedules and every metric land in `examples/<scene>.gsst.yaml`.
