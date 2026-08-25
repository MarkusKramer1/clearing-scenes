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

| scene | n | best single neighbour (median) | every neighbour at once (median) | min | rims closed |
|---|--:|--:|--:|--:|--:|
| blenheim-palace | 36 | 0.63 | 0.94 | 0.11 | 0/36 |
| bodleian-library | 112 | 0.51 | 0.95 | 0.07 | 0/112 |
| christ-church | 62 | 0.54 | 0.85 | 0.00 | 0/62 |
| hb-allen-centre | 20 | 0.63 | 0.86 | 0.10 | 0/20 |
| keble-college | 24 | 0.77 | 0.93 | 0.07 | 0/24 |
| observatory-quarter | 39 | 0.57 | 0.88 | 0.00 | 0/39 |
| **all** | **293** | **0.54** | **0.92** | **0.00** | **0/293** |

The median neighbour watches **54%** of the rim it is supposed to guard, and
occupying *every* neighbour of a vertex simultaneously **still** leaves a gap —
on all 293 vertices of all six sites.

**The second column is the ceiling, not just a neighbour statistic**, and that
is what makes this structural. A vertex that watches any part of `dD(i)` has a
non-empty guard region with `i` and is therefore *by definition* a neighbour of
`i`: "every neighbour" and "every other vertex" are the same set. So **no set
of vertices other than `i` closes `i`'s rim** — a median 8% of it, and on two
scenes the whole of some rim, is visible from nowhere but `i` itself.

<!-- a column measuring the definition rather than the geometry -->
There is an obvious third column — the rim covered by the *whole* vertex set —
and it is **vacuous**: `dD(i)` is a subset of `D(i)`, so any union that
includes `i` covers it outright and reports 1.00 everywhere. It was in
`rim_coverage.py` for one commit and is recorded here so it is not added back.

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

The gap is therefore not a tuning failure of GSST. **A detection set on this
geometry cannot be held by anybody except the robot standing in it** — so any
strategy whose vertices are regions and whose guards are neighbours is unsound
here, whatever order it visits them in. What can be repaired, and at what
price, is [the next section](#what-would-have-to-change).

This is the structural reason `clearing/kolling.py` covers the **cell**
frontier with a set cover over vertices instead of discharging edges: a set
cover can hold a boundary that no single edge, and no set of edges, describes.

---

## What would have to change

Three repairs follow from the measurement, and they attack it at three
different places. Only the third is cheap.

### 1. Never lift a robot — sound, and it costs the whole vertex set

If no rim can be held by anyone but the vertex itself, the honest fix inside
the 2010 model is to never release a swept vertex. That clears: it is
`verify.all_at_once`, which leaves only the 4–18 m² no vertex can see. But the
release rule was the only thing keeping the team small, so the team becomes the
vertex set: **62 robots at Christ Church against the cell planner's 12.**

This is not a straw man — it is roughly what A7 already does. Its traversal
holds nearly every visited node, it *does* clear the cell model, and it pays
three to five times the cell planner's team. The two results agree, and
together they say the cost of soundness under per-vertex thinking is a factor
of three to five at best.

### 2. Sample so that rims *are* closable — the upstream repair

The gap is a property of **where the vertices are**, and the sampler never
tried to close it: it places positions greedily until free space is *covered*,
which is an art-gallery objective and says nothing about whether each `D(p)`
has its boundary watched by the others. A sampler carrying the extra constraint

> for every vertex `i`, `dD(i)` must be covered by `∪ D(j)` over `j ≠ i`

would make the guard graph mean what the clearing argument assumes it means,
and the 2010 machinery would then be sound on it unchanged.

**Implemented and measured** — `e1-clearing-graph`, `e1/rim_closure.py` and
`scripts/run_rim_closure.py`: add vertices drawn from the open rim itself
(a rim cell nobody else sees is walkable, and a sensor standing on it sees it),
recompute the guard graph, re-run GSST, verify on cells.

| scene | vertices | team, GSST regular | | cell planner | cleared on cells |
|---|--:|--:|--:|--:|:--|
| | before → after | before | after | | before → after |
| hb-allen-centre | 20 → 140 | 7 | **62** | 10 | 940 m² left → **0.0 m²** |
| keble-college | 23 → 143 | 5 | **33** | 5 | 5 982 m² left → **0.0 m²** |
| observatory-quarter | 39 → 283 | 12 | **75** | 13 | 2 702 m² left → **0.0 m²** |

**The repair works.** On every scene tried, the published method goes from
leaving most of the site contaminated to clearing it outright — residual
0.0 m², all three cycle-edge variants, under the unchanged cell propagator.
The 2010 machinery is untouched; only the vertex set changed.

**And it is not affordable.** It costs **5–7× the vertices** and **6–9× the
team**, against a cell-frontier planner that clears the same scenes with 5–13
robots. The mechanism is the paper's own second finding, arriving from a new
direction: more vertices make the guard graph denser — hb-allen-centre goes
from 84 edges to 4 261 — and a graph strategy's cost is driven by the number
of cycle edges it must hold, not by how much each robot sees. **Repairing the
sampling so the abstraction becomes sound makes the instance the abstraction
is expensive on.**

Two honest qualifications:

* **The fixpoint does not close.** Each added vertex brings its own rim. Round
  one is decisive — Christ Church goes from 2 987 open rim cells to 318 for
  100 vertices — and then it stalls: eleven further rounds and 290 more
  vertices only reach 200. Runs stop at a 12-round budget with 60–203 cells
  still open on a third to a half of the vertices. **They cleared anyway**,
  because what is left is 2.4–8.1 m² spread over many rims, at or below the
  `A_min` tolerance the project already applies. So full rim closure was
  sufficient-by-a-margin, not necessary, and the vertex counts above are an
  *upper* bound on what a smarter repair would need.
* **The baseline differs slightly.** This starts from Stage 5's raw ground
  sample; the Stage 6 graph the "before" column quotes has
  `repair_joint_residual` applied on top, which is why Keble reads 23 here and
  24 there, and Christ Church 57 against 62. The repaired graph therefore
  starts from *fewer* vertices than its own baseline — the result is not
  flattered by the difference.

### 3. Hold the frontier of the cleared *region*, not the rims of its parts

This is what `clearing/kolling.py` does, and the reason it wins is arithmetic
rather than cleverness. When two cleared detection sets touch, the shared part
of their rims stops being a boundary — but per-vertex thinking pays for it
anyway. Measured at each scene's peak step, over the vertices whose detection
sets lie wholly inside the cleared region:

| scene | Σ rim cells of the cleared vertices | frontier of their union | cancelled |
|---|--:|--:|--:|
| blenheim-palace | 14 791 | 1 852 | 87% |
| bodleian-library | 93 393 | 2 228 | **98%** |
| christ-church | 23 375 | 368 | **98%** |
| hb-allen-centre | 8 649 | 176 | **98%** |
| keble-college | 5 344 | 991 | 81% |
| observatory-quarter | 18 822 | 751 | 96% |

**81–98% of the boundary a per-vertex rule would guard is interior.** The
region frontier is 2–19% of the sum of the rims, and it is coverable — which is
why the cell planner clears Christ Church with 12 robots where holding every
rim would need 62.

<callout>
<b>All three are now measured, and they agree on the shape of the answer.</b>
(1) never lift: sound, costs the whole vertex set. (2) close the rims: sound,
clears, costs 6–9× the team. (3) hold the region frontier: clears at 5–13
robots, and is what this project already does. The 2010 formulation can be
made correct on this geometry in two different ways, and both of them cost
roughly an order of magnitude — which is the argument for (3) stated as a
measurement rather than as a preference.
</callout>

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
