# clearing-scenes

Six real sites, each as one readable scenario file, and a working
implementation of the 2.5D team-visibility baseline of **Kolling et al. (2010)**
on top of them.

A scenario is a **visibility graph over a walkable surface**: vertices are
places a searcher can stand, edges are the guard regions of Kolling et al., and
`routes` gives the platform's travel time between every pair of vertices. One
YAML per scene, one npz of arrays beside it, and nothing else to install.

```bash
pip install -r requirements.txt
python scripts/run_gsst.py --drive      # the published method, driven and judged
xdg-open web/index.html                 # look at the scenes
```

There is no drone. E1's upstream graph is bipartite over a ground and an air
family; this repository keeps the ground half, which is exactly the homogeneous
setting the 2010 paper works in. Adding the air family back means keeping the
other vertices in the export and giving `Strategy` a platform label per
occupied vertex.

Every name in here is either the paper's or deliberately not the paper's, and
[docs/glossary.md](docs/glossary.md) says which is which. Short version: a
**searcher** is the paper's token, a **machine** is a physical thing with an id
and a route, and nothing is called a "robot".

**One planner, and it is the paper's.** This repository used to ship a second
one -- a GRAPH-CLEAR label over a maximum spanning forest, planning against a
greedy cover of the cell frontier -- as a baseline to measure GSST against. It
was never the 2010 paper's planner and said so in its own docstring, and it is
gone; [docs/gsst.md](docs/gsst.md) records what it measured and where the
numbers are recoverable from.

**Every scene stops somewhere.** A survey is not a scenario: the Oxford Spires
clouds run off down service avenues and out through gateways, and a search
problem posed on all of it is posed on ground nobody was asked to clear — and,
worse, on ground an evader can walk in from. So each scene is cut to the
**scenario boundary** drawn by hand in `e1-clearing-graph` and approved there:
a few lines across the openings, and everything the lines do not separate from
the middle of the site. The walkable surface, the vertices, the guard regions
and the routes here are all what survives that cut, and the viewer draws the
line so it reads as a boundary rather than as the edge of the data.

The cut is not free, and its cost is the interesting part. It takes 12 to
1 826 m² off each site, and it takes vertices out of all proportion: Christ
Church gives up 18% of its surveyed surface, and 28 of the 90 vertices E1's
raw sample put on it, behind
**9.5 m** of drawn line. The ground a boundary gives up is thin, and thin
ground is where the sampler put its vertices.

---

## The scenes

Oxford Spires Dataset (Tao et al., IJRR 2025) — terrestrial laser scan, six
Oxford sites. Each GIF turns once around the site: grey is the survey cloud,
teal is the walkable surface, amber is the guard-region graph, and blue is the
scenario boundary the surface was cut to.

Areas and vertex counts below are what the scenario boundary retains, on the
`repaired-ground` vertex set; the figure in brackets is the surveyed ground the
boundary gives up.

| | | |
|:--:|:--:|:--:|
| ![Christ Church](docs/gifs/christ-church.gif) | ![Bodleian Library](docs/gifs/bodleian-library.gif) | ![Blenheim Palace](docs/gifs/blenheim-palace.gif) |
| **Christ Church** — 8 133 m² (−1 826), 81 vertices, 100 m² each. An open quad takes a handful of vertices; the rest go to the cloisters and passages, where a 30 m sensor is stopped by a wall a few metres away. Two cuts totalling 9.5 m take out the meadow approach and a quarter of the vertex set. | **Bodleian Library** — 11 469 m² (−1 490), 151 vertices. The largest instance and the one the published method finds hardest: 29 machines against 10 for Blenheim's near-identical floor area. Enclosure, not size, is what costs searchers. | **Blenheim Palace** — 11 337 m² (−372) of mostly open forecourt, 47 vertices, 241 m² each. The sparsest graph in the set. |
| ![Keble College](docs/gifs/keble-college.gif) | ![Observatory Quarter](docs/gifs/observatory-quarter.gif) | ![HB Allen Centre](docs/gifs/hb-allen-centre.gif) |
| **Keble College** — 6 171 m² (−194) across two quads, 34 vertices, 182 m² each. Open ground again, and the one scene where the boundary halved the team. | **Observatory Quarter** — 2 838 m² (−667) between buildings, 60 vertices, 47 m² each. What a site cut up by structure looks like in the graph, and the heaviest cut in the set at 19% of its survey. | **HB Allen Centre** — 1 280 m² (−12), 32 vertices. The smallest scene, barely bounded at all, and the one to try things on first. |

---

## What the published method does with them

`scripts/run_gsst.py --drive`, on the `repaired-ground` vertex set, 100 random
spanning forests per variant, seed 1. Two verdicts per strategy, and the gap
between them is the result:

| scene | area m² | vertices | edges | best variant | team | atomic residual m² | driven residual m² |
|---|---:|---:|---:|---|---:|---:|---:|
| Blenheim Palace | 11 337 | 47 | 258 (88 shady) | regular | **10** | 0.0 | **0.0** |
| Bodleian Library | 11 469 | 151 | 1 408 (365 shady) | regular_biased | **29** | 10 920 | **0.0** |
| Christ Church | 8 133 | 81 | 367 (134 shady) | regular | **13** | 7 749 | **0.0** |
| Observatory Quarter | 2 838 | 60 | 481 (142 shady) | regular_biased | **18** | 2 762 | **0.0** |
| HB Allen Centre | 1 280 | 32 | 229 (64 shady) | regular_biased | **13** | 1 185 | **0.0** |
| Keble College | 6 171 | 34 | 183 (94 shady) | regular | **8** | 0.0 | **0.0** |

"Driven residual" is `residual_avoidable_m2` — what is left that some vertex of
this graph could have seen. All eighteen rows, every scene in every variant,
reach 0.00 there and pass the `A_min` tolerance. Bodleian, Christ Church and
Observatory keep 10.5, 1.1 and 5.5 m² that **no ground vertex can see at any
team size**; `verify.all_at_once` is the proof, and the figure is reported
separately rather than folded in.

**Two things had to change before that was true, and neither is a better
planner.**

**The vertex set.** These scenes used to ship E1's raw sample — 20 / 24 / 36 /
39 / 62 / 112 ground vertices — which is what a sampler optimising *coverage*
produces, and coverage is not the property a clearing argument needs. It left
fragments that sever the walkable surface watched by nobody, and 16–116 m² per
scene of ground that no *ground* vertex can see. Both are repaired upstream
now, at 32 / 34 / 47 / 60 / 81 / 151 vertices, and
[docs/scene-format.md](docs/scene-format.md) says exactly what was added and
why. It is a change to the problem, and `source.vertex_set` records it in every
scene file.

**The model of a step.** Atomically a step is instantaneous and nobody is ever
in transit, which is the model the paper works in and the one the "atomic
residual" column above is computed in. Real machines drive, and while one
drives it is the standing team that holds the frontier. Splitting each step
into single moves and charging for the interval is the difference between the
two columns — 7 749 m² and zero at Christ Church. `clearing/execute.py` does
the splitting and `clearing/clock.py` is the verdict.

**None of it lifts the paper's own prohibition.** Kolling et al. forbid a
driving searcher to sweep a detection-set boundary; a machine in transit here is
credited with nothing, in the admission test and in the referee alike. That is
also what keeps this repository free of the survey data: crediting the driver
would need the raycaster, crediting it with nothing needs only the detection
sets and the travel times `scenes/` already ships.

**The fleet is the whole team.** `clearing/roster.py` puts identities on a
strategy and the fleet that comes out equals `n_searchers`, except where the
executor had to *spend* a machine: when nobody on site can make a drive
without opening the frontier, it places a new one on the destination and leaves
the one that would have leaked where it stands. That is the paper's own
fallback, `machines_spent` counts it, and it is where Bodleian's 25-searcher
atomic peak becomes a fleet of 29.

---

## The published method, run as published

`clearing/gsst.py` is Kolling et al. (2010)'s own machinery — vertex
contamination after Hollinger et al., the Barrière et al. tree label with
**sliding forbidden**, and GSST's anytime layer over 100 random depth-first
spanning forests. `scripts/run_gsst.py` runs it.

**The paper's first finding reproduces.** Treating only *regular* edges as cycle
edges — dropping the shady ones, whose guard regions are strictly contained in
another vertex's — is reported as worth 2–3 searchers in the minimum. On geometry
the authors never saw, on the repaired vertex set:

| scene | n | naive | regular | regular + biased tree | saved |
|---|--:|--:|--:|--:|--:|
| Blenheim Palace | 47 | 13 | **10** | 12 | +3 |
| Bodleian Library | 151 | 30 | **25** | **25** | +5 |
| Christ Church | 81 | 14 | **11** | 12 | +3 |
| HB Allen Centre | 32 | 16 | 14 | **13** | +3 |
| Keble College | 34 | 12 | **8** | **8** | +4 |
| Observatory Quarter | 60 | 21 | **17** | 18 | +4 |
| | | | | **mean saved** | **+3.7** |

These are the peaks of the *graph* strategy — the paper's own output, a claim
about the guard graph under vertex contamination. The paper's *third* variant
still does not pay: biasing the tree toward regular edges wins once, loses three
times and ties twice.

**The graph clears, and atomically the ground does not.** Every strategy above
provably clears the guard graph, which is where the paper stops. Handed to
`clearing/verify.py`, which was told nothing about guard regions, four of six
scenes are left with thousands of square metres — Christ Church keeps 7 749 m²
of its 8 133. That is not a data artefact: every vertex occupied at once leaves
1–11 m².

The reason is one measurement, `scripts/boundary_coverage.py`, and it is about the
geometry rather than about any planner. An edge says a searcher at `j` can watch
*some* of `dD(p_i)`. The median neighbour watches **60%** of it — and
occupying **every** neighbour of a vertex at once still leaves a gap on **404
of the 405 vertices** of all six sites. The guard graph records who can watch a
piece of a boundary, never whether the boundary is closed, and an arbitrarily
fast evader needs one hole.

**What closes it is the clock, not more searchers.** The same strategies, driven
one machine at a time with nobody credited in transit, clear all six — because a step
that looked instantaneous was hiding the only moment at which the boundary gap could
be exploited, and taking the moment seriously makes the standing team hold it.
[docs/gsst.md](docs/gsst.md) has all three tables, the no-sliding label rule and
what was left unrun.

---

## The viewer

`web/index.html` opens straight off the file system: no server, no build step.
Switch scenes, toggle the survey cloud, the walkable surface, the graph and the
scenario boundary, click a vertex to see what it covers, pick a route to see the
ground the platform covers to walk it, and step through the clearing strategy.

![the viewer](docs/viewer.png)

The states it paints — cleared, contaminated, watched right now — are the ones
`clearing/clock.py` computed, shipped alongside the strategy, so the picture
cannot disagree with the numbers beside it.

**The boundary.** Drawn as a low blue fence whose foot is exactly the line. A
line lying on the floor is not visible from the overhead view a site is read
from — the walkable cells are drawn as sprites nearly two cells across and
cover it — so it is extruded to a height measured per scene: just under the
lowest ground the outline runs along, up to the higher of five metres and a metre
above the highest, which is what it takes to clear the Christ Church terrace
without burying the site under a wall. It is blue and not the amber
`e1-clearing-graph` uses, because amber here is the guard-region edges and an
amber ring around the site read as more graph. The survey cloud is *not* cut to
it: the line runs along and through those buildings, and clipping the cloud
would hide the structure the line was drawn against.

**Robot routes.** An edge of the graph is a line of sight; the ground between
its two ends may be a building. The *Robot route* picker draws what the platform
actually walks — the true shortest path over the walkable surface, the one the
travel time in `scenes/*.yaml` is made of. Pick a vertex to see every route out
of it, or a second vertex for one route with its distance, its time and how far
it is from the straight line. Median detour across the six scenes is 1.08×, but
the tail is what matters: on Blenheim Palace, edge 3–21 is 22.0 m of sight line
and 71 m of walking, round the end of a wall.

![a platform route](docs/route.png)

**The machines, with their ids.** A strategy step is a *set* of vertices: it says
which places are held and nothing about which machine holds them, so on its own
it can only be drawn as dots blinking on and off. *Show the machines* draws the
fleet instead — R0…R7 at Blenheim, R0…R21 at the Bodleian, each with its own
colour and its id over its head. Every step lists what each one is doing: the
metres and minutes it just walked, `holds` where it stayed, `parked` where the
step had no work for it, `off site` before it arrives. Click a machine, in the
view or in the list, and the surface it watches from where it stands is painted
in its colour along with everything it has walked so far. *Play* walks the team
between two steps along the paths it really takes, not along the sight lines.

![the machines](docs/robots.png)

A searcher has no position between two steps. A machine in transit is credited
with nothing, and a parked one standing at a vertex does see `D(v)` but is not
credited with it —
the guarantee quantifies over the vertices a step names, and the walking is the
strategy's secondary cost. The viewer says so where it could otherwise mislead.

The URL is a linkable view: `#scene=christ-church&step=28`,
`#scene=christ-church&vertex=12`, `#scene=blenheim-palace&from=3&to=21`,
`#scene=christ-church&step=29&machine=1`.

---

## Layout

```
scenes/
  <scene>.yaml            the scenario: graph, guard regions, routes and times
  geometry/<scene>.npz    the arrays it points at -- cells, detection sets, cloud
  index.json              what was exported, and with which parameters

clearing/
  scene.py                loading, and the one lattice rule everything shares
  gsst.py                 Kolling et al. 2010 as published: no-slide label, GSST
  verify.py               contamination propagation, atomically: does it clear?
  execute.py              the same strategy driven one machine at a time
  clock.py                and judged on a wall clock, charging for the interval
  roster.py               identities on a strategy: which machine, walking where
  render.py               a numpy software renderer, for the GIFs

scripts/
  export_from_e1.py       e1-clearing-graph -> scenes/      (needs the survey data)
  run_gsst.py             the published method, three cycle-edge variants
  boundary_coverage.py    how much of dD(p) a neighbour can actually watch
  build_web.py            scenes/ + examples/ -> web/data/
  make_gifs.py            scenes/ -> docs/gifs/

examples/<scene>.gsst.yaml       the strategies, step by step, with their
                                 metrics -- atomic and driven
web/                             the viewer
docs/glossary.md                 which names are the paper's, and which
                                 are deliberately not
docs/                            the format, the method, the GIFs
tests/                           run with `pytest tests`
```

Everything except `export_from_e1.py` reads only what is in this repository.
That one script needs `e1-clearing-graph` and its survey data, and it is where
the scenes came from.

## Writing your own planner

```python
from clearing import clock, execute, load, propagate, Strategy

scene = load("christ-church")

# scene.detection[v]  -> D(p): cells a searcher at v certifies empty
# scene.edge_ij       -> the guard-region graph
# scene.travel_seconds[i, j] -> how long the platform takes to walk from i to j

strategy = Strategy([(0,), (0, 5), (5, 12), ...])   # vertices held per step
print(propagate(scene, strategy).metrics(scene))    # atomic: steps are instants

driven = execute.sequentialise(scene, strategy)     # one machine at a time
print(clock.timeline_propagate(scene, driven.strategy, driven.roster,
                               exact=True))         # and charged per interval
```

`propagate` knows cells, adjacency and detection sets — not guard regions, not
the regular/shady classification. That is deliberate: if a graph-level strategy
claims to clear and the verifier reports a residual, the graph abstraction is
wrong for that geometry, and the verifier has to be able to say so without
assuming the thing under test.

**Run both.** On this geometry they do not agree, and the disagreement is the
interesting part: the atomic verdict is the model Kolling et al. work in, and
the driven one is what happens when the machines have to get there. A planner
that clears atomically may leave the site open for the length of a drive, and
one that looks hopeless atomically may hold it — the published method does the
second on four of the six scenes.

## Provenance

The geometry comes from `e1-clearing-graph`, the pipeline of *Heterogeneous
Guaranteed Clearing under Overhead Occlusion*: occupancy carved from the
individual E57 setups, per-voxel walkability so that structure above walkable
ground is kept as an occluder rather than absorbed into the surface, detection
sets by raycasting, and two-family sampling. The configuration exported here is
O-carved, `R = 30 m`, seed 1, inside the approved scenario boundary, on the
`repaired-ground` vertex set. `scenes/index.json` records all of it, and each
`scenes/<scene>.yaml` carries the boundary's stamp — a hash over the lines that
were drawn — so a scenario here can be matched to the decision upstream that
produced it.

**The detection predicate is upstream's and is not the paper's.** A cell counts
as detected when *all five* rays to the target cylinder arrive. Kolling et al.
require only that the target be seen, under which `D(p)` grows with the target
height; the conjunction makes it shrink. It is E1's frozen decision, recorded
there, and it is why vertex counts here are not directly comparable with the
paper's.

**The guard graph is E1's own, rebuilt and checked.** The repaired vertex set
has no committed edge list, so `scripts/export_from_e1.py` rebuilds it with
`e1.graph.build_graph` — E1's relation, not a second implementation of it — and
refuses to write anything unless the result agrees edge for edge and kind for
kind with `e2.gsst.ground_guide`. The travel matrix is rebuilt the same way and
for the same reason: the roadmap E1 wrote has no rows for the vertices the
repair added.

The cut is E1's, exactly, including where E1's own index arithmetic puts a cell
one row off at an exact face of its 0.4 m boundary raster. That moves 383 cells
across the six scenes, 0.16 to 6.72 m² per scene against 1 280 to 11 469 m²
retained. They are kept as E1 keeps them rather than corrected here, because
the claim this repository makes is that its scenario *is* the approved one, and
a scenario that is nearly the approved one is worth less than one that is
exactly it. `tests/test_boundary.py` measures the seam and bounds it to the
line; the fix belongs upstream, in `boundary_apply.inside`, where it would move
all six stamps.

The Oxford Spires data is not redistributed here; the survey clouds are tens of
gigabytes and separately licensed. What is in `scenes/geometry/` is derived
geometry: the walkable surface, the detection sets, and a decimated cloud for
context.
