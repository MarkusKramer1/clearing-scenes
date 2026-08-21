# clearing-scenes

Six real sites, each as one readable scenario file, and a working
implementation of the 2.5D team-visibility baseline of **Kolling et al. (2010)**
on top of them.

A scenario is a **visibility graph over a walkable surface**: vertices are
places a ground robot can stand, edges are the guard regions of Kolling et al.,
and `routes` gives the robot's travel time between every pair of vertices. One
YAML per scene, one npz of arrays beside it, and nothing else to install.

```bash
pip install -r requirements.txt
python scripts/run_clearing.py          # clear every scene, verified cell by cell
xdg-open web/index.html                 # look at the scenes
```

There is no drone. E1's upstream graph is bipartite over a ground and an air
family; this repository keeps the ground half, which is exactly the homogeneous
setting the 2010 paper works in. See [docs/kolling.md](docs/kolling.md) for what
adding the air family back would take.

---

## The scenes

Oxford Spires Dataset (Tao et al., IJRR 2025) — terrestrial laser scan, six
Oxford sites. Each GIF turns once around the site and then walks through it:
grey is the survey cloud, teal is the walkable surface, amber is the graph.

### Christ Church

![Christ Church](docs/gifs/christ-church.gif)

9 959 m² of quad, hall and cloister. 90 vertices, 448 edges — 111 m² per
vertex. The vertices are not spread evenly: an open quad takes a handful, and
the rest go to the cloisters and passages, where a 30 m sensor is stopped by a
wall a few metres away and one vertex genuinely covers tens of square metres
rather than hundreds.

### Bodleian Library

![Bodleian Library](docs/gifs/bodleian-library.gif)

The largest instance here: 12 959 m² and 134 vertices, and the one the baseline
finds hardest — 21 robots at the peak against 9 for Blenheim's near-identical
floor area. Enclosure, not size, is what costs robots.

### Blenheim Palace

![Blenheim Palace](docs/gifs/blenheim-palace.gif)

11 709 m² of mostly open forecourt. Almost as much floor as the Bodleian and a
third of the vertices — 260 m² each, the sparsest graph in the set.

### Keble College

![Keble College](docs/gifs/keble-college.gif)

6 365 m² across two quads, 25 vertices, 255 m² each — open ground again, and
the second sparsest graph here.

### Observatory Quarter

![Observatory Quarter](docs/gifs/observatory-quarter.gif)

3 505 m² between buildings; 47 vertices for half of Keble's area and 75 m²
each, which is what a site cut up by structure looks like in the graph.

### HB Allen Centre

![HB Allen Centre](docs/gifs/hb-allen-centre.gif)

1 292 m², 21 vertices. The smallest scene, and the one to try things on first.

---

## What the baseline does with them

`scripts/run_clearing.py`, verified cell by cell by `clearing/verify.py`:

| scene | area m² | vertices | edges | GRAPH-CLEAR label | peak team (tree order) | peak team (sweep) | steps | makespan h | residual m² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Blenheim Palace | 11 709 | 45 | 221 (100 shady) | 24 | **9** | 12 | 31 | 0.5 | 4.2 |
| Bodleian Library | 12 959 | 134 | 922 (324 shady) | 30 | **21** | 22 | 74 | 2.1 | 13.1 |
| Christ Church | 9 959 | 90 | 448 (123 shady) | 26 | **16** | 20 | 63 | 0.9 | 22.1 |
| Observatory Quarter | 3 505 | 47 | 209 (75 shady) | 21 | **13** | 15 | 31 | 0.4 | 10.0 |
| HB Allen Centre | 1 292 | 21 | 94 (45 shady) | 15 | **10** | 10 | 15 | 0.1 | 4.2 |
| Keble College | 6 365 | 25 | 96 (40 shady) | 14 | **9** | 9 | 18 | 0.2 | 5.4 |

Every scene is cleared, and every one keeps a residual. Both statements need the
qualifier: **cleared except what no vertex can see**. The sampler upstream
stopped when marginal gain fell below the smallest admissible walkable
component, so 62–499 m² per scene is covered by no detection set at all, and no
schedule over this vertex set can clear it — `verify.all_at_once`, every vertex
occupied simultaneously, is the proof. Small fragments of that set are removed
from the evader space; what is left is the residual column above.

The gap between the label and the peak team is not a defect in either. The label
charges a robot per blocked edge and cannot see that one robot standing at `p_j`
holds every guard region inside `D(p_j)` at once. The schedule covers the
frontier with a set cover over vertices instead. [docs/kolling.md](docs/kolling.md)
works through both.

The full schedules — which vertices are held at every step — are in
`examples/<scene>.clearing.yaml`.

---

## The viewer

`web/index.html` opens straight off the file system: no server, no build step.
Switch scenes, toggle the survey cloud, the walkable surface and the graph,
click a vertex to see what it covers, and step through the clearing schedule.

![the viewer](docs/viewer.png)

The states it paints — cleared, contaminated, watched right now — are the ones
`clearing/verify.py` computed, shipped alongside the schedule, so the picture
cannot disagree with the numbers beside it.

`#scene=christ-church&step=28` in the URL is a linkable view.

---

## Layout

```
scenes/
  <scene>.yaml            the scenario: graph, guard regions, routes and times
  geometry/<scene>.npz    the arrays it points at -- cells, detection sets, cloud
  index.json              what was exported, and with which parameters

clearing/
  scene.py                loading, and the one lattice rule everything shares
  kolling.py              spanning tree, GRAPH-CLEAR label, the schedule
  verify.py               contamination propagation; does a schedule clear?
  render.py               a numpy software renderer, for the GIFs

scripts/
  export_from_e1.py       e1-clearing-graph -> scenes/      (needs the survey data)
  run_clearing.py         scenes/ -> examples/
  build_web.py            scenes/ + examples/ -> web/data/
  make_gifs.py            scenes/ -> docs/gifs/

examples/<scene>.clearing.yaml   the schedules, step by step, with their metrics
web/                             the viewer
docs/                            the format, the baseline, the GIFs
tests/                           run with `pytest tests`
```

Everything except `export_from_e1.py` reads only what is in this repository.
That one script needs `e1-clearing-graph` and its survey data, and it is where
the scenes came from.

## Writing your own planner

```python
from clearing import kolling, load, propagate, Schedule

scene = load("christ-church")

# scene.detection[v]  -> cells a robot at v certifies empty
# scene.edge_ij       -> the guard-region graph
# scene.travel_seconds[i, j] -> how long the robot takes to walk from i to j

schedule = Schedule([(0,), (0, 5), (5, 12), ...])   # vertices held per step
result = propagate(scene, schedule)
print(result.metrics(scene))
```

`propagate` knows cells, adjacency and detection sets — not guard regions, not
the regular/shady classification. That is deliberate: if a graph-level strategy
claims to clear and the verifier reports a residual, the graph abstraction is
wrong for that geometry, and the verifier has to be able to say so without
assuming the thing under test.

## Provenance

The geometry comes from `e1-clearing-graph`, the pipeline of *Heterogeneous
Guaranteed Clearing under Overhead Occlusion*: occupancy carved from the
individual E57 setups, per-voxel walkability so that structure above walkable
ground is kept as an occluder rather than absorbed into the surface, detection
sets by raycasting, and two-family sampling. The configuration exported here is
O-carved, `R = 30 m`, seed 1. `scenes/index.json` records it.

The Oxford Spires data is not redistributed here; the survey clouds are tens of
gigabytes and separately licensed. What is in `scenes/geometry/` is derived
geometry: the walkable surface, the detection sets, and a decimated cloud for
context.
