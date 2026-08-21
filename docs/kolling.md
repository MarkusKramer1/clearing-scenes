# What the baseline implements, and where it departs

The reference is Kolling, Kleiner, Lewis & Sycara, *Pursuit-Evasion in 2.5D
Based on Team-Visibility*, IROS 2010, which puts guaranteed clearing on real
terrain, and the GRAPH-CLEAR machinery it clears its graph with: Kolling &
Carpin, *The GRAPH-CLEAR problem*, IROS 2007.

## The problem

An evader is somewhere on the walkable surface. It is arbitrarily fast,
omniscient, and moves continuously. A schedule places robots at sampled sensor
positions; a robot at `v` certifies `D(v)` empty for as long as it stands there.
The scene is **cleared** when no evader can remain, whatever it did.

The state is therefore a *set*, not a position: the cells that could still hold
the evader. It starts as the whole surface, loses what the robots see, and
regains -- to a fixed point -- everything reachable from what is left. There is
no evader agent anywhere in this repository, and there is no random seed. The
guarantee is the emptiness of that set, which quantifies over every evader at
once.

## The pipeline, and which half lives where

| step | where |
|---|---|
| survey cloud → occupancy → walkable surface | `e1-clearing-graph` |
| detection sets `D(p)` by line of sight | `e1-clearing-graph` |
| sampling sensor positions until the surface is covered | `e1-clearing-graph` |
| guard regions, regular/shady edges | `e1-clearing-graph` |
| **spanning tree, label, schedule** | **`clearing/kolling.py`** |
| **verification** | **`clearing/verify.py`** |

The first four are shipped as data in `scenes/`. The last two are the part you
would replace to try a different planner.

## Two numbers, answering different questions

### The GRAPH-CLEAR label

`kolling.label_tree` computes the tree recursion of Kolling & Carpin on a
maximum spanning forest of the guard graph. With `rho(v)` robots to sweep a
vertex and `w(e)` to block an edge, the label of the subtree at `v` entered from
its parent is

```
L(v, p) = max{ rho(v) + sum_c w(v, c),
               max_i [ L(c_i, v) + sum_{j>i} w(v, c_j) ] }
```

Either you are sweeping `v`, which needs a blocker on every edge to a child, or
you are down in child `i`, which needs that child's label plus a blocker on
every sibling still contaminated. There is no term for the parent edge -- you
came from there, so it is already clear. Children are entered cheapest-first,
which the usual exchange argument shows is optimal, and the root is chosen by
trying all of them.

Here `rho = w = 1`: one robot sweeps a vertex, one robot holds a guard region.

Two things make this a bound rather than an answer:

* **Edges outside the spanning tree are charged to both endpoints.** A real
  guard graph is dense -- Christ Church has 448 edges over 90 vertices -- so
  most edges are outside any tree, and this term dominates.
* **It charges every blocked edge separately.** It cannot see that one robot
  standing at `p_j` holds every guard region inside `D(p_j)` at once.

### The schedule

`kolling.plan` produces something executable, and it does not charge per edge.
It grows a cleared region one vertex at a time and, at every step, buys the
cheapest set of vertices whose detection sets cover the **frontier** of that
region -- the cells of it with a walkable neighbour outside. Cover the frontier
and the region survives the step; leave a gap and the whole region reconnects to
what is beyond it.

That single sentence is the algorithm. The cover is greedy (set cover is
NP-hard) and recomputed from scratch each step, so a guard held because the
frontier ran past it four steps ago is not paid for.

`order=` decides which vertex is attached next:

* **`kolling`** follows the spanning-tree recursion, so the schedule visits the
  graph in the order the label was computed in.
* **`sweep`** takes vertices along the principal axis of the surface, which
  keeps the cleared region a single advancing band. Taking whichever vertex adds
  the most new area instead sends the region jumping across the site, leaving
  several disconnected patches open at once with a perimeter each.

On the six shipped scenes the schedule needs between two fifths and seven
tenths of what the label says -- 9 robots against a label of 24 at Blenheim,
21 against 30 at the Bodleian.

## The tolerance, stated plainly

Two things stop these scenes from being cleared outright.

**Cells no vertex sees.** The sampler stopped when marginal gain fell below the
smallest admissible walkable component, so some surface is covered by no `D(v)`
at all. No schedule over this vertex set clears it -- `verify.all_at_once`, every
vertex occupied simultaneously, is the proof. Results therefore report both
`cleared` and `cleared_except_uncoverable`, and never fold the second into the
first.

**Wall-fringe specks.** Small connected fragments of that uncoverable set are
removed from the evader space entirely (`Scene.excluded`, default under 4 m²).
This is a real tolerance and it is the single largest lever on team size on
these scenes -- a speck left in the lattice is unobserved and adjacent to
contamination, so the flood puts it straight back and every step pays robots to
guard the rim of a crack. The upstream measurement behind the choice is quoted
in the docstring. Pass `speck_max_area_m2=0` to turn it off and watch the peak
team roughly quintuple.

## What is not here

No UAV. E1's graph is bipartite over a ground and an air family; this export
keeps the ground half, which is exactly the homogeneous setting the 2010 paper
works in. Adding the air family back means keeping the other vertices in the
export and giving `Schedule` a platform label per occupied vertex.

No monotonicity constraint. The planner may leave a frontier open and give
ground back; recontamination is counted, not forbidden. Forbidding it is a
constraint the guarantee does not require, and one worth measuring rather than
assuming.

No optimality claim. There is no lower bound computed here, and no brute-force
comparison on small instances.
