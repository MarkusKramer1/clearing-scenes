"""Loading a scenario: the YAML, the geometry beside it, and one lattice rule.

A `Scene` is everything a clearing algorithm needs and nothing else:

    cells        the walkable surface, one point per cell
    detection    D(v) per vertex, as cell indices -- what a robot there sees
    edges        the guard-region graph, regular or shady
    travel       vertex-to-vertex time and distance for the ground robot

Two things are computed here rather than shipped, because both are short and
shipping them would let a file drift away from the rule that defines it:
the evader adjacency (`Scene.evader_edges`) and the all-pairs travel times
read back out of the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes"


# ---------------------------------------------------------------------------
# the lattice
# ---------------------------------------------------------------------------


def lattice_edges(cell_ij: np.ndarray, height: np.ndarray, cell_size: float,
                  max_step: float):
    """8-connected edges over the walkable cells, as (a, b, length).

    Two cells are neighbours when their columns touch in the 8-neighbourhood
    AND their heights differ by no more than `max_step`. The length is the 3D
    step, so a ramp costs its slope rather than its plan projection.

    THE HEIGHT TEST IS NOT DECORATION. The surface is per-voxel, so one column
    can carry two cells -- the ground under an arcade and the terrace over it.
    Plain (i, j) adjacency would make those neighbours and let an evader step
    four metres vertically through a stone vault. For the same reason a column
    is joined over its FULL range of cells and not just its first one, or the
    upper layer of an overhang ends up with no neighbours at all.
    """
    n = cell_ij.shape[0]
    if n == 0:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float64),)

    ny = int(cell_ij[:, 1].max()) + 2
    key = cell_ij[:, 0].astype(np.int64) * ny + cell_ij[:, 1].astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]

    A, B, W = [], [], []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        flat = float(np.hypot(di, dj) * cell_size)
        target = key + di * ny + dj
        lo = np.searchsorted(ks, target, side="left")
        hi = np.searchsorted(ks, target, side="right")
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            continue
        a = np.repeat(np.arange(n, dtype=np.int64), cnt)
        step = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        b = order[np.repeat(lo, cnt) + step]
        dz = np.abs(height[a] - height[b])
        keep = dz <= max_step
        a, b, dz = a[keep], b[keep], dz[keep]
        A.append(a)
        B.append(b)
        W.append(np.sqrt(flat * flat + dz * dz))
    if not A:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float64),)
    return np.concatenate(A), np.concatenate(B), np.concatenate(W)


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    name: str
    title: str
    doc: dict                       # the YAML, verbatim

    cell_xyz: np.ndarray            # (n_cells, 3) surface points [m]
    cell_ij: np.ndarray             # (n_cells, 2) grid indices
    cell_size: float

    detection: list[np.ndarray]     # per vertex: cell indices it sees
    rim: list[np.ndarray]           # per vertex: cells on the boundary of D(v)
    node_xyz: np.ndarray            # (n_nodes, 3) sensor positions
    node_cell: np.ndarray           # (n_nodes,) the cell each vertex stands on

    edge_ij: np.ndarray             # (n_edges, 2) guard-region edges
    edge_shady: np.ndarray          # (n_edges,) bool
    guard: list[np.ndarray]         # per edge: the guard-region cells

    travel_seconds: np.ndarray      # (n_nodes, n_nodes), inf where unreachable
    travel_metres: np.ndarray
    uncoverable: np.ndarray         # cells no vertex sees, at any team size

    #: per guard-graph edge, the path the robot walks between its two
    #: vertices: the true shortest route over the walkable surface, simplified
    #: for drawing. Empty where the two vertices cannot reach each other.
    routes: list[np.ndarray] = field(repr=False, default_factory=list)

    cloud_xyz: np.ndarray = field(repr=False, default=None)

    #: connected uncoverable fragments below this area are taken out of the
    #: evader space entirely. 0 keeps every cell. See `excluded`.
    speck_max_area_m2: float = 4.0

    # -- sizes --------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return int(self.cell_xyz.shape[0])

    @property
    def n_nodes(self) -> int:
        return len(self.detection)

    @property
    def cell_area(self) -> float:
        return self.cell_size ** 2

    def area(self, mask_or_idx) -> float:
        a = np.asarray(mask_or_idx)
        n = int(a.sum()) if a.dtype == bool else int(a.size)
        return n * self.cell_area

    # -- derived ------------------------------------------------------------

    @cached_property
    def max_step(self) -> float:
        return float(self.doc["surface"]["max_step_m"])

    @cached_property
    def evader_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """How the evader moves: the walkable lattice, as an edge list."""
        a, b, _ = lattice_edges(self.cell_ij, self.cell_xyz[:, 2],
                                self.cell_size, self.max_step)
        ex = self.excluded
        keep = ~ex[a] & ~ex[b]
        return a[keep].astype(np.int32), b[keep].astype(np.int32)

    @cached_property
    def excluded(self) -> np.ndarray:
        """Uncoverable specks removed from the evader space, as a cell mask.

        Connected fragments of the uncoverable set smaller than
        `speck_max_area_m2`. These are wall-fringe slivers, every one within a
        voxel of masonry, and they are an artefact of two things: where the
        sampler stopped adding vertices, and a detection predicate that
        deliberately fails at a wall corner when one of its five rays clips the
        stone. E1 probed 300 of them against the nearest forty admissible
        sensor positions: 78% are fully visible from a position the sampler
        simply never placed, 14% reach four rays of five, and none reach zero.
        They are not places nothing can see.

        WHY THE EVADER SPACE AND NOT JUST THE FAILURE COUNT. A speck left in
        the lattice is unobserved and adjacent to contamination, so the flood
        puts it straight back and every step has to guard its rim. On these
        scenes that is the difference between a team of twenty and a team of
        ninety -- the schedule spends its robots watching the fringe of a
        crack, and the fringe is the artefact, not the crack.

        THE EXACT REPAIR is to sample the vertices that were never placed,
        which removes each speck and its rim at once and needs no tolerance.
        That is a change to the pipeline upstream of this repository; this is
        what can be done without one, and `scenes/*.yaml` records the amount.
        """
        out = np.zeros(self.n_cells, dtype=bool)
        if self.speck_max_area_m2 <= 0 or self.uncoverable.size == 0:
            return out
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        a, b, _ = lattice_edges(self.cell_ij, self.cell_xyz[:, 2],
                                self.cell_size, self.max_step)
        g = sparse.coo_matrix((np.ones(a.size, np.uint8), (a, b)),
                              shape=(self.n_cells, self.n_cells)).tocsr()
        g = g + g.T
        rows = self.uncoverable
        _, lab = connected_components(g[rows][:, rows], directed=False)
        areas = np.bincount(lab) * self.cell_area
        out[rows[(areas < self.speck_max_area_m2)[lab]]] = True
        return out

    @cached_property
    def coverable(self) -> np.ndarray:
        """Cells some vertex sees. Its complement is what no team can clear."""
        out = np.zeros(self.n_cells, dtype=bool)
        for d in self.detection:
            out[d] = True
        return out

    def observed(self, vertices) -> np.ndarray:
        """Union of the detection sets of the given vertices, as a cell mask."""
        out = np.zeros(self.n_cells, dtype=bool)
        for v in vertices:
            out[self.detection[int(v)]] = True
        return out

    def route(self, i: int, j: int) -> np.ndarray:
        """The walked path between two vertices, or an empty array.

        Only guard-graph edges carry one -- those are the pairs the viewer and
        the schedules ask about. `travel_metres` and `travel_seconds` cover
        every pair; the polylines cover the edges.
        """
        i, j = int(i), int(j)
        hit = np.flatnonzero(((self.edge_ij[:, 0] == i) & (self.edge_ij[:, 1] == j))
                             | ((self.edge_ij[:, 0] == j) & (self.edge_ij[:, 1] == i)))
        if hit.size == 0:
            return np.zeros((0, 3), np.float32)
        return self.routes[int(hit[0])]

    def neighbours(self, v: int) -> np.ndarray:
        """Vertices sharing a guard-region edge with `v`."""
        m = self.edge_ij[:, 0] == v
        n = self.edge_ij[:, 1] == v
        return np.unique(np.concatenate([self.edge_ij[m, 1], self.edge_ij[n, 0]]))


def load(name: str, scenes_dir: Path | str = SCENES_DIR,
         with_cloud: bool = False, speck_max_area_m2: float = 4.0) -> Scene:
    """Read `scenes/<name>.yaml` and the geometry it points at.

    `speck_max_area_m2` is the evader-space tolerance documented on
    `Scene.excluded`; pass 0 to keep every cell.
    """
    scenes_dir = Path(scenes_dir)
    doc = yaml.safe_load((scenes_dir / f"{name}.yaml").read_text())
    z = np.load(scenes_dir / doc["surface"]["geometry_file"])

    def csr(prefix):
        off, val = z[f"{prefix}_offsets"], z[f"{prefix}_cells"]
        return [val[off[i]:off[i + 1]] for i in range(off.size - 1)]

    roff, rpts = z["edge_route_offsets"], z["edge_route_points"]
    routes = [rpts[roff[k]:roff[k + 1]] for k in range(roff.size - 1)]

    return Scene(
        name=doc["scene"], title=doc["title"], doc=doc,
        cell_xyz=z["cell_xyz"], cell_ij=z["cell_ij"],
        cell_size=float(z["cell_size"]),
        detection=csr("detection"), rim=csr("rim"),
        node_xyz=z["node_xyz"], node_cell=z["node_cell"],
        edge_ij=z["edge_ij"], edge_shady=z["edge_shady"].astype(bool),
        guard=csr("guard"),
        travel_seconds=z["travel_seconds"], travel_metres=z["travel_metres"],
        uncoverable=z["uncoverable"], routes=routes,
        cloud_xyz=z["cloud_xyz"] if with_cloud else None,
        speck_max_area_m2=float(speck_max_area_m2),
    )


def available(scenes_dir: Path | str = SCENES_DIR) -> list[str]:
    return sorted(p.stem for p in Path(scenes_dir).glob("*.yaml"))
