"""Fixtures shared by the test modules: the toy corridor and the small scene.

They lived in `tests/test_kolling.py` until the GRAPH-CLEAR baseline that
module tested was removed; the scenes it built are not about that baseline and
are used by `test_gsst`, `test_roster` and `test_execute` alike.
"""

from __future__ import annotations

import numpy as np

from clearing import scene as scene_mod

#: the smallest shipped scene, used wherever a test needs real geometry
SMALL = "hb-allen-centre"


def corridor(n_cells: int = 60, span: int = 10) -> scene_mod.Scene:
    """A 1-cell-wide corridor. Every vertex sees its own block and nothing else.

    Small enough to reason about by hand: k blocks in a row means a strategy
    has to sweep from one end, and one machine behind the front is enough.
    """
    ij = np.stack([np.arange(n_cells), np.zeros(n_cells)], axis=1).astype(np.int32)
    xyz = np.column_stack([np.arange(n_cells) * 0.2, np.zeros(n_cells),
                           np.zeros(n_cells)]).astype(np.float32)
    n_vertices = n_cells // span
    detection = [np.arange(k * span, (k + 1) * span, dtype=np.int32)
                 for k in range(n_vertices)]
    vertex_cell = np.array([k * span + span // 2 for k in range(n_vertices)], np.int32)
    edges = np.array([[k, k + 1] for k in range(n_vertices - 1)], np.int32)
    T = np.abs(vertex_cell[:, None] - vertex_cell[None, :]).astype(np.float32) * 0.2
    return scene_mod.Scene(
        name="corridor", title="corridor",
        doc={"surface": {"max_step_m": 0.25}, "machine": {"speed_m_s": 1.0}},
        cell_xyz=xyz, cell_ij=ij, cell_size=0.2,
        detection=detection, detection_boundary=[d[[0, -1]] for d in detection],
        vertex_xyz=xyz[vertex_cell], vertex_cell=vertex_cell,
        edge_ij=edges, edge_shady=np.zeros(len(edges), bool),
        guard_region=[np.array([span * (k + 1) - 1], np.int32) for k in range(len(edges))],
        travel_seconds=T, travel_metres=T,
        uncoverable=np.zeros(0, np.int32), speck_max_area_m2=0.0)
