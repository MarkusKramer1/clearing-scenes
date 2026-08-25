"""Ground-robot clearing scenarios, and two planners on them.

`kolling` is the cell-frontier planner this repository ships as its baseline.
`gsst` is Kolling et al. (2010) as published -- the no-sliding tree label and
GSST over random spanning forests -- which is a different algorithm and, on
these scenes, a different answer. See `docs/gsst.md`.
"""

from . import gsst, kolling
from .scene import Scene, available, load
from .verify import Schedule, all_at_once, propagate

__all__ = ["Scene", "Schedule", "all_at_once", "available", "gsst", "kolling",
           "load", "propagate"]
