"""Python entry point of the solver.

    res = scsp.solve(model, time_limit=10.0, threads=4, seed=1, warm_start=x0)
    res.x, res.obj, res.bound, res.gap, res.status, res.trace

Two paths (res.stats["path"]):
- "sc": models whose set rows are all covering rows (>= 1) and whose side constraints are all
  sum a x <= b with a >= 0 (the DMS rider constraints). Lagrangian heuristic on most threads,
  root LP and LP-based branch and bound on one (P1 + P3).
- "bin": every other model (partitioning or packing rows, other side constraints, negative
  costs). LP-based branch and bound with Ryan-Foster branching for pure partitioning, on one
  thread (P3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .model import COVER, PACK, Model


@dataclass
class Result:
    status: str                      # optimal | time_limit | no_solution | infeasible
    x: np.ndarray | None             # 0/1 vector, None if no solution
    cols: np.ndarray                 # indices of the columns at 1
    obj: float | None
    bound: float | None
    gap: float | None                # (obj - bound) / |obj|
    runtime: float
    trace: list = field(default_factory=list)    # [(seconds, obj, bound, "sol" | "bound"), ...]
    stats: dict = field(default_factory=dict)
    verified: bool | None = None     # solution re-checked against the Model


def supported(model: Model) -> str | None:
    """None if the set covering path can take the model, else the reason it goes to the general 0-1 path."""
    if model.n_rows and not np.all(model.row_sense == COVER):
        return "set rows other than >= 1 (partitioning/packing) need the LP-based path (P3)"
    if model.n_side and not np.all(model.side_sense == PACK):
        return "side constraints other than <= are not supported yet"
    if model.n_side and np.any(model.side_val < 0):
        return "side constraints with negative coefficients are not supported yet"
    if np.any(model.cost < 0):
        return "negative costs are not supported yet"
    return None


def _warm_indices(model: Model, warm_start) -> np.ndarray | None:
    if warm_start is None:
        return None
    w = np.asarray(warm_start)
    if w.dtype == bool or (w.size == model.n_cols and model.n_cols > 1 and np.all((w == 0) | (w == 1))):
        return np.flatnonzero(w).astype(np.int64)
    return np.unique(w.astype(np.int64))


def solve(model: Model, time_limit: float = 10.0, threads: int = 4, seed: int = 1, warm_start=None,
          mip_gap: float = 0.0, trace: bool = True, presolve: bool = True, local_search: bool = True,
          enumerate: bool = True, bnb: bool = True, verify: bool = True) -> Result:
    """Solve a set covering / partitioning model. `warm_start` is a 0/1 vector or a list of
    column indices; the returned solution is never worse than a feasible warm start.

    Covering models (>= 1 rows, <= side constraints) go to the set covering solver: the
    Lagrangian heuristic on most threads plus LP-based branch and bound on one. Every other
    model (partitioning or packing rows, other side constraints) goes to the general 0-1
    path, which is LP-based branch and bound on one thread.
    """
    warm = _warm_indices(model, warm_start)
    warm_ok = False
    if warm is not None:
        x0 = np.zeros(model.n_cols)
        x0[warm] = 1.0
        warm_ok = model.is_feasible(x0)
    if supported(model) is None:
        r = _core.solve_sc(
            model.cost, model.row_ptr, model.col_idx, model.side_ptr, model.side_idx, model.side_val, model.side_rhs,
            offset=float(model.obj_offset), time_limit=float(time_limit), threads=int(threads), seed=int(seed),
            mip_gap=float(mip_gap), warm=warm.tolist() if warm_ok else None,
            presolve=presolve, local_search=local_search, enumerate=enumerate, bnb=bnb,
        )
        path = "sc"
    else:
        a, lower, upper = model.to_scipy()
        a = a.tocsr()
        a.sort_indices()
        r = _core.solve_bin(
            model.cost, a.indptr.astype(np.int64), a.indices.astype(np.int32), a.data.astype(np.float64),
            np.ascontiguousarray(lower, dtype=np.float64), np.ascontiguousarray(upper, dtype=np.float64),
            offset=float(model.obj_offset), time_limit=float(time_limit), threads=int(threads), seed=int(seed),
            mip_gap=float(mip_gap), warm=warm.tolist() if warm_ok else None,
        )
        path = "bin"
    cols = np.asarray(r["cols"], dtype=np.int64)
    obj, bound = r["obj"], r["bound"]
    x = None
    verified = None
    if obj is not None:
        x = np.zeros(model.n_cols)
        x[cols] = 1.0
        if verify:
            verified = bool(model.is_feasible(x) and abs(model.objective(x) - obj) <= 1e-6 * max(1.0, abs(obj)))
            if not verified:
                raise RuntimeError(f"{model.name}: solver returned a solution that fails verification")
    gap = None
    if obj is not None and bound is not None:
        gap = 0.0 if obj == bound else (float("inf") if obj == 0 else max(0.0, (obj - bound) / abs(obj)))
    stats = dict(r["stats"])
    stats["path"] = path
    stats["warm_given"] = warm is not None
    stats["warm_feasible"] = warm_ok
    return Result(status=r["status"], x=x, cols=cols, obj=obj, bound=bound, gap=gap, runtime=r["runtime"],
                  trace=list(r["trace"]) if trace else [], stats=stats, verified=verified)
