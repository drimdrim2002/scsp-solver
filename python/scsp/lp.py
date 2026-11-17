"""LP relaxation of a Model, solved by the native LP engine (P2).

    res = scsp.solve_lp(model)                       # 0 <= x <= 1 relaxation
    res.status, res.obj, res.x, res.row_dual, res.red_cost, res.stats

`method`: "auto" (sifting when columns greatly outnumber rows), "dual" or "sifting".
Row duals follow the convention  red_cost = cost - A' row_dual  (a tight >= row has a
dual >= 0). Rows are ordered as in Model.to_scipy(): set rows first, then side rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .model import Model


@dataclass
class LPResult:
    status: str                  # optimal | infeasible | unbounded | time_limit | iteration_limit | failed
    obj: float | None
    x: np.ndarray | None
    row_dual: np.ndarray | None
    red_cost: np.ndarray | None
    runtime: float
    stats: dict = field(default_factory=dict)


def solve_lp_arrays(cost, a_csr, row_lo, row_up, col_lb, col_ub, offset: float = 0.0, method: str = "auto",
                    time_limit: float | None = None, feas_tol: float = 1e-9, opt_tol: float = 1e-9,
                    refactor: int = 100, perturb: bool = True, seed: int = 1, sift_cap: int = 0) -> LPResult:
    """min cost'x + offset  s.t. row_lo <= A x <= row_up, col_lb <= x <= col_ub (A: scipy CSR)."""
    a = a_csr.tocsr()
    a.sort_indices()
    out = _core.solve_lp(
        np.ascontiguousarray(cost, dtype=np.float64), a.indptr.astype(np.int64), a.indices.astype(np.int32),
        a.data.astype(np.float64), np.ascontiguousarray(row_lo, dtype=np.float64),
        np.ascontiguousarray(row_up, dtype=np.float64), np.ascontiguousarray(col_lb, dtype=np.float64),
        np.ascontiguousarray(col_ub, dtype=np.float64), float(offset), method,
        1e30 if time_limit is None else float(time_limit), feas_tol, opt_tol, refactor, perturb, seed, sift_cap)
    ok = out["status"] == "optimal"
    return LPResult(status=out["status"], obj=out["obj"] if ok else None, x=out["x"] if ok else None,
                    row_dual=out["row_dual"] if ok else None, red_cost=out["red_cost"] if ok else None,
                    runtime=out["runtime"], stats=dict(out["stats"]))


def solve_lp(model: Model, method: str = "auto", time_limit: float | None = None, **kw) -> LPResult:
    """LP relaxation (0 <= x <= 1) of a set covering / partitioning model."""
    a, lower, upper = model.to_scipy()
    n = model.n_cols
    return solve_lp_arrays(model.cost, a, lower, upper, np.zeros(n), np.ones(n), offset=model.obj_offset,
                           method=method, time_limit=time_limit, **kw)
