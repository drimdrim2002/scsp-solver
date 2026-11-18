"""LP relaxation of a Model, solved by the native LP engine (P2), and the LP that stays
loaded for branch and bound (`LpSolver`, P3).

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


class LpSolver(_core.LpSolver):
    """An LP that stays loaded (the interface branch and bound uses).

        lp = scsp.LpSolver(model)                 # LP relaxation, 0 <= x <= 1
        lp.solve()                                # "optimal" | "infeasible" | "unbounded" | "limit" | "failed" | "cutoff"
        lp.obj, lp.x, lp.red_cost, lp.row_dual()
        lp.set_col_bounds(j, 1.0, 1.0)            # fix a column; the next solve() continues from the last basis
        basis = lp.get_basis(); lp.set_basis(basis)
        lp.set_cutoff(z)                          # solve() returns "cutoff" once the dual objective exceeds z

        lp.add_rows(row_ptr, col_idx, val, row_lo, row_up)   # cutting planes, row-wise over the columns;
        lp.solve()                                # ... the dual simplex repairs the violated rows from the last basis
        lp.row_status(), lp.row_activity          # 0 = not tight; lp.remove_rows([...]) keeps the basis for such rows
    """

    def __init__(self, model: Model, seed: int = 1, time_limit: float | None = None):
        a, lower, upper = model.to_scipy()
        n = model.n_cols
        self._init_arrays(model.cost, a, lower, upper, np.zeros(n), np.ones(n), model.obj_offset, seed, time_limit)

    def _init_arrays(self, cost, a_csr, row_lo, row_up, col_lb, col_ub, offset, seed, time_limit):
        a = a_csr.tocsr()
        a.sort_indices()
        _core.LpSolver.__init__(
            self, np.ascontiguousarray(cost, dtype=np.float64), a.indptr.astype(np.int64), a.indices.astype(np.int32),
            a.data.astype(np.float64), np.ascontiguousarray(row_lo, dtype=np.float64),
            np.ascontiguousarray(row_up, dtype=np.float64), np.ascontiguousarray(col_lb, dtype=np.float64),
            np.ascontiguousarray(col_ub, dtype=np.float64), float(offset), int(seed),
            1e30 if time_limit is None else float(time_limit))

    @classmethod
    def from_arrays(cls, cost, a_csr, row_lo, row_up, col_lb, col_ub, offset: float = 0.0, seed: int = 1,
                    time_limit: float | None = None) -> "LpSolver":
        """min cost'x + offset  s.t. row_lo <= A x <= row_up, col_lb <= x <= col_ub (A: scipy CSR)."""
        self = cls.__new__(cls)
        self._init_arrays(cost, a_csr, row_lo, row_up, col_lb, col_ub, offset, seed, time_limit)
        return self


def solve_lp(model: Model, method: str = "auto", time_limit: float | None = None, **kw) -> LPResult:
    """LP relaxation (0 <= x <= 1) of a set covering / partitioning model."""
    a, lower, upper = model.to_scipy()
    n = model.n_cols
    return solve_lp_arrays(model.cost, a, lower, upper, np.zeros(n), np.ones(n), offset=model.obj_offset,
                           method=method, time_limit=time_limit, **kw)
