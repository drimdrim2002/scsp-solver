"""LpSolver.add_rows / remove_rows (P4): adding rows must give the same LP as building it with them."""
import numpy as np
import pytest
import scipy.sparse as sp

import scsp
from scsp import _core
from scsp.gen import generate_dms, generate_sp
from scsp.lp import LpSolver, solve_lp_arrays

if not hasattr(_core.LpSolver, "row_status"):
    pytest.skip("the installed scsp._core predates add_rows (rebuild the package)", allow_module_level=True)


def arrays(model):
    a, lower, upper = model.to_scipy()
    return a.tocsr(), lower, upper


def random_rows(rng, n, r, x):
    """r random rows; about half of them cut the point x off."""
    rows = []
    for _ in range(r):
        idx = np.sort(rng.choice(n, size=int(rng.integers(2, 8)), replace=False))
        val = rng.integers(1, 4, size=len(idx)).astype(float)
        act = float(val @ x[idx])
        if rng.random() < 0.5:
            rows.append((idx, val, act + rng.uniform(0.05, 0.5), np.inf))      # violated: needs more
        else:
            rows.append((idx, val, -np.inf, act + rng.uniform(0.0, 0.5)))      # satisfied
    return rows


def as_csr(rows):
    ptr = np.cumsum([0] + [len(r[0]) for r in rows])
    return (ptr.tolist(), np.concatenate([r[0] for r in rows]).tolist(), np.concatenate([r[1] for r in rows]).tolist(),
            [r[2] for r in rows], [r[3] for r in rows])


@pytest.mark.parametrize("seed", range(8))
def test_added_rows_match_a_fresh_solve(seed):
    rng = np.random.default_rng(seed)
    model = (generate_sp(25, 300, seed=seed + 1) if seed % 2 else generate_dms(n_orders=30, n_bundles=300, seed=seed + 1))
    a, lower, upper = arrays(model)
    n = model.n_cols
    lp = LpSolver(model)
    assert lp.solve() == "optimal"
    for _ in range(3):                                   # three rounds of rows on top of each other
        it0 = lp.iterations
        rows = random_rows(rng, n, 6, lp.x)
        ptr, idx, val, lo, up = as_csr(rows)
        lp.add_rows(ptr, idx, val, lo, up)
        extra = sp.csr_matrix((val, idx, ptr), shape=(len(rows), n))
        a = sp.vstack([a, extra], format="csr")
        lower, upper = np.concatenate([lower, lo]), np.concatenate([upper, up])
        ref = solve_lp_arrays(model.cost, a, lower, upper, np.zeros(n), np.ones(n), offset=model.obj_offset)
        status = lp.solve()
        assert status == ref.status
        if status != "optimal":
            break
        assert lp.obj == pytest.approx(ref.obj, rel=1e-8, abs=1e-8)
        assert lp.n_rows == a.shape[0]
        act = a @ lp.x
        assert np.all(act >= lower - 1e-6) and np.all(act <= upper + 1e-6)
        assert np.allclose(lp.row_activity, act, atol=1e-7)
        assert lp.iterations - it0 < 0.9 * (ref.stats["dual_iters"] + ref.stats["primal_iters"] + 10)   # warm, not from scratch


def test_remove_rows_restores_the_old_lp():
    model = generate_dms(n_orders=30, n_bundles=400, seed=3)
    n = model.n_cols
    lp = LpSolver(model)
    assert lp.solve() == "optimal"
    z0, m0 = lp.obj, lp.n_rows
    rng = np.random.default_rng(1)
    rows = random_rows(rng, n, 8, lp.x)
    lp.add_rows(*as_csr(rows))
    assert lp.solve() == "optimal" and lp.obj >= z0 - 1e-9
    z1 = lp.obj
    slack = [m0 + i for i in range(8) if lp.row_status()[m0 + i] == 0]
    lp.remove_rows(slack)                                # rows that are not tight: nothing changes
    assert lp.n_rows == m0 + 8 - len(slack)
    assert lp.solve() == "optimal" and lp.obj == pytest.approx(z1, rel=1e-9)
    lp.remove_rows(m0)                                   # all added rows
    assert lp.n_rows == m0
    assert lp.solve() == "optimal" and lp.obj == pytest.approx(z0, rel=1e-9)
    lp.set_col_bounds(0, 1.0, 1.0)                       # the solver still works incrementally
    assert lp.solve() in ("optimal", "infeasible")


def test_bad_rows_are_rejected():
    model = generate_sp(20, 200, seed=1)
    lp = LpSolver(model)
    lp.solve()
    with pytest.raises(ValueError):
        lp.add_rows([0, 1], [model.n_cols], [1.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        lp.add_rows([0, 2], [0], [1.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        lp.remove_rows([lp.n_rows])
    lp.add_rows([0], [], [], [], [])                     # no rows: nothing happens
    assert lp.solve() == "optimal"
