"""P2 LP engine: status and objective against HiGHS LP (black-box oracle), duality checks."""
import numpy as np
import pytest
import scipy.sparse as sp

import scsp
from scsp.gen import generate_dms, generate_sp
from scsp.lp import solve_lp, solve_lp_arrays

highspy = pytest.importorskip("highspy")

INF = float("inf")


def highs_lp(cost, a, row_lo, row_up, col_lb, col_ub, offset=0.0):
    a = sp.csr_matrix(a)
    inf = highspy.kHighsInf
    clip = lambda v: np.clip(np.asarray(v, dtype=float), -inf, inf)
    lp = highspy.HighsLp()
    lp.num_col_ = a.shape[1]
    lp.num_row_ = a.shape[0]
    lp.col_cost_ = np.asarray(cost, dtype=float)
    lp.col_lower_ = clip(col_lb)
    lp.col_upper_ = clip(col_ub)
    lp.row_lower_ = clip(row_lo)
    lp.row_upper_ = clip(row_up)
    lp.offset_ = offset
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = a.indptr
    lp.a_matrix_.index_ = a.indices
    lp.a_matrix_.value_ = a.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    return h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value


def model_lp(model):
    a, lower, upper = model.to_scipy()
    n = model.n_cols
    return highs_lp(model.cost, a, lower, upper, np.zeros(n), np.ones(n), model.obj_offset)


def check_optimality(cost, a, row_lo, row_up, col_lb, col_ub, res, offset=0.0, tol=1e-7):
    """Primal feasibility, dual feasibility and equal primal and dual objectives."""
    a = sp.csr_matrix(a)
    x, y, d = res.x, res.row_dual, res.red_cost
    act = a @ x
    assert np.all(x >= col_lb - tol) and np.all(x <= col_ub + tol)
    assert np.all(act >= row_lo - tol) and np.all(act <= row_up + tol)
    assert np.allclose(d, cost - a.T @ y, atol=1e-7)
    # dual objective: every dual multiplies the bound its sign points at
    with np.errstate(invalid="ignore"):
        row_part = np.where(y > tol, y * row_lo, np.where(y < -tol, y * row_up, 0.0))
        col_part = np.where(d > tol, d * col_lb, np.where(d < -tol, d * col_ub, 0.0))
    assert np.all(np.isfinite(row_part)) and np.all(np.isfinite(col_part))    # no dual pointing at an infinite bound
    dual_obj = row_part.sum() + col_part.sum() + offset
    assert dual_obj == pytest.approx(res.obj, rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("method", ["dual", "sifting"])
@pytest.mark.parametrize("seed", range(6))
def test_set_models_match_highs(method, seed):
    models = [
        generate_dms(n_orders=20 + 10 * seed, n_bundles=300 + 100 * seed, seed=seed + 1),
        generate_dms(n_orders=40, n_bundles=600, seed=seed + 1, tight=True),
        generate_sp(20 + 5 * seed, 300 + 50 * seed, seed=seed + 1),
    ]
    for model in models:
        status, ref = model_lp(model)
        res = solve_lp(model, method=method)
        assert res.stats["method"] == method
        assert status == "Optimal" and res.status == "optimal"
        assert res.obj == pytest.approx(ref, rel=1e-8, abs=1e-8)
        a, lower, upper = model.to_scipy()
        n = model.n_cols
        check_optimality(model.cost, a, lower, upper, np.zeros(n), np.ones(n), res, model.obj_offset)


def random_lp(rng, m, n, infeasible=False):
    """General bounded LP: mixed row senses, negative costs, some columns without an upper bound."""
    a = sp.random(m, n, density=min(1.0, 4.0 / m + 0.1), random_state=np.random.RandomState(int(rng.integers(1 << 30))),
                  data_rvs=lambda k: rng.integers(-3, 4, size=k).astype(float)).tocsr()
    a.eliminate_zeros()
    x0 = rng.uniform(0, 1, size=n)                      # a point that keeps most instances feasible
    act = a @ x0
    kind = rng.integers(0, 4, size=m)
    row_lo = np.where(kind == 0, act, np.where(kind == 1, act - rng.uniform(0, 1, m), -INF))
    row_up = np.where(kind == 0, act, np.where(kind == 2, act + rng.uniform(0, 1, m), np.where(kind == 3, act + 1, INF)))
    row_lo = np.where(kind == 3, act - 1, row_lo)
    cost = rng.integers(-5, 10, size=n).astype(float)
    col_lb = np.where(rng.random(n) < 0.2, -1.0, 0.0)
    col_ub = np.where(rng.random(n) < 0.3, INF, 1.0 + rng.integers(0, 3, size=n))
    boxed = np.flatnonzero(np.isfinite(col_ub))
    if infeasible and len(boxed):                       # one extra row asks a column to exceed its upper bound
        j = int(rng.choice(boxed))
        a = sp.vstack([a, sp.csr_matrix(([1.0], ([0], [j])), shape=(1, n))], format="csr")
        row_lo = np.append(row_lo, col_ub[j] + 1.0)
        row_up = np.append(row_up, INF)
    return cost, a, row_lo, row_up, col_lb, col_ub


@pytest.mark.parametrize("seed", range(40))
def test_general_lp_matches_highs(seed):
    rng = np.random.default_rng(1000 + seed)
    m, n = int(rng.integers(3, 25)), int(rng.integers(4, 60))
    cost, a, row_lo, row_up, col_lb, col_ub = random_lp(rng, m, n, infeasible=seed % 4 == 3)
    status, ref = highs_lp(cost, a, row_lo, row_up, col_lb, col_ub)
    for method in ("dual", "sifting"):
        res = solve_lp_arrays(cost, a, row_lo, row_up, col_lb, col_ub, method=method)
        if status == "Optimal":
            assert res.status == "optimal", (method, res.status)
            assert res.obj == pytest.approx(ref, rel=1e-7, abs=1e-7)
            check_optimality(cost, a, row_lo, row_up, col_lb, col_ub, res)
        elif status == "Infeasible":
            assert res.status == "infeasible", (method, res.status)
        else:                                            # unbounded (HiGHS may also say "Unbounded or infeasible")
            assert res.status in ("unbounded", "infeasible"), (method, status, res.status)


def test_infeasible_and_unbounded():
    # x0 + x1 = 1 and x0 + x1 = 2 cannot both hold
    a = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    res = solve_lp_arrays([1.0, 1.0], a, [1.0, 2.0], [1.0, 2.0], [0.0, 0.0], [1.0, 1.0])
    assert res.status == "infeasible" and res.obj is None
    # min -x0 with x0 - x1 <= 1 and no upper bounds
    a = sp.csr_matrix(np.array([[1.0, -1.0]]))
    res = solve_lp_arrays([-1.0, 0.0], a, [-INF], [1.0], [0.0, 0.0], [INF, INF])
    assert res.status == "unbounded"


def test_refactor_frequency_and_perturbation_do_not_change_the_optimum():
    model = generate_dms(n_orders=60, n_bundles=1500, seed=3)
    _, ref = model_lp(model)
    for kw in (dict(refactor=1), dict(refactor=100000), dict(perturb=False), dict(method="sifting", sift_cap=7),
               dict(method="sifting", refactor=3)):
        res = solve_lp(model, **kw)
        assert res.status == "optimal", kw
        assert res.obj == pytest.approx(ref, rel=1e-8), kw


def test_lp_bound_is_below_the_p1_solution():
    model = generate_dms(n_orders=40, n_bundles=800, seed=5)
    lp = solve_lp(model)
    sol = scsp.solve(model, time_limit=1.0)
    assert lp.status == "optimal" and sol.obj is not None
    assert lp.obj <= sol.obj + 1e-7
    assert sol.bound <= lp.obj + 1e-6 * abs(lp.obj) or sol.status == "optimal"    # Lagrangian bound cannot beat the LP bound


def test_time_limit_and_bad_input():
    model = generate_sp(60, 3000, seed=2)
    res = solve_lp(model, time_limit=0.0)
    assert res.status in ("time_limit", "optimal") and (res.obj is None) == (res.status != "optimal")
    with pytest.raises(ValueError):
        solve_lp(model, method="barrier")
