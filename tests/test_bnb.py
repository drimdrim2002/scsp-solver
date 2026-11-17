"""P3: LP-based branch and bound (set covering proofs and the general 0-1 path)."""
import numpy as np
import pytest

import scsp
from scsp import Model
from scsp.gen import generate_dms, generate_sp

highspy = pytest.importorskip("highspy")


def highs_opt(model):
    """Optimal value from HiGHS MIP (black box): ("optimal", value) or ("infeasible", None)."""
    a, lower, upper = model.to_scipy()
    a = a.tocsr()
    n = model.n_cols
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = a.shape[0]
    lp.col_cost_ = model.cost
    lp.col_lower_ = np.zeros(n)
    lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.where(np.isinf(lower), -inf, lower)
    lp.row_upper_ = np.where(np.isinf(upper), inf, upper)
    lp.offset_ = model.obj_offset
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = a.indptr
    lp.a_matrix_.index_ = a.indices
    lp.a_matrix_.value_ = a.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.0)
    h.passModel(lp)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    if status == "Optimal":
        return "optimal", h.getInfo().objective_function_value
    assert status == "Infeasible", status
    return "infeasible", None


def random_mixed(rng, m, n):
    """Covering, partitioning and packing rows plus a >= and a <= side constraint."""
    rows = [[] for _ in range(m)]
    for j in range(n):
        for i in rng.choice(m, size=int(rng.integers(1, 4)), replace=False):
            rows[i].append(j)
    for i in range(m):
        if not rows[i]:
            rows[i].append(int(rng.integers(n)))
    sense = rng.choice(["G", "E", "L"], size=m, p=[0.5, 0.3, 0.2])
    cost = rng.integers(1, 30, size=n).astype(float)
    idx = np.arange(n)
    side = [(idx, rng.integers(1, 4, size=n).astype(float), "L", float(n)),
            (idx, np.ones(n), "G", 2.0)]
    return Model.from_rows("mixed", cost, [sorted(set(r)) for r in rows], sense, side=side)


@pytest.mark.parametrize("seed", range(8))
def test_partitioning_matches_highs(seed):
    model = generate_sp(12 + 2 * seed, 80 + 20 * seed, seed=seed + 1)
    status, ref = highs_opt(model)
    res = scsp.solve(model, time_limit=20.0, seed=seed)
    assert status == "optimal" and res.status == "optimal" and res.verified
    assert res.stats["path"] == "bin" and res.stats["bnb_status"] == "exhausted"
    assert res.obj == pytest.approx(ref) and res.bound == pytest.approx(ref)


@pytest.mark.parametrize("seed", range(12))
def test_mixed_rows_match_highs(seed):
    rng = np.random.default_rng(300 + seed)
    model = random_mixed(rng, int(rng.integers(6, 16)), int(rng.integers(15, 60)))
    status, ref = highs_opt(model)
    res = scsp.solve(model, time_limit=20.0, seed=seed)
    if status == "optimal":
        assert res.status == "optimal" and res.verified
        assert res.obj == pytest.approx(ref)
    else:
        assert status == "infeasible" and res.status == "infeasible" and res.x is None


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_covering_proof_does_not_depend_on_threads(threads):
    model = generate_dms(n_orders=30, n_bundles=400, seed=7)
    status, ref = highs_opt(model)
    res = scsp.solve(model, time_limit=20.0, threads=threads, seed=3)
    assert status == "optimal" and res.status == "optimal" and res.verified
    assert res.obj == pytest.approx(ref, rel=1e-9)
    assert res.stats["path"] == "sc"


def test_branch_and_bound_can_be_switched_off():
    model = generate_dms(n_orders=30, n_bundles=400, seed=7)
    a = scsp.solve(model, time_limit=1.0, bnb=False)
    assert a.stats["bnb_status"] == "" and a.obj is not None
    b = scsp.solve(model, time_limit=20.0)
    assert b.status == "optimal" and b.obj <= a.obj + 1e-9
    assert b.stats["lp_root"] <= b.obj + 1e-9


def test_infeasible_partitioning():
    # rows {0,1} and {0} and {1}: both columns are needed and collide in the first row
    m = Model.from_rows("inf", [1.0, 1.0], [[0, 1], [0], [1]], "E")
    res = scsp.solve(m, time_limit=1.0)
    assert res.status == "infeasible" and res.x is None


def test_partitioning_warm_start_is_kept():
    model = generate_sp(30, 400, seed=5)
    first = scsp.solve(model, time_limit=20.0)
    assert first.status == "optimal"
    again = scsp.solve(model, time_limit=0.0, warm_start=first.x)
    assert again.stats["warm_feasible"] and again.obj == pytest.approx(first.obj)
