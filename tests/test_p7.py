"""P7 tests: shallow node cuts + aging, conflict learning + root probing, in-tree sifting, side-constrained WLS."""
import pytest

import scsp
from scsp.gen import generate_dms, generate_sp


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_p7_1_shallow_node_cuts_exact_dms_and_sp(seed):
    dms = generate_dms(n_orders=60, n_bundles=800, seed=seed + 101, tight=seed == 2)
    ref_d = scsp.solve(dms, time_limit=15.0, threads=2, seed=1, cuts=0)
    res_d = scsp.solve(dms, time_limit=15.0, threads=4, seed=seed, cuts=9)
    assert ref_d.status == "optimal" and res_d.status == "optimal"
    assert res_d.verified and res_d.obj == pytest.approx(ref_d.obj, rel=1e-9, abs=1e-9)
    assert res_d.bound <= res_d.obj + 1e-7

    sp = generate_sp(55, 650, seed=seed + 121)
    ref_s = scsp.solve(sp, time_limit=20.0, threads=1, seed=1, cuts=0, local_search=False)
    res_s = scsp.solve(sp, time_limit=20.0, threads=4, seed=seed, cuts=15, local_search=False)
    assert ref_s.status == "optimal" and res_s.status == "optimal"
    assert res_s.verified and res_s.obj == pytest.approx(ref_s.obj, rel=1e-9, abs=1e-9)
    assert res_s.bound <= res_s.obj + 1e-7


def test_p7_2_probing_and_conflict_learning_infeasible_and_spp():
    import numpy as np

    # Partitioning instance with forced contradiction via 1-step conflict probing:
    # row 0: {0, 1}, row 1: {2, 3}, row 2 (pack <=1): {0, 2}, row 3 (pack <=1): {0, 3}.
    # Choosing x_0 = 1 forbids both x_2 and x_3, making row 1 uncoverable, so x_0 = 0 is forced!
    rows = [
        [0, 1],
        [2, 3],
        [0, 2],
        [0, 3],
    ]
    m = scsp.Model.from_rows("p7_probe_force0", np.array([1.0, 5.0, 2.0, 2.0]), rows, ["E", "E", "L", "L"])
    r_pre = scsp.solve(m, time_limit=5.0, threads=2, seed=1, presolve=True)
    r_nopre = scsp.solve(m, time_limit=5.0, threads=2, seed=1, presolve=False)
    assert r_pre.status == "optimal" and r_pre.verified and r_pre.obj == pytest.approx(7.0)
    assert r_nopre.status == "optimal" and r_nopre.verified and r_nopre.obj == pytest.approx(7.0)

    # Infeasible partitioning instance where LP relaxation is feasible (x = 0.5)
    # but probing/conflict learning proves integer infeasibility.
    inf_rows = [
        [0, 1],
        [1, 2],
        [0, 2],
    ]
    inf_m = scsp.Model.from_rows("p7_inf_tri", np.array([1.0, 1.0, 1.0]), inf_rows, "E")
    r_inf = scsp.solve(inf_m, time_limit=5.0, threads=2, seed=1, presolve=False)
    assert r_inf.status == "infeasible"


def test_p7_3_in_tree_sifting_wide_model_exact_pricing():
    """Verify wide model (n > 8000) in-tree sifting never produces invalid bounds or invalid pruning."""
    import numpy as np

    # Build a 40 x 8500 partitioning model where the first 5000 columns have high cost (~100),
    # columns 5000..8460 are decoys, and the optimal columns are placed at the very end (8460..8500)
    # with distinct reduced costs so sifting must price them in accurately during B&B.
    rng = np.random.default_rng(703)
    m, n = 30, 8500
    rows = [[] for _ in range(m)]
    costs = np.full(n, 100.0)
    # First 8440 columns: random pairs/triples covering rows at high cost
    for j in range(8440):
        sz = int(rng.integers(2, 5))
        for i in rng.choice(m, size=sz, replace=False):
            rows[int(i)].append(j)
        costs[j] = float(rng.integers(60, 140))
    # Force fractional LP root + branching over columns near the end (8440..8500)
    for t in range(10):
        r0, r1, r2 = 3 * t, 3 * t + 1, 3 * t + 2
        base = 8440 + 6 * t
        rows[r0].extend([base + 0, base + 2, base + 3])
        rows[r1].extend([base + 0, base + 1, base + 4])
        rows[r2].extend([base + 1, base + 2, base + 5])
        costs[base + 0] = 10.0
        costs[base + 1] = 10.0
        costs[base + 2] = 10.0
        costs[base + 3] = 6.0
        costs[base + 4] = 6.0
        costs[base + 5] = 6.0
    model = scsp.Model.from_rows("p7_wide_sift", costs, rows, "E")
    res = scsp.solve(model, time_limit=15.0, threads=4, seed=1, presolve=False)
    assert res.status == "optimal" and res.verified
    # Optimal for each 3-row triangle is one 2-row col (cost 10) + one 1-row col (cost 6) = 16 -> total 160
    assert res.obj == pytest.approx(160.0)
    assert res.bound == pytest.approx(160.0)


def test_p7_4_side_constrained_wls_and_sc_specialist():
    import numpy as np

    dms = generate_dms(n_orders=80, n_bundles=1200, seed=704, tight=True)
    # Heuristic-only path exercises pass() + penalty-guided wls_step on ns > 0 models.
    res = scsp.solve(dms, time_limit=3.0, threads=4, seed=1, bnb=False, local_search=True)
    assert res.status in ("optimal", "feasible")
    assert res.verified and res.obj < 1e18

    # Exact reference confirms feasibility and reasonable quality.
    ref = scsp.solve(dms, time_limit=10.0, threads=4, seed=1)
    assert ref.status == "optimal" and ref.verified
    assert res.obj >= ref.obj - 1e-7

    # Pure SC with side_rows=0 and large tree_giveup fallback check (ns == 0 specialist & fallback).
    rng = np.random.default_rng(705)
    m, n = 100, 1200
    rows = [[] for _ in range(m)]
    for j in range(n):
        for i in rng.choice(m, size=int(rng.integers(4, 9)), replace=False):
            rows[int(i)].append(j)
    for i in range(m):
        if len(rows[i]) < 2:
            rows[i].extend([0, 1])
    costs = rng.integers(10, 120, size=n).astype(float)
    sc_m = scsp.Model.from_rows("p7_sc_spec", costs, rows, "G")
    sc_res = scsp.solve(sc_m, time_limit=2.0, threads=4, seed=1, bnb=True, local_search=True)
    assert sc_res.status in ("optimal", "feasible", "time_limit") and sc_res.verified
    assert sc_res.obj < 1e18


def test_p7_lp_set_bounds_then_same_basis_cutoff():
    """Regression: set_basis(b) with b == st after set_col_bounds must recompute primal x before cutoff check."""
    import numpy as np
    from scsp.lp import LpSolver

    m = scsp.Model.from_rows("lp_same_basis", np.array([50.0, 60.0]), [[0, 1]], "G")
    lp = LpSolver(m)
    assert lp.solve() == "optimal"
    assert lp.obj == pytest.approx(50.0)
    basis = lp.get_basis()

    lp.set_col_bounds(1, 1.0, 1.0)
    lp.set_basis(basis)
    lp.set_cutoff(80.0)
    assert lp.solve() == "optimal"
    assert lp.obj == pytest.approx(60.0)




