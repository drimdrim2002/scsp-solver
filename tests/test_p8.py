"""P8 tests: knapsack cover/MIR cuts, conflict-graph cliques, {0,1/d} covering cuts, presolve probing."""
import numpy as np
import pytest

import scsp
from scsp.gen import generate_dms, generate_sp


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_p8_knapsack_cuts_exact_on_tight_dms(seed):
    dms = generate_dms(n_orders=70, n_bundles=900, seed=seed + 801, tight=True)
    ref = scsp.solve(dms, time_limit=15.0, threads=2, seed=1, cuts=0, presolve=False)
    res = scsp.solve(dms, time_limit=15.0, threads=4, seed=seed, cuts=-1, presolve=True)
    assert ref.status == "optimal" and res.status == "optimal"
    assert res.verified and res.obj == pytest.approx(ref.obj, rel=1e-9, abs=1e-9)
    assert res.bound <= res.obj + 1e-7
    # P7.5 observability check: P7 counters are exposed in Python stats.
    for key in ("node_cuts", "node_cut_rounds", "root_probed", "learned_clauses"):
        assert key in res.stats


def test_p8_presolve_dms_mandatory_side_capacity():
    # Cover row 0: {0, 1} both use >= 6 of side resource 0 (rhs = 10).
    # Cover row 1: {2, 3} where col 2 uses 5 of side resource 0 (5 + 6 = 11 > 10 -> col 2 infeasible!)
    # and col 3 uses 3 of side resource 0 (3 + 6 = 9 <= 10 -> col 3 forced!).
    m = scsp.Model.from_rows(
        "p8_dms_side_probe",
        np.array([2.0, 3.0, 1.0, 10.0]),
        [[0, 1], [2, 3]],
        "G",
        side=[([0, 1, 2, 3], [6.0, 7.0, 5.0, 3.0], "L", 10.0)],
    )
    r_pre = scsp.solve(m, time_limit=5.0, threads=1, seed=1, presolve=True)
    r_no = scsp.solve(m, time_limit=5.0, threads=1, seed=1, presolve=False)
    assert r_pre.status == "optimal" and r_no.status == "optimal"
    assert r_pre.verified and r_pre.obj == pytest.approx(12.0)
    assert r_no.obj == pytest.approx(12.0)


def test_p8_presolve_spp_size2_common_conflict():
    # Equality row 0: {0, 1} (x0 + x1 = 1).
    # Packing row 1: {0, 2}, packing row 2: {1, 2} -> col 2 conflicts with both 0 and 1, so x2 = 0 is forced!
    # Equality row 3: {2, 3} -> x3 = 1 is forced!
    rows = [[0, 1], [0, 2], [1, 2], [2, 3]]
    m = scsp.Model.from_rows("p8_spp_sz2", np.array([4.0, 5.0, 1.0, 9.0]), rows, ["E", "L", "L", "E"])
    r_pre = scsp.solve(m, time_limit=5.0, threads=1, seed=1, presolve=True)
    r_no = scsp.solve(m, time_limit=5.0, threads=1, seed=1, presolve=False)
    assert r_pre.status == "optimal" and r_no.status == "optimal"
    assert r_pre.verified and r_pre.obj == pytest.approx(13.0)
    assert r_no.obj == pytest.approx(13.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_p8_spp_probing_cliques_and_sos1_exact(seed):
    sp = generate_sp(55, 700, seed=seed + 820)
    ref = scsp.solve(sp, time_limit=20.0, threads=1, seed=1, cuts=0, local_search=False)
    res = scsp.solve(sp, time_limit=20.0, threads=4, seed=seed, cuts=15, local_search=False)
    assert ref.status == "optimal" and res.status == "optimal"
    assert res.verified and res.obj == pytest.approx(ref.obj, rel=1e-9, abs=1e-9)
    assert res.bound <= res.obj + 1e-7
    for key in ("node_cuts", "node_cut_rounds", "root_probed", "learned_clauses", "rins_calls", "rins_found"):
        assert key in res.stats
