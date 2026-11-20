"""P6: hypersparse LU/LP, branching conflict/domain propagation, root GMI cuts, and SPP primal search."""
import pytest

import scsp
from scsp.gen import generate_dms, generate_sp


@pytest.mark.parametrize("seed", range(3))
def test_gmi_cuts_valid_and_tighten_covering(seed):
    model = generate_dms(n_orders=50, n_bundles=700, seed=seed + 51, tight=seed == 1)
    ref = scsp.solve(model, time_limit=15.0, threads=2, seed=1, cuts=0)
    assert ref.status == "optimal" and ref.verified
    for cm in (8, 9):
        res = scsp.solve(model, time_limit=15.0, threads=2, seed=1, cuts=cm)
        assert res.status == "optimal" and res.verified
        assert res.obj == pytest.approx(ref.obj, rel=1e-9, abs=1e-9)
        assert res.bound <= res.obj + 1e-7
        assert res.stats["lp_root_cut"] <= res.obj + 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_gmi_and_propagation_valid_on_partitioning(seed):
    model = generate_sp(45, 500, seed=seed + 61)
    ref = scsp.solve(model, time_limit=20.0, threads=1, seed=1, cuts=0, local_search=False)
    assert ref.status == "optimal" and ref.verified
    for cm in (7, 8, 15):
        res = scsp.solve(model, time_limit=20.0, threads=2, seed=1, cuts=cm, local_search=False)
        assert res.status == "optimal" and res.verified
        assert res.obj == pytest.approx(ref.obj, rel=1e-9, abs=1e-9)
        assert res.bound <= res.obj + 1e-7
        assert res.stats["lp_root_cut"] <= res.obj + 1e-6


@pytest.mark.parametrize("seed", [1, 2])
def test_large_spp_fast_incumbent_and_lns_valid(seed):
    model = generate_sp(120, 1200, seed=seed + 71)
    res = scsp.solve(model, time_limit=4.0, threads=4, seed=seed)
    assert res.verified
    assert res.status in ("optimal", "time_limit")
    sol_events = [ev for ev in res.trace if ev[1] is not None]
    assert sol_events, "concurrent first-incumbent sprint should find a feasible solution"
    assert sol_events[0][0] < 2.0
    assert res.bound <= res.obj + 1e-7


def test_sc_three_flip_and_dynamic_weights_pure_and_side_constrained():
    """Verify 3-flip + dynamic row weights on pure SCP and side-constrained DMS with warm start."""
    import numpy as np

    # Pure SCP: col 0 covers all 6 rows at cost 10; cols 1 and 2 cover {0,1,2} and {3,4,5}
    # at cost 4 each (total 8 < 10), testing 1-drop + 2-add (3-flip) and dynamic row weights.
    rows = [
        [0, 1, 3, 6],
        [0, 1, 4, 7],
        [0, 1, 5, 6],
        [0, 2, 3, 7],
        [0, 2, 4, 6],
        [0, 2, 5, 7],
    ]
    costs = np.array([10.0, 4.0, 4.0, 6.0, 6.0, 6.0, 7.0, 7.0])
    model = scsp.Model.from_rows("p6_3flip_pure", costs, rows, "G")
    res = scsp.solve(model, time_limit=3.0, threads=2, seed=1, bnb=False, enumerate=True, local_search=True)
    assert res.verified and res.obj == pytest.approx(8.0)

    # Harder pure SCP (wide gap) to exercise dynamic row-weight tabu search (wls_step).
    rng = np.random.default_rng(123)
    m, n = 160, 1200
    rlist = [[] for _ in range(m)]
    for j in range(n):
        for i in rng.choice(m, size=int(rng.integers(12, 22)), replace=False):
            rlist[int(i)].append(j)
    for i in range(m):
        if not rlist[i]:
            rlist[i].append(int(rng.integers(n)))
    big_sc = scsp.Model.from_rows("p6_wls_pure", rng.integers(1, 12, size=n).astype(float), rlist, "G")
    r_wls = scsp.solve(big_sc, time_limit=1.0, threads=4, seed=1, bnb=False, enumerate=True, local_search=True)
    assert r_wls.verified
    assert r_wls.stats["ls_moves"] > 0

    # Side-constrained DMS with warm start: must remain feasible and never worsen warm start.
    dms = generate_dms(n_orders=80, n_bundles=1200, seed=91, tight=True)
    r0 = scsp.solve(dms, time_limit=1.0, threads=2, seed=1, bnb=False)
    assert r0.verified
    r1 = scsp.solve(dms, time_limit=2.0, threads=4, seed=2, warm_start=r0.cols)
    assert r1.verified
    assert r1.obj <= r0.obj + 1e-9
