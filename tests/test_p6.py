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
