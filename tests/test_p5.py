"""P5: parallel branch-and-cut tree (BnbTeam cooperative workers, LpSolver::clone),
gap-driven SPP tree/LNS scheduling, and improved SC tree giveup."""
import numpy as np
import pytest

import scsp
from scsp.gen import generate_dms, generate_sp


@pytest.mark.parametrize("seed", range(3))
def test_parallel_bnb_matches_single_thread_on_partitioning(seed):
    model = generate_sp(45, 600, seed=seed + 31)
    ref = scsp.solve(model, time_limit=20.0, threads=1, seed=1, local_search=False)
    assert ref.status == "optimal"
    # threads=4 with local_search=False exercises 4 pure parallel B&B workers via BnbTeam
    for threads, ls in ((2, False), (4, False), (4, True)):
        res = scsp.solve(model, time_limit=20.0, threads=threads, seed=1, local_search=ls)
        assert res.status == "optimal"
        assert res.verified
        assert res.obj == pytest.approx(ref.obj, rel=1e-9)
        assert res.bound <= res.obj + 1e-7


@pytest.mark.parametrize("seed", range(3))
def test_parallel_bnb_matches_single_thread_on_covering(seed):
    model = generate_dms(n_orders=70, n_bundles=1200, seed=seed + 41)
    ref = scsp.solve(model, time_limit=20.0, threads=1, seed=1)
    assert ref.status == "optimal"
    for threads in (2, 4):
        res = scsp.solve(model, time_limit=20.0, threads=threads, seed=1)
        assert res.status == "optimal"
        assert res.verified
        assert res.obj == pytest.approx(ref.obj, rel=1e-9)
        assert res.bound <= res.obj + 1e-7


def test_parallel_bnb_warm_start_and_early_stop():
    model = generate_sp(60, 800, seed=77)
    init = scsp.solve(model, time_limit=1.0, threads=4, seed=1)
    assert init.x is not None and init.verified
    res = scsp.solve(model, time_limit=1.0, threads=4, seed=2, warm_start=init.x)
    assert res.verified
    assert res.obj <= init.obj + 1e-7
    if res.bound is not None:
        assert res.bound <= res.obj + 1e-7


def test_parallel_tree_giveup_clean_exit():
    # Tiny giveup threshold forces tree termination while helper threads may be joining/active
    model = generate_dms(n_orders=120, n_bundles=2000, seed=9)
    for seed in (1, 2):
        res = scsp.solve(model, time_limit=1.5, threads=4, seed=seed, tree_giveup=1e-9)
        assert res.verified
        assert res.obj is not None
        if res.bound is not None:
            assert res.bound <= res.obj + 1e-7
