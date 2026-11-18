"""P4: cutting planes at the root, node-level fixing, set partitioning presolve, neighbourhood search,
handing the tree thread back. Every switch must leave the optimum unchanged."""
import itertools

import numpy as np
import pytest

import scsp
from scsp import Model, _core
from scsp.gen import generate_dms, generate_sp

if "cuts" not in (_core.solve_sc.__doc__ or ""):
    pytest.skip("the installed scsp._core predates P4 (rebuild the package)", allow_module_level=True)


def brute_force(model):
    best = None
    n = model.n_cols
    for bits in itertools.product((0.0, 1.0), repeat=n):
        x = np.array(bits)
        if model.is_feasible(x):
            v = model.objective(x)
            best = v if best is None else min(best, v)
    return best


@pytest.mark.parametrize("seed", range(4))
def test_cuts_do_not_change_the_optimum_of_covering_models(seed):
    model = generate_dms(n_orders=60, n_bundles=900, seed=seed + 11)
    ref = scsp.solve(model, time_limit=20.0, threads=2, cuts=0)
    assert ref.status == "optimal"
    for cuts in (1, None):
        res = scsp.solve(model, time_limit=20.0, threads=2, cuts=cuts)
        assert res.status == "optimal"
        assert res.obj == pytest.approx(ref.obj, rel=1e-9)
        if res.stats["lp_root"] is not None:              # None: solved before the root LP was needed
            assert res.stats["lp_root_cut"] >= res.stats["lp_root"] - 1e-7
            assert res.stats["lp_root_cut"] <= res.obj + 1e-7     # a cut that removed the optimum would push the bound past it


@pytest.mark.parametrize("seed", range(4))
def test_cuts_presolve_and_search_switches_agree_on_partitioning_models(seed):
    model = generate_sp(40, 500, seed=seed + 21)
    ref = scsp.solve(model, time_limit=20.0, threads=1, cuts=0, presolve=False, local_search=False)
    assert ref.status == "optimal"
    seen_cuts = False
    for kw in (dict(cuts=1), dict(cuts=2), dict(cuts=4), dict(cuts=7), dict(presolve=True), dict(local_search=True),
               dict(threads=4), dict()):
        args = dict(time_limit=20.0, threads=1, cuts=0, presolve=False, local_search=False)
        args.update(kw)
        if not kw:
            args = dict(time_limit=20.0, threads=4)       # everything on, as shipped
        res = scsp.solve(model, **args)
        assert res.status == "optimal", kw
        assert res.obj == pytest.approx(ref.obj, rel=1e-9), kw
        if res.stats["lp_root"] is not None:
            assert res.stats["lp_root_cut"] <= res.obj + 1e-7, kw
        seen_cuts = seen_cuts or res.stats["cut_rounds"] > 0
    assert seen_cuts or ref.stats["bnb_nodes"] <= 1       # the cut rounds ran unless the root was already integral


def test_partitioning_presolve_reductions():
    # rows 0..3; row 0 = {0, 1}, row 1 = {0, 1, 2}  -> column 2 must be zero and row 1 repeats row 0;
    # columns 3 and 4 are identical (4 dearer); row 3 = {6} fixes column 6 at one, which empties row 2's others.
    cols = [[0, 1], [0, 1], [1], [2], [2], [2, 3], [3]]
    cost = [5.0, 7.0, 1.0, 4.0, 6.0, 1.0, 9.0]
    model = Model.from_columns("presolve", np.array(cost), 4,
                               np.cumsum([0] + [len(c) for c in cols]), np.concatenate(cols), "E")
    best = brute_force(model)
    on = scsp.solve(model, time_limit=5.0, threads=1)
    off = scsp.solve(model, time_limit=5.0, threads=1, presolve=False)
    assert on.status == off.status == "optimal"
    assert on.obj == pytest.approx(best) and off.obj == pytest.approx(best)
    assert model.is_feasible(on.x)
    assert on.stats["red_cols"] < model.n_cols and on.stats["red_rows"] < model.n_rows
    assert on.stats["fixed_cols"] >= 1


@pytest.mark.parametrize("seed", range(6))
def test_small_partitioning_models_match_brute_force_with_everything_on(seed):
    rng = np.random.default_rng(100 + seed)
    m, n = 7, 16
    cols = [np.sort(rng.choice(m, size=int(rng.integers(1, 4)), replace=False)) for _ in range(n)]
    if seed % 2:
        cols[3] = cols[2].copy()                             # an identical pair
    model = Model.from_columns(f"sp{seed}", rng.integers(1, 20, size=n).astype(float), m,
                               np.cumsum([0] + [len(c) for c in cols]), np.concatenate(cols), "E")
    best = brute_force(model)
    for threads in (1, 3):
        res = scsp.solve(model, time_limit=5.0, threads=threads)
        if best is None:
            assert res.status == "infeasible"
        else:
            assert res.status == "optimal" and res.obj == pytest.approx(best)
            assert model.is_feasible(res.x)


def test_handing_the_tree_thread_back_keeps_results_valid():
    model = generate_dms(n_orders=80, n_bundles=1500, seed=5)
    ref = scsp.solve(model, time_limit=20.0, threads=4, tree_giveup=0)
    assert ref.status == "optimal"
    for giveup in (None, 1e-9):                              # default rule; a rule that gives up at the first reading
        res = scsp.solve(model, time_limit=3.0, threads=4, tree_giveup=giveup)
        assert model.is_feasible(res.x)
        assert res.obj >= ref.obj - 1e-7
        assert res.bound <= ref.obj + 1e-7
        if res.status == "optimal":
            assert res.obj == pytest.approx(ref.obj, rel=1e-9)
