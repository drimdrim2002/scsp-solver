"""P1 solver: correctness on small models (brute force / HiGHS as a black-box oracle)."""
import itertools

import numpy as np
import pytest

import scsp
from scsp import Model, _core
from scsp.gen import generate_dms, generate_sp


def random_sc(rng, m, n, with_side):
    rows = [[] for _ in range(m)]
    for j in range(n):
        for i in rng.choice(m, size=rng.integers(1, min(m, 4) + 1), replace=False):
            rows[i].append(j)
    for i in range(m):
        if not rows[i]:
            rows[i].append(int(rng.integers(n)))
    cost = rng.integers(1, 20, size=n).astype(float)
    side = []
    if with_side:
        typ = rng.integers(0, 2, size=n)
        for t in range(2):
            idx = np.flatnonzero(typ == t)
            side.append((idx, rng.integers(1, 3, size=len(idx)).astype(float), "L", float(rng.integers(2, 7))))
    return Model.from_rows("rnd", cost, rows, "G", side=side)


def brute_force(model):
    best = None
    n = model.n_cols
    for bits in itertools.product((0.0, 1.0), repeat=n):
        x = np.array(bits)
        if model.is_feasible(x):
            v = model.objective(x)
            if best is None or v < best - 1e-9:
                best = v
    return best


@pytest.mark.parametrize("with_side", [False, True])
def test_matches_brute_force(with_side):
    rng = np.random.default_rng(7 + with_side)
    for _ in range(25):
        model = random_sc(rng, m=int(rng.integers(3, 8)), n=int(rng.integers(5, 13)), with_side=with_side)
        opt = brute_force(model)
        res = scsp.solve(model, time_limit=2.0, threads=2, seed=1)
        if opt is None:
            assert res.status in ("infeasible", "no_solution")
            continue
        assert res.status == "optimal", res
        assert res.obj == pytest.approx(opt)
        assert res.verified and res.bound == pytest.approx(opt)


def test_options_do_not_change_feasibility():
    model = generate_dms(n_orders=30, n_bundles=300, seed=3)
    for kw in ({"presolve": False}, {"local_search": False}, {"enumerate": False}, {"threads": 1}):
        res = scsp.solve(model, time_limit=0.5, seed=2, **kw)
        assert res.x is not None and model.is_feasible(res.x)
        assert res.bound is not None and res.bound <= res.obj + 1e-9


def test_dms_matches_highs_oracle():
    highspy = pytest.importorskip("highspy")
    del highspy
    from scsp.bench.runners import run_highs

    for seed in (1, 2):
        model = generate_dms(n_orders=40, n_bundles=400, seed=seed, tight=seed == 2)
        ref = run_highs(model, time_limit=20.0, threads=2, seed=1)
        assert ref.status == "optimal"
        res = scsp.solve(model, time_limit=5.0, threads=2, seed=1)
        assert model.is_feasible(res.x)
        assert res.obj >= ref.obj - 1e-6 * abs(ref.obj)                 # never below the proven optimum
        assert res.obj <= ref.obj * 1.01                                # and close to it
        assert res.bound <= ref.obj + 1e-6 * abs(ref.obj)               # the bound is valid
        if res.status == "optimal":
            assert res.obj == pytest.approx(ref.obj, rel=1e-7)


def test_warm_start_is_never_lost():
    model = generate_dms(n_orders=40, n_bundles=400, seed=5)
    warm = model.meta["warm_start"]
    x0 = np.zeros(model.n_cols)
    x0[warm] = 1.0
    res = scsp.solve(model, time_limit=0.05, threads=1, seed=1, warm_start=warm)
    assert res.stats["warm_feasible"] and res.obj <= model.objective(x0) + 1e-9
    res2 = scsp.solve(model, time_limit=0.05, threads=1, seed=1, warm_start=x0)   # 0/1 vector form
    assert res2.obj <= model.objective(x0) + 1e-9
    # an infeasible warm start is ignored, not returned
    res3 = scsp.solve(model, time_limit=0.2, threads=1, seed=1, warm_start=[0])
    assert not res3.stats["warm_feasible"] and model.is_feasible(res3.x)


def test_presolve_maps_back():
    model = generate_dms(n_orders=30, n_bundles=600, seed=9)
    d = _core.presolve_sc(model.cost, model.row_ptr, model.col_idx, model.side_ptr, model.side_idx,
                          model.side_val, model.side_rhs, 0.0)
    assert not d["infeasible"] and d["n_cols"] <= model.n_cols and d["n_rows"] <= model.n_rows
    assert len(d["col_orig"]) == d["n_cols"] and len(set(d["col_orig"])) == d["n_cols"]
    a = scsp.solve(model, time_limit=1.0, threads=1, presolve=True)
    b = scsp.solve(model, time_limit=1.0, threads=1, presolve=False)
    if a.status == b.status == "optimal":
        assert a.obj == pytest.approx(b.obj)


def test_non_covering_models_take_the_general_path():
    sp = generate_sp(n_rows=20, n_cols=100, seed=1)
    res = scsp.solve(sp, time_limit=5.0)
    assert res.stats["path"] == "bin" and res.status == "optimal" and res.verified
    m = Model.from_rows("ge", [1.0, 1.0], [[0, 1]], "G", side=[([0, 1], [1.0, 1.0], "G", 2.0)])
    res = scsp.solve(m, time_limit=1.0)
    assert res.stats["path"] == "bin" and res.status == "optimal" and res.obj == pytest.approx(2.0)


def test_infeasible_model():
    m = Model.from_rows("inf", [1.0, 1.0], [[0], [1]], "G", side=[([0, 1], [1.0, 1.0], "L", 1.0)])
    res = scsp.solve(m, time_limit=0.2)
    assert res.status in ("infeasible", "no_solution") and res.x is None


class _Rider:
    def __init__(self, capa, fixed, var):
        self.info = [capa, fixed, var]

    def extract(self):
        return None, self.info


def test_dms_adapter_same_interface():
    from scsp.dms import solveSP

    rng = np.random.default_rng(1)
    K = 12
    pts = rng.uniform(0, 3000, size=(2 * K, 2))
    dist = np.rint(np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))).astype(int)
    riders = {"WALK": _Rider(70, 5000, 30), "BIKE": _Rider(100, 6000, 60), "CAR": _Rider(200, 7000, 100)}

    def bundle(t, orders):
        src, dst = list(orders), list(orders)
        d = sum(dist[src[i], src[i + 1]] for i in range(len(src) - 1)) + dist[src[-1], dst[0] + K]
        d += sum(dist[dst[i] + K, dst[i + 1] + K] for i in range(len(dst) - 1))
        _, fixed, var = riders[t].extract()[1]
        return [t, fixed + d / 100 * var, src, dst]

    bundles = [bundle("CAR", [k]) for k in range(K)]                      # start solution: one car per order
    for _ in range(150):
        size = int(rng.integers(2, 4))
        bundles.append(bundle(str(rng.choice(["WALK", "BIKE", "CAR"])), rng.choice(K, size=size, replace=False).tolist()))
    helper = [[i for i, b in enumerate(bundles) if k in b[2]] for k in range(K)]
    avail = {"WALK": 3, "BIKE": 3, "CAR": K}
    np.random.seed(0)
    best_sol, sol_edge, sol_rider, init_bundle, obj = solveSP(list(range(K)), helper, avail, bundles, 1.0, 0, dist, riders, -1)

    covered = sorted(k for _, src, _ in best_sol for k in src)
    assert covered == list(range(K))                                      # a partition of the orders
    assert all(sorted(src) == sorted(dst) for _, src, dst in best_sol)
    assert sol_edge.shape == (2 * K, 2 * K) and sol_rider.shape == (K,)
    assert init_bundle is bundles
    start_obj = sum(b[1] for b in bundles[:K]) / K
    assert obj <= start_obj + 1e-9
    used = {t: sum(1 for b in best_sol if b[0] == t) for t in avail}
    assert all(used[t] <= avail[t] for t in avail)
