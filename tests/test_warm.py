"""Warm starts that imitate a good heuristic solution (bench.warm)."""
import numpy as np
import pytest

import scsp
from scsp.bench.warm import perturb
from scsp.gen import generate_dms, generate_sp


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_start_is_feasible_and_inside_the_window(seed):
    model = generate_dms(n_orders=60, n_bundles=1200, seed=seed)
    best = scsp.solve(model, time_limit=1.0, threads=1)
    assert best.obj is not None
    cols, obj, _ = perturb(model, np.flatnonzero(best.x > 0.5), best.obj, gap=3.0, window=1.0, seed=seed)
    x = np.zeros(model.n_cols)
    x[cols] = 1.0
    assert model.is_feasible(x)
    assert obj == pytest.approx(model.objective(x))
    assert best.obj * 1.02 <= obj <= best.obj * 1.04
    again = perturb(model, np.flatnonzero(best.x > 0.5), best.obj, gap=3.0, window=1.0, seed=seed)
    assert again[0] == cols                                # same seed, same start


def test_solver_never_returns_worse_than_the_start():
    model = generate_dms(n_orders=60, n_bundles=1200, seed=4)
    best = scsp.solve(model, time_limit=1.0, threads=1)
    cols, obj, _ = perturb(model, np.flatnonzero(best.x > 0.5), best.obj, gap=3.0, window=1.0, seed=7)
    for bnb in (True, False):
        res = scsp.solve(model, time_limit=0.05, threads=2, warm_start=cols, bnb=bnb)
        assert res.obj <= obj + 1e-9
        assert res.stats["warm_feasible"]


def test_partitioning_models_are_rejected():
    model = generate_sp(20, 300, seed=1)
    with pytest.raises(ValueError):
        perturb(model, model.meta["warm_start"], 1.0, gap=2.0, window=0.5, seed=1)
