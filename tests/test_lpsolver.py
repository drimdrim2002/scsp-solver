"""scsp.LpSolver: the LP that stays loaded (bound changes, warm re-solve, basis, cutoff)."""
import numpy as np
import pytest

import scsp
from scsp.gen import generate_dms, generate_sp
from scsp.lp import LpSolver, solve_lp, solve_lp_arrays


def fresh(model, lb, ub):
    """Reference: a one-shot solve of the same LP with the given column bounds."""
    a, lower, upper = model.to_scipy()
    return solve_lp_arrays(model.cost, a, lower, upper, lb, ub, offset=model.obj_offset, method="dual")


@pytest.mark.parametrize("make", [lambda: generate_dms(n_orders=40, n_bundles=600, seed=2),
                                  lambda: generate_sp(30, 400, seed=3)])
def test_first_solve_matches_one_shot(make):
    model = make()
    ref = solve_lp(model)
    lp = LpSolver(model)
    assert (lp.n_rows, lp.n_cols) == (model.n_cons, model.n_cols)
    assert lp.solve() == "optimal"
    assert lp.obj == pytest.approx(ref.obj, rel=1e-9)
    assert np.allclose(lp.red_cost, model.cost - model.to_scipy()[0].T @ lp.row_dual(), atol=1e-7)
    assert lp.iterations > 0


@pytest.mark.parametrize("seed", range(5))
def test_bound_changes_match_a_fresh_solve(seed):
    rng = np.random.default_rng(seed)
    model = generate_dms(n_orders=30, n_bundles=400, seed=seed + 1)
    n = model.n_cols
    lp = LpSolver(model)
    assert lp.solve() == "optimal"
    lb, ub = np.zeros(n), np.ones(n)
    for step in range(25):
        x = lp.x
        frac = np.flatnonzero((x > 1e-6) & (x < 1 - 1e-6))
        j = int(rng.choice(frac)) if len(frac) and step % 5 != 4 else int(rng.integers(n))
        if lb[j] == ub[j]:                      # release a fixed column again (a tree search backtracks like this)
            lb[j], ub[j] = 0.0, 1.0
        else:
            lb[j] = ub[j] = float(rng.integers(0, 2))
        lp.set_col_bounds(j, lb[j], ub[j])
        status = lp.solve()
        ref = fresh(model, lb, ub)
        assert status == ref.status, (step, status, ref.status)
        if status == "optimal":
            assert lp.obj == pytest.approx(ref.obj, rel=1e-8, abs=1e-8)
            assert np.all(lp.x >= lb - 1e-7) and np.all(lp.x <= ub + 1e-7)
        else:                                   # infeasible: undo and carry on
            lb[j], ub[j] = 0.0, 1.0
            lp.set_col_bounds(j, 0.0, 1.0)
            assert lp.solve() == "optimal"


def test_warm_resolve_is_cheaper_than_from_scratch():
    model = generate_dms(n_orders=60, n_bundles=1500, seed=4)
    lp = LpSolver(model)
    lp.solve()
    first = lp.iterations
    x = lp.x
    j = int(np.argmax(np.minimum(x, 1 - x)))
    lp.set_col_bounds(j, 1.0, 1.0)
    assert lp.solve() == "optimal"
    assert lp.iterations - first < first / 2


def test_basis_round_trip():
    model = generate_sp(30, 400, seed=6)
    lp = LpSolver(model)
    lp.solve()
    z0, basis = lp.obj, lp.get_basis()
    assert basis.shape == (model.n_cons + model.n_cols,) and int((basis == 0).sum()) == model.n_cons
    x = lp.x
    j = int(np.argmax(np.minimum(x, 1 - x)))
    lp.set_col_bounds(j, 0.0, 0.0)
    lp.solve()
    lp.set_col_bounds(j, 0.0, 1.0)
    lp.set_basis(basis)
    before = lp.iterations
    assert lp.solve() == "optimal" and lp.obj == pytest.approx(z0)
    assert lp.iterations == before             # the saved basis is still optimal: no pivots
    bad = basis.copy()
    bad[:] = 1
    with pytest.raises(ValueError):
        lp.set_basis(bad)
    with pytest.raises(ValueError):
        lp.set_basis(basis[:-1])


def test_cutoff_and_iteration_limit():
    model = generate_dms(n_orders=60, n_bundles=1500, seed=4)
    lp = LpSolver(model)
    assert lp.solve() == "optimal"
    z = lp.obj
    x = lp.x
    order = np.argsort(-np.minimum(x, 1 - x))
    for j in order[:3]:
        lp.set_col_bounds(int(j), 0.0, 0.0)     # push the LP value up
    probe = LpSolver(model)
    probe.solve()
    for j in order[:3]:
        probe.set_col_bounds(int(j), 0.0, 0.0)
    assert probe.solve() == "optimal" and probe.obj > z + 1e-6
    lp.set_cutoff(z + 0.25 * (probe.obj - z))   # below the new LP value: the solve must stop early
    assert lp.solve() == "cutoff"
    lp.set_cutoff(None)
    assert lp.solve() == "optimal" and lp.obj == pytest.approx(probe.obj)
    lp.set_cutoff(probe.obj + 1.0)              # above it: no effect
    assert lp.solve() == "optimal"
    fresh_lp = LpSolver(model)
    assert fresh_lp.solve(max_iter=3) == "limit"
    assert fresh_lp.solve() == "optimal" and fresh_lp.obj == pytest.approx(z)


def test_errors_and_p4_placeholders():
    model = generate_sp(20, 200, seed=1)
    lp = LpSolver(model)
    with pytest.raises(IndexError):
        lp.set_col_bounds(model.n_cols, 0.0, 1.0)
    with pytest.raises(ValueError):
        lp.set_col_bounds(0, 1.0, 0.0)
    with pytest.raises(NotImplementedError):
        lp.add_rows([0, 1], [0], [1.0], [1.0], [1.0])
    with pytest.raises(NotImplementedError):
        lp.remove_rows(0)
    assert scsp.LpSolver is LpSolver
