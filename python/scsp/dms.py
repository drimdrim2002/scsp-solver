"""Drop-in replacement for `HighsSP.solveSP` / `GurobiSP.solveSP` of the DMS code.

    from scsp.dms import solveSP          # instead of: from HighsSP import solveSP

Same arguments and same return value:

    best_sol, sol_edge, sol_rider, init_bundle, obj = solveSP(
        initial_sol, constraint_helper, riders_available, bundle_vectors,
        timelimit, verbose, dist_mat, riders, heur_time)

The model is the one of HighsSP.py:

    min  sum_i c_i x_i / K
    s.t. sum_{i contains k} x_i >= 1            for every order k
         sum_{i : type(i) = t} x_i <= avail_t   for t in WALK, BIKE, CAR

and it is solved by the scsp set covering heuristic instead of HiGHS. The set
covering answer is turned into a partition by `sc_sol_to_answer`, which follows
the logic of the DMS code (an order covered more than once stays in the bundle
where removing it would save the least). The DMS folder itself is not imported.
"""
from __future__ import annotations

import time

import numpy as np

from .model import Model
from .solve import solve

RIDER_DICT = {"WALK": 0, "BIKE": 1, "CAR": 2}
THREADS = 4
SEED = 42


def build_model(constraint_helper, riders_available: dict, bundle_costs, bundle_riders, name: str = "dms") -> Model:
    """K cover rows (one per order) + 3 rider-type rows, cost = bundle cost / K."""
    K = len(constraint_helper)
    bundle_riders = np.asarray(bundle_riders, dtype=np.int64)
    side = []
    for t in range(3):
        idx = np.flatnonzero(bundle_riders == t)
        side.append((idx, np.ones(len(idx)), "L", float(riders_available.get(t, 0))))
    rows = [np.asarray(r, dtype=np.int64) for r in constraint_helper]
    return Model.from_rows(name, np.asarray(bundle_costs, dtype=np.float64) / K, rows, "G", side=side)


def solve_sc(initial_sol, constraint_helper, riders_available: dict, bundle_costs, bundle_riders, timelimit: float,
             verbose: int = 0, threads: int = THREADS, seed: int = SEED):
    """Returns (0/1 bool vector, objective / K, scsp.Result). Never worse than `initial_sol`."""
    t0 = time.time()
    K = len(constraint_helper)
    init = np.asarray(initial_sol, dtype=bool)
    costs = np.asarray(bundle_costs, dtype=np.float64)
    model = build_model(constraint_helper, riders_available, costs, bundle_riders)
    res = solve(model, time_limit=max(0.05, float(timelimit) - (time.time() - t0)), threads=threads, seed=seed,
                warm_start=init if init.any() else None)
    sol = init
    if res.x is not None:
        cand = res.x > 0.5
        if not init.any() or float(costs @ cand) <= float(costs @ init) + 1e-9:
            sol = cand
        elif verbose >= 1:
            print("scsp solution worse than start; keeping the start solution")
    obj = float(costs @ sol / K)
    if verbose >= 1:
        print(f"scsp status : {res.status}, gap : {res.gap}, warm start used : {res.stats.get('warm_feasible')}")
    if verbose == 2:
        print("Objective value:", obj)
    return sol, obj, res


def sc_sol_to_answer(sc_sol, bundle_vectors, dist_mat, riders):
    """Set covering -> set partitioning, same rule as the DMS code: an order covered
    more than once stays only in the bundle where removing it would save the least."""
    K = dist_mat.shape[0] // 2
    set_cover_sol = [[bundle_vectors[i][0], list(bundle_vectors[i][2]), list(bundle_vectors[i][3])]
                     for i in np.nonzero(sc_sol)[0]]

    cnt = np.zeros(K, dtype=int)
    for _, source, _ in set_cover_sol:
        for x in source:
            cnt[x] += 1

    order_pool = np.array([i for i, x in enumerate(cnt) if x > 1])
    np.random.shuffle(order_pool)

    for order in order_pool:
        best_cost = np.inf
        where_to_keep = 0
        for bundle_id, (rider_type, source, dest) in enumerate(set_cover_sol):
            if order not in source:
                continue
            _, fixed_cost, var_cost = riders[rider_type].extract()[1]
            n = len(source)
            if n == 1:
                cost_incremental = fixed_cost + dist_mat[order, order + K] / 100 * var_cost
            elif source[n - 1] == order and dest[0] == order:
                bef, aft = source[n - 2], dest[1] + K
                d = dist_mat[bef, order] + dist_mat[order, order + K] + dist_mat[order + K, aft] - dist_mat[bef, aft]
                cost_incremental = d / 100 * var_cost
            else:
                d = 0
                i = source.index(order)
                if i == 0:
                    d += dist_mat[source[i], source[i + 1]]
                elif i == n - 1:
                    d += dist_mat[source[i - 1], source[i]] + dist_mat[source[i], dest[0] + K] - dist_mat[source[i - 1], dest[0] + K]
                else:
                    d += dist_mat[source[i - 1], source[i]] + dist_mat[source[i], source[i + 1]] - dist_mat[source[i - 1], source[i + 1]]
                i = dest.index(order)
                if i == n - 1:
                    d += dist_mat[dest[i - 1], dest[i]]
                elif i == 0:
                    d += dist_mat[source[n - 1], dest[i] + K] + dist_mat[dest[i] + K, dest[i + 1] + K] - dist_mat[source[n - 1], dest[i + 1] + K]
                else:
                    d += dist_mat[dest[i - 1] + K, dest[i] + K] + dist_mat[dest[i] + K, dest[i + 1] + K] - dist_mat[dest[i - 1] + K, dest[i + 1] + K]
                cost_incremental = d / 100 * var_cost
            if cost_incremental < best_cost:
                best_cost = cost_incremental
                where_to_keep = bundle_id

        for bundle_id, (rider_type, source, dest) in enumerate(set_cover_sol):
            if order not in source or bundle_id == where_to_keep:
                continue
            source.remove(order)
            dest.remove(order)

    return [[rider_type, source, dest] for rider_type, source, dest in set_cover_sol if len(source) > 0]


def solveSP(initial_sol_for_gurobi, constraint_helper_for_gurobi, riders_available, bundle_vectors_for_gurobi,
            timelimit, verbose, dist_mat, riders, heur_time):
    """Same signature and return value as HighsSP.solveSP. `heur_time` is accepted
    for compatibility and not used."""
    K = len(dist_mat) // 2
    initial_sol_processed = np.zeros(len(bundle_vectors_for_gurobi), dtype=bool)
    initial_sol_processed[initial_sol_for_gurobi] = True

    bundle_costs = np.array([x[1] for x in bundle_vectors_for_gurobi])
    bundle_riders = np.array([RIDER_DICT[x[0]] for x in bundle_vectors_for_gurobi])
    riders_available_processed = {RIDER_DICT[k]: v for k, v in riders_available.items()}

    sc_sol, obj_val, _ = solve_sc(initial_sol_processed, constraint_helper_for_gurobi, riders_available_processed,
                                  bundle_costs, bundle_riders, timelimit, verbose)
    best_sol = sc_sol_to_answer(sc_sol, bundle_vectors_for_gurobi, dist_mat, riders)

    # prepare for the next ALNS iteration (same as HighsSP.solveSP)
    sol_edge = np.zeros((2 * K, 2 * K), dtype=int)
    sol_rider = np.zeros(K, dtype=int)
    init_bundle = bundle_vectors_for_gurobi
    for rider_type, source, dest in best_sol:
        n = len(source)
        for i in range(n - 1):
            sol_edge[source[i], source[i + 1]] = 1
        sol_edge[source[n - 1], dest[0] + K] = 1
        for i in range(n - 1):
            sol_edge[dest[i] + K, dest[i + 1] + K] = 1
        for x in source:
            sol_rider[x] = RIDER_DICT[rider_type]

    return best_sol, sol_edge, sol_rider, init_bundle, obj_val
