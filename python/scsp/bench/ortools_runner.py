"""Google OR-Tools as a reference solver (black-box use only, docs/design.md section 2).

Three configs, kept apart from `runners.CONFIGS` so the shared harness files stay
untouched; this module has its own worker and run loop with the same JSONL format
as `scsp.bench.run` / `scsp.bench.worker`:

    python -m scsp.bench.ortools_runner --configs ortools-cpsat --sets scp4 dms --time-limits 60 --seeds 1 2 3
    python -m scsp.bench.ortools_runner --configs ortools-cpsat --sets dms --time-limits 30 --seeds 1 2 3 --warm-start
    python -m scsp.bench.ortools_runner --worker --config ortools-cpsat --instance scp41 --time-limit 10 --seed 1

  * `ortools-cpsat`: CP-SAT. Cover rows are `add_bool_or`, partition rows
    `add_exactly_one`, packing rows `add_at_most_one`, side rows linear. CP-SAT
    needs integers, so costs (and side rows) are multiplied by the smallest 10^k
    (k <= 6) that makes them integral; the run fails rather than round. "sol"
    events come from the solution callback, "bound" samples from
    `best_bound_callback`. A warm start becomes a full solution hint.
  * `ortools-scip`: SCIP through MathOpt (`SolverType.GSCIP`), always sequential
    (one thread; the `threads` field keeps the harness slot, 4). SCIP is a
    sequential solver; its multi-thread mode in OR-Tools is the concurrent race
    of independent solvers, and in that mode SCIP prints no progress lines until
    the end, so no trace could be read. The trace is read from the SCIP log
    stream (incumbent and dual bound columns, timestamped on arrival). A warm
    start becomes a MathOpt solution hint; when SCIP reports it transformed the
    hint, the hint's objective is recorded as a "sol" event at that moment.
  * `ortools-setcover`: the `ortools.set_cover` local search (greedy start, then
    steepest descent and guided local search in chunks until the time limit).
    Single thread, no bound, pure set covering only (no side rows).

Trace times are wall seconds from just before the solve call; building the
solver model is `build_time`. Returned solutions are re-checked with `Model`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import threading
import time

import numpy as np

from ..model import COVER, PACK, PARTITION, Model
from .catalog import discover, find
from .run import load_runs, results_dir, run_key
from .runners import RunResult, _check_solution, _finite, _Trace, _warm_vector, rel_gap
from .warm import stored


def _ortools_version() -> str:
    import ortools
    return ortools.__version__


def decimal_scale(values: np.ndarray, max_k: int = 6) -> int:
    """Smallest k <= max_k with values * 10^k integral (to 1e-6), else ValueError."""
    v = np.asarray(values, dtype=np.float64)
    for k in range(max_k + 1):
        s = v * 10.0 ** k
        if np.all(np.abs(s - np.round(s)) <= 1e-6) and np.all(np.abs(s) < 2 ** 52):
            return k
    raise ValueError(f"values are not integral after scaling by 10^{max_k}")


class _LockedTrace(_Trace):
    """`_Trace` whose updates may come from solver threads."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()

    def solution(self, obj, bound) -> None:
        with self.lock:
            super().solution(obj, bound)

    def sample_bound(self, bound) -> None:
        with self.lock:
            super().sample_bound(bound)


def _result(model: Model, solver: str, config: str, time_limit, threads, seed, warm_start, options) -> RunResult:
    return RunResult(solver=solver, config=config, solver_version=_ortools_version(), instance=model.name,
                     n_cols=model.n_cols, n_cons=model.n_cons, time_limit=time_limit, threads=threads, seed=seed,
                     warm_start=warm_start is not None, options=dict(options or {}))


# ------------------------------------------------------------------- CP-SAT
def run_cpsat(model: Model, time_limit: float, threads: int = 4, seed: int = 0, warm_start=None,
              options: dict | None = None, config: str = "ortools-cpsat") -> RunResult:
    from ortools.sat.python import cp_model

    res = _result(model, "ortools", config, time_limit, threads, seed, warm_start, options)
    t_build = time.perf_counter()
    k = decimal_scale(model.cost)
    scale = 10 ** k
    icost = np.round(model.cost * scale).astype(np.int64)
    m = cp_model.CpModel()
    x = [m.new_bool_var(f"x{j}") for j in range(model.n_cols)]
    add = {COVER: m.add_bool_or, PARTITION: m.add_exactly_one, PACK: m.add_at_most_one}
    rp, ci = model.row_ptr, model.col_idx
    for i in range(model.n_rows):
        add[model.row_sense[i]]([x[j] for j in ci[rp[i]:rp[i + 1]]])
    for i in range(model.n_side):
        lo, hi = model.side_ptr[i], model.side_ptr[i + 1]
        vals, rhs = model.side_val[lo:hi], model.side_rhs[i]
        ks = decimal_scale(np.append(vals, rhs))
        iv = np.round(vals * 10 ** ks).astype(np.int64)
        irhs = int(round(rhs * 10 ** ks))
        expr = cp_model.LinearExpr.weighted_sum([x[j] for j in model.side_idx[lo:hi]], iv.tolist())
        sense = model.side_sense[i]
        lb = irhs if sense in (COVER, PARTITION) else cp_model.INT_MIN
        ub = irhs if sense in (PACK, PARTITION) else cp_model.INT_MAX
        m.add_linear_constraint(expr, lb, ub)
    m.minimize(cp_model.LinearExpr.weighted_sum(x, icost.tolist()))
    x0 = _warm_vector(model, warm_start)
    if x0 is not None:
        for j, v in enumerate(x0):
            m.add_hint(x[j], bool(v > 0.5))

    solver = cp_model.CpSolver()
    p = solver.parameters
    p.num_workers = int(threads)
    p.random_seed = int(seed)
    p.max_time_in_seconds = float(time_limit)
    p.relative_gap_limit = 0.0
    p.absolute_gap_limit = 0.0
    p.log_search_progress = False
    for key, v in (options or {}).items():
        setattr(p, key, v)
    res.options = {**(options or {}), "cost_scale": scale}
    res.build_time = time.perf_counter() - t_build

    off = model.obj_offset
    trace = _LockedTrace()

    class _Cb(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            trace.solution(self.ObjectiveValue() / scale + off, self.BestObjectiveBound() / scale + off)

    solver.best_bound_callback = lambda b: trace.sample_bound(b / scale + off)
    trace.start()
    st = solver.solve(m, _Cb())
    res.runtime = trace.now()

    S = cp_model
    res.status = {S.OPTIMAL: "optimal", S.FEASIBLE: "time_limit", S.UNKNOWN: "time_limit",
                  S.INFEASIBLE: "infeasible"}.get(st, "cpsat_" + solver.status_name(st).lower())
    res.nodes = float(solver.num_branches)
    if st in (S.OPTIMAL, S.FEASIBLE):
        res.obj = int(round(solver.objective_value)) / scale + off
        res.bound = int(round(solver.best_objective_bound)) / scale + off
        sol = np.asarray(list(solver.response_proto.solution), dtype=np.float64)[: model.n_cols]
        _check_solution(res, model, sol)
    elif st == S.UNKNOWN:
        res.bound = _finite(solver.best_objective_bound / scale + off)
    res.gap = rel_gap(res.obj, res.bound)
    res.trace = trace.finish(res.runtime, res.obj, res.bound)
    return res


# --------------------------------------------------------- SCIP via MathOpt
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_HINT = re.compile(r"transformed (\d+)/(\d+) original solutions")
_FOUND = re.compile(r"feasible solution found by .* after ([\d.]+) seconds, objective value ([-+\d.eE]+)")


def _scip_number(s: str):
    s = s.strip()
    if s.startswith("--") or s in ("", "-", "inf", "+inf", "-inf", "Inf"):
        return None
    return float(s) if _NUM.match(s) else None


class _ScipLog:
    """Reads incumbent / dual bound columns from SCIP's display table as lines arrive."""

    def __init__(self, trace: _Trace, warm_obj: float | None = None) -> None:
        self.trace = trace
        self.warm_obj = warm_obj
        self.cols: dict | None = None
        self.lines = 0

    def __call__(self, lines) -> None:
        for line in lines:
            self.lines += 1
            if "|" in line and "dualbound" in line and "primalbound" in line:
                names = [c.strip() for c in line.split("|")]
                self.cols = {"dual": names.index("dualbound"), "primal": names.index("primalbound"), "n": len(names)}
                continue
            m = _HINT.search(line)
            if m and self.warm_obj is not None and int(m.group(1)) > 0:
                self.trace.solution(self.warm_obj, None)
                continue
            m = _FOUND.search(line)
            if m:
                self.trace.solution(float(m.group(2)), None)
                continue
            if self.cols and "|" in line:
                f = line.split("|")
                if len(f) != self.cols["n"]:
                    continue
                primal, dual = _scip_number(f[self.cols["primal"]]), _scip_number(f[self.cols["dual"]])
                if primal is not None:
                    self.trace.solution(primal, dual)
                if dual is not None:
                    self.trace.sample_bound(dual)


def _mathopt_model(model: Model):
    from ortools.math_opt import model_pb2
    from ortools.math_opt.python import mathopt

    n = model.n_cols
    pr = model_pb2.ModelProto(name=model.name)
    v = pr.variables
    v.ids.extend(range(n))
    v.lower_bounds.extend(np.zeros(n).tolist())
    v.upper_bounds.extend(np.ones(n).tolist())
    v.integers.extend([True] * n)
    pr.objective.maximize = False
    pr.objective.offset = float(model.obj_offset)
    nz = np.flatnonzero(model.cost)
    pr.objective.linear_coefficients.ids.extend(nz.tolist())
    pr.objective.linear_coefficients.values.extend(model.cost[nz].tolist())
    a, lower, upper = model.to_scipy()                   # CSR, rows sorted, set rows then side rows
    a.sort_indices()
    c = pr.linear_constraints
    c.ids.extend(range(a.shape[0]))
    c.lower_bounds.extend(lower.tolist())
    c.upper_bounds.extend(upper.tolist())
    mat = pr.linear_constraint_matrix
    mat.row_ids.extend(np.repeat(np.arange(a.shape[0]), np.diff(a.indptr)).tolist())
    mat.column_ids.extend(a.indices.tolist())
    mat.coefficients.extend(a.data.tolist())
    return mathopt.Model.from_model_proto(pr)


def run_scip(model: Model, time_limit: float, threads: int = 4, seed: int = 0, warm_start=None,
             options: dict | None = None, config: str = "ortools-scip") -> RunResult:
    from ortools.math_opt.python import mathopt

    res = _result(model, "ortools", config, time_limit, threads, seed, warm_start, options)
    t_build = time.perf_counter()
    om = _mathopt_model(model)
    params = mathopt.SolveParameters(time_limit=datetime.timedelta(seconds=float(time_limit)), threads=1,
                                     random_seed=int(seed), relative_gap_tolerance=0.0, absolute_gap_tolerance=0.0,
                                     enable_output=True)
    model_params = None
    x0 = _warm_vector(model, warm_start)
    if x0 is not None:
        hint = {om.get_variable(j): float(x0[j]) for j in range(model.n_cols)}
        model_params = mathopt.ModelSolveParameters(solution_hints=[mathopt.SolutionHint(variable_values=hint)])
    solver = mathopt.IncrementalSolver(om, mathopt.SolverType.GSCIP)   # builds the SCIP problem
    res.build_time = time.perf_counter() - t_build

    trace = _LockedTrace()
    log = _ScipLog(trace, None if x0 is None else model.objective(x0))
    trace.start()
    r = solver.solve(params=params, model_params=model_params, msg_cb=log)
    res.runtime = trace.now()

    T = mathopt.TerminationReason
    reason = r.termination.reason
    if reason == T.OPTIMAL:
        res.status = "optimal"
    elif reason in (T.FEASIBLE, T.NO_SOLUTION_FOUND):
        res.status = "time_limit"
    elif reason == T.INFEASIBLE:
        res.status = "infeasible"
    else:
        res.status = "scip_" + reason.name.lower()
    res.nodes = float(r.solve_stats.node_count)
    res.bound = _finite(r.termination.objective_bounds.dual_bound)
    if r.has_primal_feasible_solution():
        res.obj = float(r.objective_value())
        proto = r.to_proto().solutions[0].primal_solution.variable_values
        sol = np.zeros(model.n_cols)
        sol[np.asarray(proto.ids, dtype=np.int64)] = np.asarray(proto.values)
        _check_solution(res, model, sol)
    if x0 is not None and (res.obj is None or res.obj > model.objective(x0) + 1e-6 * max(1.0, abs(res.obj))):
        # SCIP did not keep the hint: drop the "sol" event made from the transform message
        w = model.objective(x0)
        trace.events = [e for e in trace.events if not (e[3] == "sol" and e[1] == w)]
        res.note = (res.note + " solution hint not kept by SCIP").strip()
    res.gap = rel_gap(res.obj, res.bound)
    res.options = {**(options or {}), "scip_threads": 1, "log_lines": log.lines}
    res.trace = trace.finish(res.runtime, res.obj, res.bound)
    return res


# ------------------------------------------------------------- set_cover
def run_setcover(model: Model, time_limit: float, threads: int = 1, seed: int = 0, warm_start=None,
                 options: dict | None = None, config: str = "ortools-setcover") -> RunResult:
    from ortools.set_cover.python import set_cover as sc

    res = _result(model, "ortools", config, time_limit, 1, seed, warm_start, options)
    if model.n_side or np.any(model.row_sense != COVER):
        res.status = "skipped"
        res.note = "set_cover handles pure set covering only (no side rows, all rows >= 1)"
        return res
    chunk = int((options or {}).get("chunk", 1000))
    t_build = time.perf_counter()
    scm = sc.SetCoverModel()
    a, _, _ = model.to_scipy()
    at = a.T.tocsr()                                     # column -> rows
    for j in range(model.n_cols):
        scm.add_empty_subset(float(model.cost[j]))
        for i in at.indices[at.indptr[j]:at.indptr[j + 1]]:
            scm.add_element_to_last_subset(int(i))
    inv = sc.SetCoverInvariant(scm)
    x0 = _warm_vector(model, warm_start)
    res.build_time = time.perf_counter() - t_build

    def chosen() -> np.ndarray:
        return np.asarray(inv.is_selected(), dtype=np.float64)

    trace = _Trace()
    trace.start()
    if x0 is not None:
        inv.load_solution((x0 > 0.5).tolist())
    else:
        sc.GreedySolutionGenerator(inv).next_solution()
    best_x, best = chosen(), inv.cost()
    trace.solution(best, None)
    steep = sc.SteepestSearch(inv)
    steep.set_max_iterations(chunk)
    gls = sc.GuidedLocalSearch(inv)
    gls.initialize()
    gls.set_max_iterations(chunk)
    rounds = 0
    while trace.now() < time_limit:
        rounds += 1
        steep.next_solution()
        if inv.cost() < best - 1e-9:
            best_x, best = chosen(), inv.cost()
            trace.solution(best, None)
        if trace.now() >= time_limit:
            break
        gls.next_solution()
        if inv.cost() < best - 1e-9 and inv.num_uncovered_elements() == 0:
            best_x, best = chosen(), inv.cost()
            trace.solution(best, None)
    res.runtime = trace.now()
    res.status = "time_limit"
    res.obj = float(best) + model.obj_offset
    _check_solution(res, model, best_x)
    res.options = {**(options or {}), "rounds": rounds}
    res.trace = trace.finish(res.runtime, res.obj, None)
    return res


# ------------------------------------------------------------------ configs
CONFIGS = {
    "ortools-cpsat": (run_cpsat, {}),
    # one documented parameter: with 4 workers the CP-SAT portfolio is [core, default_lp, no_lp] and none of
    # them linearizes the clauses, so there is no LP bound; level 2 (what the max_lp worker of larger
    # portfolios uses) puts the set rows into the LP of the lp workers
    "ortools-cpsat-lp2": (run_cpsat, {"linearization_level": 2}),
    # CP-SAT's documented minimum portfolio (run with --threads 8: more cores than the other solvers get)
    "ortools-cpsat8": (run_cpsat, {}),
    "ortools-scip": (run_scip, {}),
    "ortools-setcover": (run_setcover, {}),
}


def run_config(config: str, model: Model, time_limit: float, threads: int = 4, seed: int = 0,
               warm_start=None) -> RunResult:
    runner, options = CONFIGS[config]
    return runner(model, time_limit, threads=threads, seed=seed, warm_start=warm_start,
                  options=options, config=config)


# ------------------------------------------------------------------ worker
def worker(args) -> None:
    model = find(args.instance).load()
    warm = source = None
    if args.warm_start:
        warm, source = stored(args.instance), "warm_starts.json"
        if warm is None:
            warm, source = model.meta.get("warm_start"), "meta"
        if warm is None:
            raise SystemExit(f"{args.instance}: no warm start stored for the instance")
    res = run_config(args.config, model, args.time_limit, threads=args.threads, seed=args.seed, warm_start=warm)
    rec = res.to_dict()
    if warm is not None:
        x0 = np.zeros(model.n_cols)
        x0[np.asarray(warm, dtype=np.int64)] = 1.0
        rec["warm_obj"], rec["warm_source"] = model.objective(x0), source
    sys.stdout.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------- run loop
def run_all(args) -> None:
    catalog = discover()
    chosen = [i for i in catalog if i.set in args.sets or i.name in args.instances]
    if not args.sets and not args.instances:
        chosen = catalog
    missing = set(args.instances) - {i.name for i in catalog}
    if missing:
        raise SystemExit(f"unknown instances: {sorted(missing)}")
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = done = 0
    t_start = time.time()
    for config in args.configs:
        out = out_dir / f"{config}{args.tag}.jsonl"
        seen = {run_key(r) for r in load_runs(out)}
        for tl in sorted(args.time_limits, reverse=True):
            for inst in chosen:
                for seed in args.seeds:
                    key = (config, inst.name, float(tl), seed, args.threads, args.warm_start)
                    if key in seen:
                        continue
                    if args.budget_min and time.time() - t_start + tl + 30 > 60 * args.budget_min:
                        print(f"budget of {args.budget_min:g} min reached, stopping", flush=True)
                        return
                    planned += 1
                    if args.dry_run:
                        continue
                    cmd = [sys.executable, "-I", "-m", "scsp.bench.ortools_runner", "--worker", "--config", config,
                           "--instance", inst.name, "--time-limit", str(tl), "--threads", str(args.threads),
                           "--seed", str(seed)]
                    if args.warm_start:
                        cmd.append("--warm-start")
                    base = {"config": config, "solver": "ortools", "instance": inst.name, "time_limit": tl,
                            "threads": args.threads, "seed": seed, "warm_start": args.warm_start, "obj": None,
                            "bound": None, "gap": None, "trace": []}
                    t0 = time.time()
                    try:
                        p = subprocess.run(cmd, capture_output=True, text=True, timeout=tl * 2 + 120)
                        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                        if p.returncode != 0 or not lines:
                            rec = {**base, "status": "error", "runtime": time.time() - t0,
                                   "note": (p.stderr or p.stdout)[-400:]}
                        else:
                            rec = json.loads(lines[-1])
                    except subprocess.TimeoutExpired:
                        rec = {**base, "status": "killed", "runtime": time.time() - t0,
                               "note": "worker exceeded 2*tl+120 s"}
                    rec["wall"] = round(time.time() - t0, 3)
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    seen.add(key)
                    done += 1
                    obj = "-" if rec.get("obj") is None else f"{rec['obj']:.6g}"
                    bnd = "-" if rec.get("bound") is None else f"{rec['bound']:.6g}"
                    print(f"[{time.time() - t_start:7.0f}s] {config:16s} {inst.name:26s} tl={tl:<4g} s={seed} "
                          f"{rec['status']:10s} obj={obj:>12s} bound={bnd:>12s} t={rec['runtime']:.2f} "
                          f"build={rec.get('build_time', 0):.1f} ok={rec.get('verified')} {rec.get('note', '')[:80]}",
                          flush=True)
    print(f"{'planned' if args.dry_run else 'finished'}: {planned if args.dry_run else done} runs "
          f"in {time.time() - t_start:.0f}s", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="run one solve in this process and print it as JSON")
    ap.add_argument("--config", choices=sorted(CONFIGS))
    ap.add_argument("--instance")
    ap.add_argument("--time-limit", type=float)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--configs", nargs="+", default=["ortools-cpsat"], choices=sorted(CONFIGS))
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--instances", nargs="*", default=[])
    ap.add_argument("--time-limits", nargs="+", type=float, default=[60])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget-min", type=float, default=0, help="start no run that could end after this many minutes")
    args = ap.parse_args(argv)
    if args.worker:
        if not (args.config and args.instance and args.time_limit):
            ap.error("--worker needs --config, --instance and --time-limit")
        worker(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
