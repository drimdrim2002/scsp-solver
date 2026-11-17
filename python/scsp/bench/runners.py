"""Reference-solver runners (black-box use only, docs/design.md section 2).

`run_gurobi` and `run_highs` solve a `Model` and return a `RunResult` holding
the final answer and a trace of (seconds, incumbent objective, dual bound):

  * "sol" events come from the improving-solution callbacks
    (Gurobi `MIPSOL`, HiGHS `kCallbackMipImprovingSolution`);
  * "bound" events are throttled samples of the dual bound from the progress
    callbacks (Gurobi `MIP`, HiGHS `kCallbackMipLogging`);
  * one "end" event carries the final objective and bound.

Trace times are wall-clock seconds measured from just before the solver's
optimize call; building the solver model is timed separately (`build_time`).
The returned solution is re-checked against the `Model`, not trusted.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..model import COVER, PARTITION, Model
from .catalog import GUROBI_MAX_CONS, GUROBI_MAX_VARS, gurobi_ok

BOUND_SAMPLE_SECONDS = 0.25


@dataclass
class RunResult:
    solver: str
    config: str
    solver_version: str
    instance: str
    n_cols: int
    n_cons: int
    time_limit: float
    threads: int
    seed: int
    warm_start: bool
    status: str = "error"            # optimal | time_limit | infeasible | skipped | error | killed
    obj: float | None = None         # solver-reported incumbent objective
    bound: float | None = None       # solver-reported dual bound
    gap: float | None = None         # (obj - bound) / |obj|
    runtime: float = 0.0             # wall seconds inside optimize
    build_time: float = 0.0
    nodes: float | None = None
    verified: bool | None = None     # incumbent feasible for the Model and objective matches
    obj_check: float | None = None   # objective recomputed from the returned 0/1 vector
    sol: list | None = None          # indices of the columns at 1
    trace: list = field(default_factory=list)   # [t, obj | None, bound | None, kind]
    options: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) and abs(v) < 1e99 else None


def rel_gap(obj: float | None, bound: float | None) -> float | None:
    if obj is None or bound is None:
        return None
    if obj == bound:
        return 0.0
    if obj == 0.0:
        return math.inf
    return max(0.0, (obj - bound) / abs(obj))


class _Trace:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list = []
        self.best: float | None = None
        self.bound: float | None = None
        self._last_bound_t = -math.inf

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def solution(self, obj, bound) -> None:
        obj, bound = _finite(obj), _finite(bound)
        if bound is not None:
            self.bound = bound if self.bound is None else max(self.bound, bound)
        if obj is not None and (self.best is None or obj < self.best - 1e-9 * max(1.0, abs(obj))):
            self.best = obj
            self.events.append([round(self.now(), 4), obj, self.bound, "sol"])

    def sample_bound(self, bound) -> None:
        bound = _finite(bound)
        if bound is None:
            return
        t = self.now()
        if self.bound is not None and bound <= self.bound + 1e-9 * max(1.0, abs(bound)):
            return
        self.bound = bound
        if t - self._last_bound_t >= BOUND_SAMPLE_SECONDS:
            self._last_bound_t = t
            self.events.append([round(t, 4), self.best, bound, "bound"])

    def finish(self, runtime: float, obj, bound) -> list:
        self.events.append([round(runtime, 4), _finite(obj), _finite(bound), "end"])
        return self.events


def _check_solution(res: RunResult, model: Model, x) -> None:
    if x is None:
        return
    xb = (np.asarray(x, dtype=np.float64) > 0.5).astype(np.float64)
    res.obj_check = model.objective(xb)
    viol = model.violations(xb)
    close = res.obj is not None and abs(res.obj_check - res.obj) <= 1e-6 * max(1.0, abs(res.obj))
    res.verified = bool(not any(viol.values()) and close)
    if not res.verified:
        res.note = (res.note + f" check failed: violations={viol}, obj={res.obj}, recomputed={res.obj_check}").strip()
    res.sol = np.flatnonzero(xb).tolist()


def _warm_vector(model: Model, warm_start) -> np.ndarray | None:
    if warm_start is None:
        return None
    x0 = np.zeros(model.n_cols)
    x0[np.asarray(warm_start, dtype=np.int64)] = 1.0
    if not model.is_feasible(x0):
        raise ValueError(f"{model.name}: warm start is not feasible")
    return x0


# ------------------------------------------------------------------- Gurobi
def run_gurobi(model: Model, time_limit: float, threads: int = 4, seed: int = 0, warm_start=None,
               options: dict | None = None, config: str = "gurobi") -> RunResult:
    import gurobipy as gp
    from gurobipy import GRB

    options = dict(options or {})
    res = RunResult(
        solver="gurobi", config=config, solver_version=".".join(map(str, gp.gurobi.version())),
        instance=model.name, n_cols=model.n_cols, n_cons=model.n_cons, time_limit=time_limit,
        threads=threads, seed=seed, warm_start=warm_start is not None, options=options,
    )
    if not gurobi_ok(model.n_cols, model.n_cons):
        res.status = "skipped"
        res.note = (f"size-limited Gurobi licence: needs vars <= {GUROBI_MAX_VARS} and "
                    f"constraints <= {GUROBI_MAX_CONS}")
        return res

    t_build = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        m = gp.Model(model.name, env=env)
        x = m.addMVar(model.n_cols, vtype=GRB.BINARY, obj=model.cost, name="x")
        a, _, _ = model.to_scipy()
        sense_map = {COVER: ">", PARTITION: "=", "L": "<"}
        sense = np.array([sense_map[s] for s in np.concatenate([model.row_sense, model.side_sense])])
        rhs = np.concatenate([np.ones(model.n_rows), model.side_rhs])
        m.addMConstr(a, x, sense, rhs)
        m.ObjCon = model.obj_offset
        m.ModelSense = GRB.MINIMIZE
        m.Params.Threads = threads
        m.Params.TimeLimit = time_limit
        m.Params.Seed = seed
        m.Params.MIPGap = 0.0
        for k, v in options.items():
            m.setParam(k, v)
        x0 = _warm_vector(model, warm_start)
        if x0 is not None:
            x.Start = x0
        m.update()
        res.build_time = time.perf_counter() - t_build

        trace = _Trace()

        def callback(cb_model, where):
            if where == GRB.Callback.MIPSOL:
                trace.solution(cb_model.cbGet(GRB.Callback.MIPSOL_OBJ), cb_model.cbGet(GRB.Callback.MIPSOL_OBJBND))
            elif where == GRB.Callback.MIP:
                trace.sample_bound(cb_model.cbGet(GRB.Callback.MIP_OBJBND))

        trace.start()
        try:
            m.optimize(callback)
        except gp.GurobiError as e:
            res.status = "error"
            res.note = f"GurobiError {e.errno}: {e.message}"
            return res
        res.runtime = trace.now()

        status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible",
                  GRB.INF_OR_UNBD: "infeasible", GRB.INTERRUPTED: "time_limit"}.get(m.Status, f"gurobi_status_{m.Status}")
        res.status = status
        res.nodes = float(m.NodeCount)
        res.bound = _finite(m.ObjBound)
        if m.SolCount > 0:
            res.obj = float(m.ObjVal)
            _check_solution(res, model, x.X)
        res.gap = rel_gap(res.obj, res.bound)
        res.trace = trace.finish(res.runtime, res.obj, res.bound)
        return res
    finally:
        env.dispose()


# -------------------------------------------------------------------- HiGHS
def run_highs(model: Model, time_limit: float, threads: int = 4, seed: int = 0, warm_start=None,
              options: dict | None = None, config: str = "highs") -> RunResult:
    import highspy

    options = dict(options or {})
    res = RunResult(
        solver="highs", config=config, solver_version=highspy.Highs().version(),
        instance=model.name, n_cols=model.n_cols, n_cons=model.n_cons, time_limit=time_limit,
        threads=threads, seed=seed, warm_start=warm_start is not None, options=options,
    )
    t_build = time.perf_counter()
    inf = highspy.kHighsInf
    a, lower, upper = model.to_scipy()
    n = model.n_cols
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = model.n_cons
    lp.col_cost_ = model.cost
    lp.col_lower_ = np.zeros(n)
    lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.where(np.isfinite(lower), lower, -inf)
    lp.row_upper_ = np.where(np.isfinite(upper), upper, inf)
    lp.offset_ = model.obj_offset
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = a.indptr
    lp.a_matrix_.index_ = a.indices
    lp.a_matrix_.value_ = a.data

    h = highspy.Highs()
    # logging stays on (but silent) because kCallbackMipLogging only fires with it
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("random_seed", int(seed))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    for k, v in options.items():
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected option {k}={v!r}")
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS passModel failed")
    x0 = _warm_vector(model, warm_start)
    if x0 is not None:
        start = highspy.HighsSolution()
        start.col_value = x0
        start.value_valid = True
        h.setSolution(start)
    res.build_time = time.perf_counter() - t_build

    trace = _Trace()

    def on_solution(event):
        d = event.data_out
        trace.solution(d.objective_function_value, d.mip_dual_bound)

    def on_logging(event):
        trace.sample_bound(event.data_out.mip_dual_bound)

    h.cbMipImprovingSolution.subscribe(on_solution)
    h.cbMipLogging.subscribe(on_logging)
    trace.start()
    h.run()
    res.runtime = trace.now()

    info = h.getInfo()
    ms = h.getModelStatus()
    S = highspy.HighsModelStatus
    res.status = {S.kOptimal: "optimal", S.kTimeLimit: "time_limit", S.kInfeasible: "infeasible",
                  S.kInterrupt: "time_limit"}.get(ms, "highs_" + h.modelStatusToString(ms).lower().replace(" ", "_"))
    res.nodes = float(info.mip_node_count)
    res.bound = _finite(info.mip_dual_bound)
    if info.primal_solution_status == 2:                 # 2 = feasible point available
        res.obj = float(info.objective_function_value)
        _check_solution(res, model, np.asarray(h.getSolution().col_value))
    res.gap = rel_gap(res.obj, res.bound)
    res.trace = trace.finish(res.runtime, res.obj, res.bound)
    return res


# --------------------------------------------------------------------- scsp
def run_scsp(model: Model, time_limit: float, threads: int = 4, seed: int = 0, warm_start=None,
             options: dict | None = None, config: str = "scsp") -> RunResult:
    """Our own solver. Trace times are shifted so that they count from the call of
    the native solver, including the conversion of the model."""
    from .. import _core
    from ..solve import _warm_indices, solve

    options = dict(options or {})
    res = RunResult(
        solver="scsp", config=config, solver_version=_core.version(),
        instance=model.name, n_cols=model.n_cols, n_cons=model.n_cons, time_limit=time_limit,
        threads=threads, seed=seed, warm_start=warm_start is not None, options=options,
    )
    if warm_start is not None:
        _warm_vector(model, warm_start)                  # raises if infeasible
    t0 = time.perf_counter()
    r = solve(model, time_limit=time_limit, threads=threads, seed=seed,
              warm_start=None if warm_start is None else _warm_indices(model, warm_start), verify=False, **options)
    res.runtime = time.perf_counter() - t0
    shift = max(0.0, res.runtime - r.runtime)
    res.status = r.status
    res.obj, res.bound = _finite(r.obj), _finite(r.bound)
    res.options = {**options, "stats": r.stats}
    if r.x is not None:
        _check_solution(res, model, r.x)
    res.gap = rel_gap(res.obj, res.bound)
    events, best = [], None
    for t, obj, bound, kind in r.trace:
        obj, bound = _finite(obj), _finite(bound)
        if kind == "sol":
            best = obj
        events.append([round(t + shift, 4), best if kind == "bound" else obj, bound, kind])
    events.append([round(res.runtime, 4), res.obj, res.bound, "end"])
    res.trace = events
    return res


# ------------------------------------------------------------------ configs
# name -> (runner, extra solver options). Defaults only: Threads, TimeLimit,
# Seed and a zero relative gap are set by the runners; nothing else is tuned.
CONFIGS = {
    "gurobi": (run_gurobi, {}),
    "highs": (run_highs, {}),
    # same heuristic setting as the DMS competition code (HighsSP.py)
    "highs-dms": (run_highs, {"mip_heuristic_effort": 0.3}),
    "scsp": (run_scsp, {}),
    # the P1 heuristic alone (no root LP, no branch and bound); only covering models
    "scsp-p1": (run_scsp, {"bnb": False}),
}


def run_config(config: str, model: Model, time_limit: float, threads: int = 4, seed: int = 0,
               warm_start=None) -> RunResult:
    runner, options = CONFIGS[config]
    return runner(model, time_limit, threads=threads, seed=seed, warm_start=warm_start,
                  options=options, config=config)
