import math

import numpy as np
import pytest

from scsp import Model
from scsp.bench import metrics
from scsp.gen import generate_dms, generate_sp


def test_trace_metrics():
    trace = [[0.5, 20.0, None, "sol"], [2.0, 12.0, 8.0, "sol"], [4.0, 10.0, 9.0, "sol"], [6.0, 10.0, 10.0, "end"]]
    assert metrics.obj_at(trace, 0.1) is None
    assert metrics.obj_at(trace, 0.5) == 20.0 and metrics.obj_at(trace, 3.9) == 12.0 and metrics.obj_at(trace, 60) == 10.0
    assert metrics.bound_at(trace, 3) == 8.0 and metrics.bound_at(trace, 1) is None
    assert metrics.primal_gap(None, 10) == 1.0 and metrics.primal_gap(10, 10) == 0.0
    assert metrics.primal_gap(20, 10) == 0.5 and metrics.primal_gap(-1, 10) == 1.0
    # 1.0 on [0, 0.5), 0.5 on [0.5, 2), 1/6 on [2, 4), 0 afterwards
    assert metrics.primal_integral(trace, 10.0, 6.0) == pytest.approx(0.5 + 0.75 + 2 / 6)
    assert metrics.primal_integral(trace, 10.0, 1.0) == pytest.approx(0.5 + 0.25)
    assert metrics.primal_integral([], 10.0, 5.0) == 5.0
    assert metrics.gap_pct(11.0, 10.0) == pytest.approx(10.0) and metrics.gap_pct(None, 10.0) is None
    assert metrics.shifted_geomean([0.0, 30.0], 10.0) == pytest.approx(math.sqrt(10 * 40) - 10)


def test_per_run_rows_prefers_exact_runs_then_traces():
    def run(tl, status, obj, runtime, trace):
        return {"config": "highs", "instance": "i", "seed": 1, "threads": 4, "warm_start": False, "time_limit": tl,
                "status": status, "obj": obj, "bound": 9.0, "gap": 0.1, "runtime": runtime, "trace": trace,
                "n_cols": 3, "n_cons": 2, "verified": True}

    long = run(60, "time_limit", 10.0, 60.0, [[1.0, 30.0, None, "sol"], [8.0, 12.0, 8.0, "sol"], [40.0, 10.0, 9.0, "sol"]])
    short = run(10, "time_limit", 13.0, 10.0, [[1.0, 30.0, None, "sol"], [9.0, 13.0, 8.0, "sol"]])
    rows = metrics.per_run_rows([long, short], {"i": {"bks": "10"}}, {"i": "s"})
    (r,) = rows
    assert (r["obj_5"], r["src_5"]) == (30.0, "trace")
    assert (r["obj_10"], r["src_10"]) == (13.0, "run")
    assert (r["obj_30"], r["src_30"]) == (12.0, "trace")
    assert (r["obj_60"], r["src_60"]) == (10.0, "run")
    assert r["gap_10"] == pytest.approx(30.0) and r["status"] == "time_limit" and r["proof_time"] is None


@pytest.mark.parametrize("solver", ["highs", "gurobi"])
def test_runners_solve_small_models(solver):
    pytest.importorskip("highspy" if solver == "highs" else "gurobipy")
    from scsp.bench import runners

    run = runners.run_highs if solver == "highs" else runners.run_gurobi
    tiny = Model.from_rows("tiny", [2, 3, 4, 1], [[0, 1], [1, 2], [0, 2, 3]], "G")
    r = run(tiny, 10.0, threads=2, seed=1)
    assert (r.status, r.obj, r.verified, r.sol) == ("optimal", 4.0, True, [1, 3])
    assert r.trace[-1][3] == "end" and r.trace[-1][1] == 4.0

    dms = generate_dms(30, 200, seed=5)
    r = run(dms, 20.0, threads=2, seed=1, warm_start=dms.meta["warm_start"])
    assert r.status == "optimal" and r.verified and r.obj <= dms.meta["warm_start_obj"] + 1e-9
    assert any(ev[3] == "sol" for ev in r.trace)

    sp = generate_sp(30, 150, seed=2)
    r = run(sp, 20.0, threads=2, seed=1)
    x = np.zeros(sp.n_cols)
    x[r.sol] = 1
    assert r.status == "optimal" and r.verified and sp.is_feasible(x)


def test_gurobi_size_limit_is_respected():
    pytest.importorskip("gurobipy")
    from scsp.bench import runners

    big = Model.from_rows("big", np.ones(2001), [list(range(2001))], "G")
    r = runners.run_gurobi(big, 1.0)
    assert r.status == "skipped" and "size-limited" in r.note
