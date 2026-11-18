"""OR-Tools reference runners (scsp.bench.ortools_runner) agree with HiGHS on small models."""
import numpy as np
import pytest

from scsp import Model
from scsp.gen import generate_dms, generate_sp

pytest.importorskip("ortools")
highspy = pytest.importorskip("highspy")

from scsp.bench import ortools_runner as ort  # noqa: E402
from scsp.bench import runners  # noqa: E402

RUNNERS = {
    # 8 workers: the 4-worker CP-SAT portfolio has no LP-bound worker and does not prove even these small
    # models in 30 s (a finding of docs/ortools_report.md); the tests check the model, not that portfolio
    "cpsat": lambda m, tl, **kw: ort.run_cpsat(m, tl, options={"num_workers": 8}, **kw),
    "scip": ort.run_scip,
}


def dms(n_orders, n_bundles, seed):
    """Generated DMS model with costs rounded to 4 decimals, like the stored benchmark files (CP-SAT needs
    costs that become integral after scaling by 10^k)."""
    m = generate_dms(n_orders, n_bundles, seed=seed)
    m.cost = np.round(m.cost, 4)
    return m


def test_decimal_scale():
    assert ort.decimal_scale(np.array([1.0, 2.0])) == 0
    assert ort.decimal_scale(np.array([53.857, 40.72])) == 3
    with pytest.raises(ValueError):
        ort.decimal_scale(np.array([1 / 3]))


def test_scip_log_parser():
    tr = runners._Trace()
    log = ort._ScipLog(tr)
    log(["feasible solution found by trivial heuristic after 0.0 seconds, objective value 1.559000e+04",
         " time | mem |  dualbound   | primalbound  |  gap   ",
         "$ 0.3s| 331M| 1.129364e+03 | 1.174000e+03 |   3.95%",
         "  0.6s| 382M| 1.131979e+03 | 1.174000e+03 |   3.58%",
         "transformed 1/1 original solutions to the transformed problem space",
         "  0.1s|     1 |     0 |   123 |     - |  20M |   0 | 300 | 400 | 400 |   0 |  0 |   0 |   0 | 1.0e+03 |      --      |    Inf | unknown"])
    sols = [e[1] for e in tr.events if e[3] == "sol"]
    assert sols == [15590.0, 1174.0]
    assert tr.bound == pytest.approx(1131.979)


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_ortools_matches_highs_optimum(name):
    run = RUNNERS[name]
    tiny = Model.from_rows("tiny", [2, 3, 4, 1], [[0, 1], [1, 2], [0, 2, 3]], "G")
    r = run(tiny, 10.0, threads=2, seed=1)
    assert (r.status, r.obj, r.verified, r.sol) == ("optimal", 4.0, True, [1, 3])
    assert r.trace[-1][3] == "end" and r.trace[-1][1] == 4.0

    models = [dms(30, 200, 5),            # covering + `<=` side rows, decimal costs
              generate_sp(30, 150, seed=2),              # partitioning
              dms(60, 600, 3)]
    for m in models:
        ref = runners.run_highs(m, 30.0, threads=2, seed=1)
        r = run(m, 30.0, threads=2, seed=1)
        assert ref.status == "optimal" and r.status == "optimal", (m.name, ref.status, r.status)
        assert r.verified and abs(r.obj - ref.obj) <= 1e-6 * max(1.0, abs(ref.obj)), (m.name, r.obj, ref.obj)
        assert any(ev[3] == "sol" for ev in r.trace)


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_ortools_warm_start(name):
    m = dms(30, 200, 5)
    x0 = np.zeros(m.n_cols)
    x0[m.meta["warm_start"]] = 1.0
    r = RUNNERS[name](m, 20.0, threads=2, seed=1, warm_start=m.meta["warm_start"])
    assert r.warm_start and r.status == "optimal" and r.verified
    assert r.obj <= m.objective(x0) + 1e-9
    first = next(ev for ev in r.trace if ev[3] == "sol")
    assert first[1] <= m.objective(x0) + 1e-6                    # the hint was taken as a solution


def test_setcover_heuristic():
    m = dms(30, 200, 5)
    assert ort.run_setcover(m, 1.0).status == "skipped"           # side rows: not handled
    tiny = Model.from_rows("tiny", [2, 3, 4, 1], [[0, 1], [1, 2], [0, 2, 3]], "G")
    r = ort.run_setcover(tiny, 0.5)
    assert r.verified and r.obj >= 4.0
