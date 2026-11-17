import numpy as np

from scsp.gen import generate_dms, generate_sp
from scsp.gen.suite import model_digest


def _warm(m):
    x = np.zeros(m.n_cols)
    x[m.meta["warm_start"]] = 1
    return x


def test_dms_shape_and_feasibility():
    for tight in (False, True):
        m = generate_dms(40, 300, seed=3, tight=tight)
        assert (m.n_rows, m.n_cols, m.n_side) == (40, 300, 3)
        assert m.kind == "SC+side" and set(m.side_sense.tolist()) == {"L"}
        # every column has exactly one rider type
        assert np.array_equal(np.sort(m.side_idx), np.arange(300)) and np.all(m.side_val == 1)
        assert m.side_rhs.tolist() == m.meta["avail"]
        x = _warm(m)
        assert m.is_feasible(x)
        assert abs(m.objective(x) - m.meta["warm_start_obj"]) < 1e-9
        assert m.stats()["col_len_max"] <= 5 and m.stats()["row_len_min"] >= 1


def test_dms_is_deterministic():
    a, b = generate_dms(30, 200, seed=5), generate_dms(30, 200, seed=5)
    assert model_digest(a) == model_digest(b)
    assert model_digest(a) != model_digest(generate_dms(30, 200, seed=6))


def test_sp_shape_and_feasibility():
    m = generate_sp(60, 400, seed=2)
    assert (m.n_rows, m.n_cols, m.n_side, m.kind) == (60, 400, 0, "SP")
    assert m.is_feasible(_warm(m))
    assert model_digest(m) == model_digest(generate_sp(60, 400, seed=2))
