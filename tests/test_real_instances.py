"""Cross-checks on the downloaded instances; skipped when the files are absent."""
import numpy as np
import pytest

from scsp.bench.catalog import discover

try:
    CATALOG = {i.name: i for i in discover()}
except FileNotFoundError:
    CATALOG = {}


def _load(name):
    if name not in CATALOG:
        pytest.skip(f"{name} not downloaded (scripts/download_instances.sh)")
    return CATALOG[name].load()


def _column_signature(m):
    """Multiset of columns as (cost, rows) so that column order does not matter."""
    col_ptr, row_idx = m.csc()
    return sorted((float(m.cost[j]), tuple(row_idx[col_ptr[j]:col_ptr[j + 1]].tolist())) for j in range(m.n_cols))


@pytest.mark.parametrize("mps_name, orlib_name", [("air04", "sppaa01"), ("air05", "sppaa04"), ("nw04", "sppnw04")])
def test_mps_and_orlib_readers_agree(mps_name, orlib_name):
    a, b = _load(mps_name), _load(orlib_name)
    assert (a.n_rows, a.n_cols, a.nnz, a.kind) == (b.n_rows, b.n_cols, b.nnz, "SP")
    assert a.n_side == 0 and a.obj_offset == 0.0
    if a.n_cols <= 10000:                              # row order may differ between the two files
        assert sorted(np.diff(a.row_ptr).tolist()) == sorted(np.diff(b.row_ptr).tolist())
        assert sorted(a.cost.tolist()) == sorted(b.cost.tolist())


@pytest.mark.parametrize("name, rows, cols, nnz", [
    ("scp41", 200, 1000, 4009), ("scpnrh5", 1000, 10000, 499179), ("rail507", 507, 63009, 409349),
    ("sppnw41", 17, 197, 740), ("sppus01", 145, 1053137, 13636541),
])
def test_sizes_of_known_instances(name, rows, cols, nnz):
    m = _load(name)
    assert (m.n_rows, m.n_cols, m.nnz) == (rows, cols, nnz)
    assert m.stats()["row_len_min"] >= 1


@pytest.mark.parametrize("name", ["air05", "scp41", "sppnw41"])
def test_lp_relaxation_matches_highs_reading_the_same_data(name, tmp_path):
    """Our Model, passed to HiGHS as an LP, must give the LP value HiGHS gets
    from its own reader (MPS) or, for OR-Library files, a value below the optimum."""
    highspy = pytest.importorskip("highspy")
    m = _load(name)
    a, lower, upper = m.to_scipy()
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = m.n_cols, m.n_cons
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = m.cost, np.zeros(m.n_cols), np.ones(m.n_cols)
    lp.row_lower_ = np.where(np.isfinite(lower), lower, -inf)
    lp.row_upper_ = np.where(np.isfinite(upper), upper, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = a.indptr, a.indices, a.data
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    ours = h.getInfo().objective_function_value
    if CATALOG[name].fmt == "mps":
        import gzip
        path = tmp_path / f"{name}.mps"
        path.write_bytes(gzip.decompress(CATALOG[name].path.read_bytes()))
        g = highspy.Highs()
        g.setOptionValue("output_flag", False)
        g.readModel(str(path))
        lp2 = g.getLp()
        lp2.integrality_ = []
        g.passModel(lp2)
        g.run()
        assert ours == pytest.approx(g.getInfo().objective_function_value, rel=1e-9)
    else:
        known = {"scp41": 429.0, "sppnw41": 11307.0}[name]
        assert 0 < ours <= known + 1e-6
