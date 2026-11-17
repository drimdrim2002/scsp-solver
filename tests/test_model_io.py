import gzip
from pathlib import Path

import numpy as np
import pytest

from scsp import Model
from scsp.io import NotASetModel, read_mps_model, read_scp, read_scp_rail, read_spp

DATA = Path(__file__).parent / "data"


def brute_force(m: Model):
    best = None
    for mask in range(1 << m.n_cols):
        x = np.array([(mask >> j) & 1 for j in range(m.n_cols)], dtype=float)
        if m.is_feasible(x):
            obj = m.objective(x)
            if best is None or obj < best:
                best = obj
    return best


def test_model_from_rows_and_checks():
    m = Model.from_rows("t", [1, 2, 3], [[1, 0], [1, 2]], "G", side=[([0, 2], [1, 1], "L", 1)])
    assert (m.n_rows, m.n_cols, m.nnz, m.n_side, m.n_cons) == (2, 3, 4, 1, 3)
    assert m.kind == "SC+side"
    assert m.row(0).tolist() == [0, 1]
    assert m.is_feasible(np.array([0, 1, 0]))
    assert m.violations(np.array([1, 0, 1])) == {"set_rows": 0, "side_rows": 1, "non_binary": 0}
    assert m.violations(np.array([1, 0, 0]))["set_rows"] == 1
    assert m.violations(np.array([0.5, 1, 0]))["non_binary"] == 1
    assert brute_force(m) == 2.0
    col_ptr, row_idx = m.csc()
    assert col_ptr.tolist() == [0, 1, 3, 4] and row_idx.tolist() == [0, 0, 1, 1]
    a, lo, up = m.to_scipy()
    assert a.shape == (3, 3) and lo.tolist() == [1, 1, -np.inf] and up.tolist() == [np.inf, np.inf, 1]


def test_model_rejects_bad_input():
    with pytest.raises(ValueError):
        Model.from_rows("t", [1, 2], [[0, 5]], "G")
    with pytest.raises(ValueError):
        Model.from_rows("t", [1, 2], [[0, 0]], "G")
    with pytest.raises(ValueError):
        Model.from_rows("t", [1, 2], [[0, 1]], "X")


def test_npz_roundtrip(tmp_path):
    m = Model.from_rows("t", [1.5, 2, 3], [[0, 1], [1, 2]], ["E", "L"], side=[([0], [2.0], "G", 1.0)],
                        obj_offset=7.0, meta={"warm_start": [1]})
    m.save_npz(tmp_path / "t.npz")
    r = Model.load_npz(tmp_path / "t.npz")
    for f in ("cost", "row_ptr", "col_idx", "row_sense", "side_ptr", "side_idx", "side_val", "side_sense", "side_rhs"):
        assert np.array_equal(getattr(m, f), getattr(r, f)), f
    assert (r.name, r.obj_offset, r.meta) == ("t", 7.0, {"warm_start": [1]})


def test_read_scp_row_layout():
    m = read_scp(DATA / "tiny_scp.txt")
    assert (m.name, m.kind, m.n_rows, m.n_cols) == ("tiny_scp", "SC", 3, 4)
    assert m.cost.tolist() == [2, 3, 4, 1]
    assert [m.row(i).tolist() for i in range(3)] == [[0, 1], [1, 2], [0, 2, 3]]
    assert brute_force(m) == 4.0                      # columns 2 and 4


def test_read_scp_rail_and_spp_layout(tmp_path):
    rail = read_scp_rail(DATA / "tiny_rail.txt")
    assert (rail.kind, rail.n_rows, rail.n_cols) == ("SC", 3, 4)
    assert rail.cost.tolist() == [2, 3, 4, 1]
    assert [rail.row(i).tolist() for i in range(3)] == [[0, 1], [1, 2], [0, 2, 3]]
    spp = read_spp(DATA / "tiny_rail.txt")
    assert spp.kind == "SP" and np.array_equal(spp.col_idx, rail.col_idx)
    assert brute_force(spp) == 4.0                    # columns 2 and 4 partition the rows
    gz = tmp_path / "tiny_rail.gz"
    gz.write_bytes(gzip.compress((DATA / "tiny_rail.txt").read_bytes()))
    assert np.array_equal(read_spp(gz).col_idx, spp.col_idx)


def test_orlib_reader_rejects_truncated_file(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("3 4\n2 3 4 1\n2 1 2\n2 2 3\n3 1 3\n")
    with pytest.raises(ValueError):
        read_scp(p)
    p.write_text("3 4\n2 3 4 1\n2 1 2\n2 2 3\n3 1 3 9\n")
    with pytest.raises(ValueError):
        read_scp(p)


def test_read_mps():
    m = read_mps_model(DATA / "tiny.mps")
    assert (m.name, m.n_cols, m.n_rows, m.n_side) == ("tiny", 3, 3, 1)
    assert m.row_sense.tolist() == ["E", "G", "L"]
    assert m.cost.tolist() == [2, 3, 4] and m.obj_offset == 5.0
    assert [m.row(i).tolist() for i in range(3)] == [[0, 1], [0, 2], [1, 2]]
    assert m.side_idx.tolist() == [0, 1] and m.side_val.tolist() == [2, 3]
    assert m.side_sense.tolist() == ["L"] and m.side_rhs.tolist() == [4]
    assert brute_force(m) == 7.0                      # x1 alone: 2 + offset 5


def test_mps_rejects_non_binary(tmp_path):
    text = (DATA / "tiny.mps").read_text().replace(" UP bnd       x1                   1", " UP bnd       x1                   3")
    p = tmp_path / "general.mps"
    p.write_text(text)
    with pytest.raises(NotASetModel):
        read_mps_model(p)
