"""OR-Library readers (J.E. Beasley, http://people.brunel.ac.uk/~mastjjb/jeb/info.html).

Three text layouts, all 1-based:

  SCP row-wise  (scp4x .. scpe, scpnre .. scpnrh, scpclr, scpcyc)
      m n / c_1 .. c_n / for each row: k, then the k columns covering it
  SCP column-wise  (rail507 ..)
      m n / for each column: cost, k, then the k rows it covers
  SPP  (sppnw, sppaa, sppus, sppkl)
      m n / for each column: cost, k, then the k rows it covers     (rows are "= 1")

A column listed twice in the same row is kept once; the count is stored in
meta["duplicates_removed"].
"""
from __future__ import annotations

import numpy as np

from ..model import COVER, PARTITION, Model
from ._text import instance_name, numeric_tokens, to_index


def _header(tok: np.ndarray, name: str) -> tuple[int, int]:
    if tok.size < 2:
        raise ValueError(f"{name}: empty file")
    m, n = int(tok[0]), int(tok[1])
    if m <= 0 or n <= 0 or m != tok[0] or n != tok[1]:
        raise ValueError(f"{name}: bad header {tok[0]} {tok[1]}")
    return m, n


def _records(tok: np.ndarray, pos: int, count: int, lead: int, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Walk `count` variable-length records laid out as
    [lead values] k [k indices]. Returns (record start positions, k per record, end)."""
    starts = np.empty(count, dtype=np.int64)
    sizes = np.empty(count, dtype=np.int64)
    size = tok.size
    for r in range(count):
        if pos + lead >= size:
            raise ValueError(f"{name}: file ends inside record {r + 1} of {count}")
        k = int(tok[pos + lead])
        if k < 0 or k != tok[pos + lead]:
            raise ValueError(f"{name}: bad entry count in record {r + 1}")
        starts[r] = pos
        sizes[r] = k
        pos += lead + 1 + k
    if pos > size:
        raise ValueError(f"{name}: file ends inside the last record")
    return starts, sizes, pos


def _gather(tok: np.ndarray, starts: np.ndarray, sizes: np.ndarray, lead: int) -> np.ndarray:
    ptr = np.zeros(sizes.size + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    first = starts + lead + 1
    return tok[np.repeat(first - ptr[:-1], sizes) + np.arange(ptr[-1], dtype=np.int64)]


def _build(name: str, m: int, n: int, cost: np.ndarray, row: np.ndarray, col: np.ndarray, sense: str, meta: dict) -> Model:
    key = np.unique(row * n + col)                      # sorts by row then column, drops duplicates
    meta = dict(meta, duplicates_removed=int(row.size - key.size))
    row_ptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(key // n, minlength=m), out=row_ptr[1:])
    return Model(name=name, cost=cost, row_ptr=row_ptr, col_idx=(key % n).astype(np.int32),
                 row_sense=np.full(m, sense, dtype="U1"), meta=meta)


def read_scp(path, name: str | None = None) -> Model:
    """OR-Library set covering, row-wise layout."""
    name = name or instance_name(path)
    tok = numeric_tokens(path)
    m, n = _header(tok, name)
    if tok.size < 2 + n:
        raise ValueError(f"{name}: file ends inside the cost block")
    cost = tok[2:2 + n].copy()
    starts, sizes, end = _records(tok, 2 + n, m, 0, name)
    if end != tok.size:
        raise ValueError(f"{name}: {tok.size - end} unexpected trailing tokens")
    col = to_index(_gather(tok, starts, sizes, 0), n, "column", name)
    row = np.repeat(np.arange(m, dtype=np.int64), sizes)
    return _build(name, m, n, cost, row, col, COVER, {"format": "orlib-scp-row", "source": "OR-Library"})


def _read_colwise(path, name: str | None, sense: str, fmt: str) -> Model:
    name = name or instance_name(path)
    tok = numeric_tokens(path)
    m, n = _header(tok, name)
    starts, sizes, end = _records(tok, 2, n, 1, name)
    if end != tok.size:
        raise ValueError(f"{name}: {tok.size - end} unexpected trailing tokens")
    cost = tok[starts].copy()
    row = to_index(_gather(tok, starts, sizes, 1), m, "row", name)
    col = np.repeat(np.arange(n, dtype=np.int64), sizes)
    return _build(name, m, n, cost, row, col, sense, {"format": fmt, "source": "OR-Library"})


def read_scp_rail(path, name: str | None = None) -> Model:
    """OR-Library set covering, column-wise layout (rail instances)."""
    return _read_colwise(path, name, COVER, "orlib-scp-col")


def read_spp(path, name: str | None = None) -> Model:
    """OR-Library set partitioning (Chu & Beasley 1998 test set)."""
    return _read_colwise(path, name, PARTITION, "orlib-spp")
