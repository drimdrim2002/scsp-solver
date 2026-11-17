"""Common model representation shared by readers, generators and solver runners.

    min  cost @ x + obj_offset
    s.t. sum_{j in S_i} x_j  (>= | = | <=)  1      set rows   (row_sense[i] in "G", "E", "L")
         side_val @ x        (<= | = | >=)  rhs    side rows  (side_sense[k] in "L", "E", "G")
         x binary

Set rows are stored as a 0/1 pattern in CSR form (row_ptr, col_idx); all their
coefficients are 1 and their right-hand side is 1. Side rows are a general
sparse CSR matrix (side_ptr, side_idx, side_val) with their own senses and
right-hand sides. See docs/design.md section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

COVER, PARTITION, PACK = "G", "E", "L"
_SENSES = (COVER, PARTITION, PACK)


def _as_ptr(lengths: Iterable[int]) -> np.ndarray:
    lengths = np.fromiter(lengths, dtype=np.int64)
    ptr = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])
    return ptr


@dataclass
class Model:
    name: str
    cost: np.ndarray                      # float64, length n_cols
    row_ptr: np.ndarray                   # int64, length n_rows + 1
    col_idx: np.ndarray                   # int32, length nnz, sorted within a row
    row_sense: np.ndarray                 # "U1", length n_rows, values "G" / "E" / "L"
    side_ptr: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    side_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    side_val: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    side_sense: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="U1"))
    side_rhs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obj_offset: float = 0.0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    def __post_init__(self) -> None:
        self.cost = np.ascontiguousarray(self.cost, dtype=np.float64)
        self.row_ptr = np.ascontiguousarray(self.row_ptr, dtype=np.int64)
        self.col_idx = np.ascontiguousarray(self.col_idx, dtype=np.int32)
        self.row_sense = np.asarray(self.row_sense, dtype="U1")
        self.side_ptr = np.ascontiguousarray(self.side_ptr, dtype=np.int64)
        self.side_idx = np.ascontiguousarray(self.side_idx, dtype=np.int32)
        self.side_val = np.ascontiguousarray(self.side_val, dtype=np.float64)
        self.side_sense = np.asarray(self.side_sense, dtype="U1")
        self.side_rhs = np.ascontiguousarray(self.side_rhs, dtype=np.float64)
        self.validate()

    @classmethod
    def from_rows(
        cls,
        name: str,
        cost: Sequence[float],
        rows: Sequence[Sequence[int]],
        row_sense: Sequence[str] | str,
        side: Sequence[tuple[Sequence[int], Sequence[float], str, float]] = (),
        obj_offset: float = 0.0,
        meta: dict | None = None,
    ) -> "Model":
        """Build from per-row column lists (0-based). `side` is a list of
        (indices, coefficients, sense, rhs)."""
        if isinstance(row_sense, str):
            row_sense = [row_sense] * len(rows)
        row_ptr = _as_ptr(len(r) for r in rows)
        col_idx = np.zeros(int(row_ptr[-1]), dtype=np.int32)
        for i, r in enumerate(rows):
            col_idx[row_ptr[i]:row_ptr[i + 1]] = np.sort(np.asarray(r, dtype=np.int32))
        side_ptr = _as_ptr(len(s[0]) for s in side)
        side_idx = np.zeros(int(side_ptr[-1]), dtype=np.int32)
        side_val = np.zeros(int(side_ptr[-1]), dtype=np.float64)
        for k, (idx, val, _, _) in enumerate(side):
            order = np.argsort(np.asarray(idx, dtype=np.int32), kind="stable")
            side_idx[side_ptr[k]:side_ptr[k + 1]] = np.asarray(idx, dtype=np.int32)[order]
            side_val[side_ptr[k]:side_ptr[k + 1]] = np.asarray(val, dtype=np.float64)[order]
        return cls(
            name=name, cost=np.asarray(cost, dtype=np.float64), row_ptr=row_ptr, col_idx=col_idx,
            row_sense=np.asarray(list(row_sense), dtype="U1"),
            side_ptr=side_ptr, side_idx=side_idx, side_val=side_val,
            side_sense=np.asarray([s[2] for s in side], dtype="U1"),
            side_rhs=np.asarray([s[3] for s in side], dtype=np.float64),
            obj_offset=obj_offset, meta=dict(meta or {}),
        )

    @classmethod
    def from_columns(
        cls,
        name: str,
        cost: Sequence[float],
        n_rows: int,
        col_ptr: np.ndarray,
        row_idx: np.ndarray,
        row_sense: Sequence[str] | str,
        **kwargs,
    ) -> "Model":
        """Build from a column-wise (CSC) 0/1 pattern, e.g. the OR-Library SPP layout."""
        col_ptr = np.asarray(col_ptr, dtype=np.int64)
        row_idx = np.asarray(row_idx, dtype=np.int64)
        n_cols = len(col_ptr) - 1
        col_of = np.repeat(np.arange(n_cols, dtype=np.int64), np.diff(col_ptr))
        order = np.lexsort((col_of, row_idx))           # by row, then column
        counts = np.bincount(row_idx, minlength=n_rows)
        row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(counts, out=row_ptr[1:])
        if isinstance(row_sense, str):
            row_sense = [row_sense] * n_rows
        return cls(name=name, cost=np.asarray(cost, dtype=np.float64), row_ptr=row_ptr,
                   col_idx=col_of[order].astype(np.int32),
                   row_sense=np.asarray(list(row_sense), dtype="U1"), **kwargs)

    # ------------------------------------------------------------- properties
    @property
    def n_cols(self) -> int:
        return int(self.cost.shape[0])

    @property
    def n_rows(self) -> int:
        """Number of set rows (cover + partition + pack)."""
        return int(self.row_ptr.shape[0] - 1)

    @property
    def n_side(self) -> int:
        return int(self.side_ptr.shape[0] - 1)

    @property
    def n_cons(self) -> int:
        """All constraints: set rows plus side rows."""
        return self.n_rows + self.n_side

    @property
    def nnz(self) -> int:
        return int(self.col_idx.shape[0])

    @property
    def nnz_side(self) -> int:
        return int(self.side_idx.shape[0])

    @property
    def kind(self) -> str:
        """"SC", "SP", "PACK" or "MIXED" by set-row senses, with "+side" if side rows exist."""
        present = {s for s in _SENSES if np.any(self.row_sense == s)}
        base = {frozenset("G"): "SC", frozenset("E"): "SP", frozenset("L"): "PACK"}.get(frozenset(present), "MIXED")
        return base + ("+side" if self.n_side else "")

    def validate(self) -> None:
        n, m, k = self.n_cols, self.n_rows, self.n_side
        if self.row_ptr[0] != 0 or np.any(np.diff(self.row_ptr) < 0) or self.row_ptr[-1] != self.nnz:
            raise ValueError(f"{self.name}: bad row_ptr")
        if self.row_sense.shape[0] != m or not np.all(np.isin(self.row_sense, _SENSES)):
            raise ValueError(f"{self.name}: bad row_sense")
        if self.nnz and (self.col_idx.min() < 0 or self.col_idx.max() >= n):
            raise ValueError(f"{self.name}: column index out of range")
        if self.side_ptr[0] != 0 or np.any(np.diff(self.side_ptr) < 0) or self.side_ptr[-1] != self.nnz_side:
            raise ValueError(f"{self.name}: bad side_ptr")
        if self.side_val.shape[0] != self.nnz_side or self.side_sense.shape[0] != k or self.side_rhs.shape[0] != k:
            raise ValueError(f"{self.name}: side arrays have inconsistent lengths")
        if k and not np.all(np.isin(self.side_sense, _SENSES)):
            raise ValueError(f"{self.name}: bad side_sense")
        if self.nnz_side and (self.side_idx.min() < 0 or self.side_idx.max() >= n):
            raise ValueError(f"{self.name}: side column index out of range")
        if not np.all(np.isfinite(self.cost)):
            raise ValueError(f"{self.name}: non-finite cost")
        # a column may appear at most once per set row
        row_of = np.repeat(np.arange(m, dtype=np.int64), np.diff(self.row_ptr))
        key = row_of * max(n, 1) + self.col_idx
        if np.unique(key).shape[0] != key.shape[0]:
            raise ValueError(f"{self.name}: duplicate column inside a set row")

    # ----------------------------------------------------------------- views
    def row(self, i: int) -> np.ndarray:
        return self.col_idx[self.row_ptr[i]:self.row_ptr[i + 1]]

    def csc(self) -> tuple[np.ndarray, np.ndarray]:
        """Column-wise view of the set rows: (col_ptr, row_idx)."""
        m, n = self.n_rows, self.n_cols
        row_of = np.repeat(np.arange(m, dtype=np.int32), np.diff(self.row_ptr))
        order = np.argsort(self.col_idx, kind="stable")
        col_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.col_idx, minlength=n), out=col_ptr[1:])
        return col_ptr, row_of[order]

    def to_scipy(self):
        """Full constraint system as (A csr, lower, upper): set rows first, then side rows."""
        import scipy.sparse as sp

        m, k, n = self.n_rows, self.n_side, self.n_cols
        a_set = sp.csr_matrix((np.ones(self.nnz), self.col_idx, self.row_ptr), shape=(m, n))
        a_side = sp.csr_matrix((self.side_val, self.side_idx, self.side_ptr), shape=(k, n))
        lower = np.concatenate([
            np.where(self.row_sense == PACK, -np.inf, 1.0),
            np.where(self.side_sense == PACK, -np.inf, self.side_rhs),
        ])
        upper = np.concatenate([
            np.where(self.row_sense == COVER, np.inf, 1.0),
            np.where(self.side_sense == COVER, np.inf, self.side_rhs),
        ])
        return sp.vstack([a_set, a_side], format="csr"), lower, upper

    # ------------------------------------------------------------- solutions
    def objective(self, x: np.ndarray) -> float:
        return float(self.cost @ np.asarray(x, dtype=np.float64) + self.obj_offset)

    def activities(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(set-row activity, side-row activity) of a 0/1 vector."""
        x = np.asarray(x, dtype=np.float64)
        csum = np.concatenate([[0.0], np.cumsum(x[self.col_idx])])
        set_act = csum[self.row_ptr[1:]] - csum[self.row_ptr[:-1]]
        ssum = np.concatenate([[0.0], np.cumsum(self.side_val * x[self.side_idx])])
        side_act = ssum[self.side_ptr[1:]] - ssum[self.side_ptr[:-1]]
        return set_act, side_act

    def violations(self, x: np.ndarray, tol: float = 1e-6) -> dict:
        """Counts of violated constraints and non-binary entries of x."""
        x = np.asarray(x, dtype=np.float64)
        set_act, side_act = self.activities(x)
        rs, ss = self.row_sense, self.side_sense
        set_bad = (((rs == COVER) & (set_act < 1 - tol)) | ((rs == PACK) & (set_act > 1 + tol))
                   | ((rs == PARTITION) & (np.abs(set_act - 1) > tol)))
        side_bad = (((ss == COVER) & (side_act < self.side_rhs - tol)) | ((ss == PACK) & (side_act > self.side_rhs + tol))
                    | ((ss == PARTITION) & (np.abs(side_act - self.side_rhs) > tol)))
        nonbin = int(np.sum(np.minimum(np.abs(x), np.abs(x - 1)) > tol))
        return {"set_rows": int(set_bad.sum()), "side_rows": int(side_bad.sum()), "non_binary": nonbin}

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        return not any(self.violations(x, tol).values())

    # ------------------------------------------------------------------ misc
    def stats(self) -> dict:
        m, n = self.n_rows, self.n_cols
        row_len = np.diff(self.row_ptr)
        col_len = np.bincount(self.col_idx, minlength=n) if n else np.zeros(0, dtype=np.int64)
        return {
            "name": self.name, "kind": self.kind, "rows": m, "cols": n, "nnz": self.nnz,
            "density": self.nnz / (m * n) if m and n else 0.0,
            "cover_rows": int(np.sum(self.row_sense == COVER)),
            "partition_rows": int(np.sum(self.row_sense == PARTITION)),
            "pack_rows": int(np.sum(self.row_sense == PACK)),
            "side_rows": self.n_side, "side_nnz": self.nnz_side,
            "row_len_min": int(row_len.min()) if m else 0, "row_len_max": int(row_len.max()) if m else 0,
            "col_len_min": int(col_len.min()) if n else 0, "col_len_max": int(col_len.max()) if n else 0,
            "cost_min": float(self.cost.min()) if n else 0.0, "cost_max": float(self.cost.max()) if n else 0.0,
            "cost_integral": bool(np.all(self.cost == np.round(self.cost))) if n else True,
            "empty_cols": int(np.sum(col_len == 0)) if n else 0,
        }

    def save_npz(self, path) -> None:
        import json

        np.savez_compressed(
            path, name=np.asarray(self.name), cost=self.cost, row_ptr=self.row_ptr, col_idx=self.col_idx,
            row_sense=self.row_sense, side_ptr=self.side_ptr, side_idx=self.side_idx, side_val=self.side_val,
            side_sense=self.side_sense, side_rhs=self.side_rhs, obj_offset=np.asarray(self.obj_offset),
            meta=np.asarray(json.dumps(self.meta)),
        )

    @classmethod
    def load_npz(cls, path) -> "Model":
        import json

        with np.load(path, allow_pickle=False) as z:
            return cls(
                name=str(z["name"]), cost=z["cost"], row_ptr=z["row_ptr"], col_idx=z["col_idx"],
                row_sense=z["row_sense"], side_ptr=z["side_ptr"], side_idx=z["side_idx"], side_val=z["side_val"],
                side_sense=z["side_sense"], side_rhs=z["side_rhs"], obj_offset=float(z["obj_offset"]),
                meta=json.loads(str(z["meta"])),
            )

    def __repr__(self) -> str:
        return (f"Model({self.name!r}, {self.kind}, rows={self.n_rows}, cols={self.n_cols}, "
                f"nnz={self.nnz}, side={self.n_side})")
