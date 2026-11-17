"""Small synthetic set partitioning instances (crew-pairing flavour).

Rows are "legs" ordered in time. A column is a "pairing": it starts at a leg
and repeatedly jumps a few legs ahead, so it covers 2..max_len legs inside a
short window. Cost = fixed + per_leg * legs + span_cost * (last - first), times
a random factor, rounded to an integer, so longer pairings are cheaper per leg
but harder to fit together.

Feasible by construction: a hidden partition built from the same pairing
process is part of the pool; its column indices are in meta["warm_start"].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..model import PARTITION, Model


@dataclass(frozen=True)
class SpParams:
    n_rows: int
    n_cols: int
    seed: int = 1
    max_len: int = 7
    max_gap: int = 4
    fixed: float = 400.0
    per_leg: float = 150.0
    span_cost: float = 25.0
    noise: float = 0.25


def _cost(rows: tuple, p: SpParams, rng: np.random.Generator) -> float:
    base = p.fixed + p.per_leg * len(rows) + p.span_cost * (rows[-1] - rows[0])
    return float(np.rint(base * (1.0 + rng.uniform(-p.noise, p.noise))))


def generate_sp(n_rows: int, n_cols: int, seed: int = 1, name: str | None = None, **overrides) -> Model:
    p = SpParams(n_rows=n_rows, n_cols=n_cols, seed=seed, **overrides)
    m = p.n_rows
    if m < 4:
        raise ValueError("need at least 4 rows")
    rng = np.random.default_rng([p.seed, m, p.n_cols])

    # hidden partition: sweep the rows, each new column picks later unused rows
    free = np.ones(m, dtype=bool)
    hidden = []
    for i in range(m):
        if not free[i]:
            continue
        rows, cur = [i], i
        free[i] = False
        for _ in range(int(rng.integers(1, p.max_len + 1)) - 1):
            ahead = [r for r in range(cur + 1, min(m, cur + 1 + 2 * p.max_gap)) if free[r]]
            if not ahead:
                break
            cur = ahead[int(rng.integers(min(len(ahead), p.max_gap)))]
            free[cur] = False
            rows.append(cur)
        hidden.append(tuple(rows))
    if len(hidden) > p.n_cols:
        raise ValueError(f"n_cols={p.n_cols} is smaller than the hidden partition ({len(hidden)} columns)")

    pool = {rows: _cost(rows, p, rng) for rows in hidden}
    attempts, max_attempts = 0, 200 * p.n_cols + 1000
    while len(pool) < p.n_cols and attempts < max_attempts:
        attempts += 1
        cur = int(rng.integers(m))
        rows = [cur]
        for _ in range(int(rng.integers(2, p.max_len + 1)) - 1):
            cur += int(rng.integers(1, p.max_gap + 1))
            if cur >= m:
                break
            rows.append(cur)
        rows = tuple(rows)
        if rows not in pool:
            pool[rows] = _cost(rows, p, rng)
    if len(pool) < p.n_cols:
        raise RuntimeError(f"only {len(pool)} distinct columns found (wanted {p.n_cols})")

    keys = list(pool)
    keys = [keys[j] for j in rng.permutation(len(keys))]
    index_of = {k: j for j, k in enumerate(keys)}
    cost = np.array([pool[k] for k in keys])
    col_ptr = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum([len(k) for k in keys], out=col_ptr[1:])
    row_idx = np.fromiter((r for k in keys for r in k), dtype=np.int64, count=int(col_ptr[-1]))
    warm = sorted(index_of[k] for k in hidden)
    return Model.from_columns(
        name or f"sp_m{m}_n{p.n_cols}_s{p.seed}", cost, m, col_ptr, row_idx, PARTITION,
        meta={"format": "synthetic-sp", "params": asdict(p), "warm_start": warm,
              "warm_start_obj": float(cost[warm].sum())},
    )
