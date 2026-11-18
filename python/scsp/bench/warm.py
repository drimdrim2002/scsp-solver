"""Warm starts that imitate a good heuristic solution (the DMS use case: an ALNS
incumbent is passed to the set covering solver, which then has a few seconds).

    python -m scsp.bench.warm --sets dms --gap 2.0                  # kind "near": benchmarks/warm_starts.json
    python -m scsp.bench.warm --sets dms --gap 2.0 --kind indep     # kind "indep": benchmarks/warm_starts_indep.json

No solver is involved in either kind, so a start favours none of them.

kind "near" (close to the best known solution):
  1. take the best solution recorded for the instance in results/runs/*.jsonl;
  2. repeat: remove a random part of its columns and cover the open rows again
     greedily, by cost per newly covered row with random noise on the score and
     without exceeding any `<=` side row; then drop columns that became redundant;
  3. stop as soon as the objective is `gap` percent (within +-`window`) above
     the best known value.
  The result shares about 90% of its columns with the best known solution.

kind "indep" (independent of the best known solution): a small destroy/repair
descent, i.e. what an ALNS does, written here in a few lines:
  1. build a solution from nothing with the same noisy greedy;
  2. repeat: remove a random part of the columns, repair, keep the result if it
     is better;
  3. stop as soon as the objective is within `gap` percent of the best known
     value. Only that value is used, as the stopping level; none of the best
     known solution's columns are looked at.

Only covering models (set rows `>= 1`, side rows `<=`) are handled.
`run --warm-start` uses the "near" start, `run --warm-start --warm-kind indep`
the other one; an instance without a stored start falls back to
meta["warm_start"] (the generator's hidden solution, which is far from optimal).
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np

from ..model import COVER, PACK, Model
from .catalog import bench_dir, discover

FILES = {"near": "warm_starts.json", "indep": "warm_starts_indep.json"}


def path(root: Path | None = None, kind: str = "near") -> Path:
    return (root or bench_dir()) / FILES[kind]


def load(root: Path | None = None, kind: str = "near") -> dict:
    p = path(root, kind)
    return json.loads(p.read_text()) if p.exists() else {}


def stored(name: str, root: Path | None = None, kind: str = "near"):
    """Column indices of the stored start of an instance, or None."""
    rec = load(root, kind).get(name)
    return None if rec is None else rec["cols"]


def best_recorded(name: str, root: Path | None = None):
    """(objective, columns) of the best verified solution in results/runs/*.jsonl."""
    best = None
    for f in sorted(((root or bench_dir()) / "results" / "runs").glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if f'"instance": "{name}"' not in line:
                    continue
                r = json.loads(line)
                if r.get("verified") and r.get("sol") and (best is None or r["obj"] < best[0]):
                    best = (float(r["obj"]), r["sol"])
    return best


class _Repair:
    """Noisy greedy cover of the open rows, then removal of redundant columns."""

    def __init__(self, model: Model):
        if not (np.all(model.row_sense == COVER) and np.all(model.side_sense == PACK)):
            raise ValueError(f"{model.name}: only covering models with <= side rows are handled")
        self.model = model
        a, _, self.upper = model.to_scipy()
        self.a = a
        self.at = a.T.tocsr()                            # column -> rows
        self.at_set = self.at[:, :model.n_rows]
        self.m = model.n_rows

    def __call__(self, x: np.ndarray, rng, noise: float) -> bool:
        """Completes the 0/1 vector x in place. False if the greedy got stuck."""
        model, at, m = self.model, self.at, self.m
        act = self.a @ x.astype(np.float64)
        while True:
            open_rows = act[:m] < 1 - 1e-9
            if not open_rows.any():
                break
            gain = self.at_set @ open_rows.astype(np.float64)
            blocked = act + 1e-9 > self.upper            # a `<=` row without room for one more unit
            cand = (gain > 0) & ~x
            if blocked.any():
                cand &= (at @ blocked.astype(np.float64)) == 0
            idx = np.flatnonzero(cand)
            if len(idx) == 0:
                return False
            score = model.cost[idx] / gain[idx] * (1.0 + noise * rng.random(len(idx)))
            j = int(idx[np.argmin(score)])
            x[j] = True
            act += at[j].toarray().ravel()
        for j in sorted(np.flatnonzero(x), key=lambda j: -model.cost[j]):   # drop redundant columns, dearest first
            rows = at[j].indices
            rows = rows[rows < m]
            if np.all(act[rows] >= 2 - 1e-9):
                x[j] = False
                act -= at[j].toarray().ravel()
        return bool(model.is_feasible(x.astype(np.float64)))


def perturb(model: Model, start, target: float, gap: float, window: float, seed: int, frac: float = 0.15,
            noise: float = 1.0, max_rounds: int = 2000):
    """Kind "near": destroy and repair `start` until its objective is gap +- window percent above `target`."""
    repair = _Repair(model)
    rng = np.random.default_rng(seed)
    lo = target * (1 + (gap - window) / 100.0)
    hi = target * (1 + (gap + window) / 100.0)
    cur = np.zeros(model.n_cols, dtype=bool)
    cur[np.asarray(start, dtype=np.int64)] = True
    cur_obj = model.objective(cur)
    rounds = 0
    while cur_obj < lo and rounds < max_rounds:
        rounds += 1
        x = cur.copy()
        sel = np.flatnonzero(x)
        x[rng.choice(sel, size=max(1, int(round(frac * len(sel)))), replace=False)] = False
        if not repair(x, rng, noise):
            continue
        obj = model.objective(x)
        if obj <= hi:
            cur, cur_obj = x, obj                        # may be better or worse than before: a random walk below `hi`
    if not (lo <= cur_obj <= hi):
        raise RuntimeError(f"{model.name}: no start within the window after {rounds} rounds (objective {cur_obj:.6g})")
    return np.flatnonzero(cur).tolist(), cur_obj, rounds


def descend(model: Model, target: float, gap: float, window: float, seed: int, noise: float = 0.3,
            max_rounds: int = 30000):
    """Kind "indep": greedy from nothing, then destroy/repair descent until the objective is at most
    gap + window percent above `target`. Steps that would land below gap - window percent are refused,
    so the result is inside the window unless the descent stalls (then the best found is returned)."""
    repair = _Repair(model)
    rng = np.random.default_rng(seed)
    lo = target * (1 + (gap - window) / 100.0)
    hi = target * (1 + (gap + window) / 100.0)
    cur = np.zeros(model.n_cols, dtype=bool)
    for _ in range(50):
        if repair(cur, rng, noise):
            break
        cur[:] = False
    else:
        raise RuntimeError(f"{model.name}: the greedy found no feasible solution")
    cur_obj = first_obj = model.objective(cur)
    rounds = 0
    while cur_obj > hi and rounds < max_rounds:
        rounds += 1
        x = cur.copy()
        sel = np.flatnonzero(x)
        frac = rng.uniform(0.05, 0.3)
        x[rng.choice(sel, size=max(1, int(round(frac * len(sel)))), replace=False)] = False
        if not repair(x, rng, noise * rng.random()):
            continue
        obj = model.objective(x)
        if lo <= obj < cur_obj:
            cur, cur_obj = x, obj
    return np.flatnonzero(cur).tolist(), cur_obj, rounds, first_obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["dms"])
    ap.add_argument("--instances", nargs="*", default=[])
    ap.add_argument("--kind", choices=sorted(FILES), default="near")
    ap.add_argument("--gap", type=float, default=2.0, help="percent above the best known value")
    ap.add_argument("--window", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--max-rounds", type=int, default=30000, help="kind indep: destroy/repair rounds before giving up")
    args = ap.parse_args(argv)

    root = bench_dir()
    with open(root / "bks.csv") as f:
        bks = {r["instance"]: float(r["bks"]) for r in csv.DictReader(f)}
    out = load(root, args.kind)
    for inst in discover(root):
        if inst.set not in args.sets and inst.name not in args.instances:
            continue
        model = inst.load()
        best = best_recorded(inst.name, root)
        if best is None:
            print(f"{inst.name}: no recorded solution, skipped")
            continue
        seed = args.seed + zlib.crc32(inst.name.encode())
        b = bks[inst.name]
        if args.kind == "near":
            cols, obj, rounds = perturb(model, best[1], b, args.gap, args.window, seed)
            rec = {"from_obj": best[0],
                   "method": f"destroy/repair of the best recorded solution, frac=0.15, noise=1.0, target {args.gap}+-{args.window}%"}
        else:
            cols, obj, rounds, first = descend(model, b, args.gap, args.window, seed, max_rounds=args.max_rounds)
            rec = {"greedy_obj": first,
                   "method": f"noisy greedy from nothing, then destroy/repair descent (frac 5-30%, noise<=0.3) down to {args.gap}+-{args.window}%"}
        shared = len(set(cols) & set(best[1]))
        out[inst.name] = {"cols": cols, "obj": obj, "bks": b, "gap_pct": 100.0 * (obj - b) / abs(b), "shared_cols": shared,
                          "best_cols": len(best[1]), "n_cols": len(cols), "rounds": rounds, "seed": seed, **rec}
        print(f"{inst.name:28s} start {obj:12.4f}  bks {b:12.4f}  gap {out[inst.name]['gap_pct']:.2f}%  "
              f"cols {len(cols)} (shared with best {shared}/{len(best[1])})  rounds {rounds}", flush=True)
        path(root, args.kind).write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print(f"wrote {len(out)} starts to {path(root, args.kind)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
