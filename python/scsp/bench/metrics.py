"""Benchmark metrics (docs/design.md section 5.3).

  1. objective at fixed times, as a gap to the best known value (BKS), in %
  2. time to proven optimality: shifted geometric mean (shift 10 s) and solved count
  3. primal integral (Berthold 2013)
  4. final gap reported by the solver, (obj - bound) / |obj|

    python -m scsp.bench.metrics                  # writes benchmarks/results/summary_*.csv and baseline_report.md

How the value "at time t" is obtained: if a run with time limit exactly t
exists it is used (its final objective); otherwise the value is read from the
trace of the shortest longer run of the same (config, instance, seed). The
`source` column of summary_by_run.csv says which ("run" or "trace").
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from .catalog import bench_dir, discover
from .run import load_runs, results_dir

TIMES = (5.0, 10.0, 30.0, 60.0)
SHIFT = 10.0


# ------------------------------------------------------------- trace maths
def obj_at(trace: list, t: float) -> float | None:
    """Best incumbent objective known at time t (None if there is none yet)."""
    best = None
    for ev in trace:
        if ev[0] > t + 1e-9:
            break
        if ev[1] is not None and (best is None or ev[1] < best):
            best = ev[1]
    return best


def bound_at(trace: list, t: float) -> float | None:
    best = None
    for ev in trace:
        if ev[0] > t + 1e-9:
            break
        if ev[2] is not None and (best is None or ev[2] > best):
            best = ev[2]
    return best


def primal_gap(obj: float | None, bks: float) -> float:
    """Berthold (2013): 0 if equal, 1 if no solution or opposite signs, else
    |obj - bks| / max(|obj|, |bks|). Always in [0, 1]."""
    if obj is None:
        return 1.0
    if obj == bks:
        return 0.0
    if obj * bks < 0:
        return 1.0
    return min(1.0, abs(obj - bks) / max(abs(obj), abs(bks)))


def primal_integral(trace: list, bks: float, horizon: float) -> float:
    """Integral over [0, horizon] of the primal gap step function, in seconds."""
    total, t_prev, best = 0.0, 0.0, None
    for ev in trace:
        t = min(float(ev[0]), horizon)
        if ev[1] is None or (best is not None and ev[1] >= best):
            continue
        total += primal_gap(best, bks) * (t - t_prev)
        t_prev, best = t, ev[1]
        if t >= horizon:
            break
    total += primal_gap(best, bks) * max(0.0, horizon - t_prev)
    return total


def gap_pct(obj: float | None, bks: float | None) -> float | None:
    """100 * (obj - bks) / |bks|; None without a solution or a BKS."""
    if obj is None or bks is None:
        return None
    if bks == 0:
        return 0.0 if obj == 0 else math.inf
    return 100.0 * (obj - bks) / abs(bks)


def shifted_geomean(values, shift: float = SHIFT) -> float | None:
    values = list(values)
    if not values:
        return None
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


# ------------------------------------------------------------------ tables
def load_bks(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["instance"]: r for r in csv.DictReader(f)}


def load_all_runs(root: Path | None = None) -> list:
    runs = []
    for path in sorted(results_dir(root).glob("*.jsonl")):
        runs.extend(load_runs(path))
    return runs


def run_value(r: dict):
    """Objective of the solution a run returned: recomputed from the solution when the record has
    it (`obj_check`), so a solver that reports a wrong objective is charged what its solution costs."""
    v = r.get("obj_check")
    return r["obj"] if v is None else v


def per_run_rows(runs: list, bks: dict, sets: dict, times=TIMES) -> list:
    """One row per (config, instance, seed, threads, warm_start)."""
    groups = defaultdict(list)
    for r in runs:
        if r["status"] in ("skipped", "error", "killed"):
            continue
        groups[(r["config"], r["instance"], int(r["seed"]), int(r["threads"]), bool(r["warm_start"]))].append(r)

    rows = []
    for (config, inst, seed, threads, warm), rs in sorted(groups.items()):
        rs.sort(key=lambda r: float(r["time_limit"]))
        longest = rs[-1]
        ref = bks.get(inst)
        bks_val = float(ref["bks"]) if ref and ref["bks"] not in ("", None) else None
        row = {"config": config, "instance": inst, "set": sets.get(inst, ""), "seed": seed, "threads": threads,
               "warm_start": int(warm), "n_cols": longest.get("n_cols"), "n_cons": longest.get("n_cons"),
               "bks": bks_val, "max_time_limit": float(longest["time_limit"])}
        for t in times:
            exact = [r for r in rs if float(r["time_limit"]) == t]
            longer = [r for r in rs if float(r["time_limit"]) > t]
            # a run that proved optimality before t is the same run under any limit >= its runtime
            proved = [r for r in rs if r["status"] == "optimal" and float(r["runtime"]) <= t]
            if exact:
                obj, src = run_value(exact[0]), "run"
            elif proved:
                obj, src = run_value(proved[0]), "run"
            elif longer and longer[0].get("verified") is False:
                obj, src = None, "unverified"            # the trace of such a run cannot be trusted either
            elif longer:
                obj, src = obj_at(longer[0]["trace"], t), "trace"
            else:
                obj, src = None, ""
            tag = f"{t:g}"
            row[f"obj_{tag}"] = obj
            row[f"gap_{tag}"] = gap_pct(obj, bks_val)
            row[f"src_{tag}"] = src
        solved = [r for r in rs if r["status"] == "optimal" and r.get("verified") is not False]
        row["status"] = "optimal" if solved else ("unverified" if longest["status"] == "optimal" else longest["status"])
        row["proof_time"] = min(float(r["runtime"]) for r in solved) if solved else None
        row["final_obj"] = run_value(longest)
        row["final_bound"] = longest["bound"]
        row["final_gap_pct"] = None if longest["gap"] is None else 100.0 * longest["gap"]
        horizon = float(longest["time_limit"])
        row["primal_integral"] = None if bks_val is None else primal_integral(longest["trace"], bks_val, horizon)
        row["primal_integral_pct"] = (None if row["primal_integral"] is None
                                      else 100.0 * row["primal_integral"] / horizon)
        row["verified"] = all(r.get("verified") in (True, None) for r in rs)
        rows.append(row)
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def per_instance_rows(run_rows: list, times=TIMES) -> list:
    """Average over seeds."""
    groups = defaultdict(list)
    for r in run_rows:
        groups[(r["config"], r["instance"], r["threads"], r["warm_start"])].append(r)
    out = []
    for (config, inst, threads, warm), rs in sorted(groups.items()):
        horizon = max(r["max_time_limit"] for r in rs)
        row = {"config": config, "instance": inst, "set": rs[0]["set"], "threads": threads, "warm_start": warm,
               "seeds": len(rs), "n_cols": rs[0]["n_cols"], "n_cons": rs[0]["n_cons"], "bks": rs[0]["bks"]}
        for t in times:
            tag = f"{t:g}"
            row[f"gap_{tag}"] = _mean(r[f"gap_{tag}"] for r in rs)
            row[f"found_{tag}"] = sum(r[f"obj_{tag}"] is not None for r in rs)
        row["solved"] = sum(r["status"] == "optimal" for r in rs)
        row["proof_time_sgm"] = shifted_geomean(r["proof_time"] if r["proof_time"] is not None else horizon for r in rs)
        row["final_gap_pct"] = _mean(r["final_gap_pct"] for r in rs)
        row["primal_integral_pct"] = _mean(r["primal_integral_pct"] for r in rs)
        row["best_obj"] = min((r["final_obj"] for r in rs if r["final_obj"] is not None), default=None)
        row["verified"] = all(r["verified"] for r in rs)
        out.append(row)
    return out


def per_set_rows(inst_rows: list, run_rows: list, times=TIMES) -> list:
    """Average over the instances of a set (instances weigh equally)."""
    groups = defaultdict(list)
    for r in inst_rows:
        groups[(r["config"], r["set"], r["threads"], r["warm_start"])].append(r)
    horizon_of = defaultdict(float)
    for r in run_rows:
        k = (r["config"], r["set"], r["threads"], r["warm_start"])
        horizon_of[k] = max(horizon_of[k], r["max_time_limit"])
    out = []
    for key, rs in sorted(groups.items()):
        config, set_name, threads, warm = key
        row = {"config": config, "set": set_name, "threads": threads, "warm_start": warm, "instances": len(rs),
               "runs": sum(r["seeds"] for r in rs)}
        for t in times:
            tag = f"{t:g}"
            row[f"gap_{tag}"] = _mean(r[f"gap_{tag}"] for r in rs)
            row[f"found_{tag}"] = sum(r[f"found_{tag}"] for r in rs)
        row["solved_runs"] = sum(r["solved"] for r in rs)
        row["proof_time_sgm"] = shifted_geomean(r["proof_time_sgm"] for r in rs)
        row["final_gap_pct"] = _mean(r["final_gap_pct"] for r in rs)
        row["primal_integral_pct"] = _mean(r["primal_integral_pct"] for r in rs)
        out.append(row)
    return out


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float):
            return f"{v:.10g}"
        return v

    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: fmt(v) for k, v in r.items()})


def _cell(v, digits=2) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def markdown_report(set_rows: list, inst_rows: list, times=TIMES) -> str:
    lines = ["# Baseline report (reference solvers)", "",
             "Generated by `python -m scsp.bench.metrics`. Gaps are `100 * (obj - BKS) / |BKS|`, averaged over",
             "seeds, then over the instances of a set. `found` counts runs that had a solution at that time.",
             f"`proof sgm` is the shifted geometric mean (shift {SHIFT:g} s) of the time to proven optimality;",
             "an unsolved run counts with its time limit. `PI %` is the primal integral divided by the time limit.", ""]
    head = ["set", "config", "inst", "runs"] + [f"gap@{t:g}s %" for t in times] + [f"found@{times[0]:g}s", "solved runs",
                                                                                 "proof sgm s", "final gap %", "PI %"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in sorted(set_rows, key=lambda r: (r["warm_start"], r["set"], r["config"])):
        name = r["set"] + (" (warm)" if r["warm_start"] else "")
        cells = [name, r["config"], r["instances"], r["runs"]] + [_cell(r[f"gap_{t:g}"], 3) for t in times]
        cells += [r[f"found_{times[0]:g}"], r["solved_runs"], _cell(r["proof_time_sgm"]), _cell(r["final_gap_pct"]),
                  _cell(r["primal_integral_pct"], 3)]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    # head-to-head on the instances both solvers ran
    by = defaultdict(dict)
    for r in inst_rows:
        if not r["warm_start"]:
            by[r["instance"]][r["config"]] = r
    both = {i: d for i, d in by.items() if "gurobi" in d and "highs" in d}
    if both:
        lines += ["", f"## Gurobi vs HiGHS on the {len(both)} instances both can run", "",
                  "| time | Gurobi better | tie | HiGHS better | mean gap Gurobi % | mean gap HiGHS % |", "|---|---|---|---|---|---|"]
        for t in times:
            tag = f"gap_{t:g}"
            g = [d["gurobi"][tag] for d in both.values()]
            h = [d["highs"][tag] for d in both.values()]
            win = tie = lose = 0
            for a, b in zip(g, h):
                a = math.inf if a is None else a
                b = math.inf if b is None else b
                if abs(a - b) <= 1e-9 or (a == b):
                    tie += 1
                elif a < b:
                    win += 1
                else:
                    lose += 1
            lines.append(f"| {t:g} s | {win} | {tie} | {lose} | {_cell(_mean(g), 3)} | {_cell(_mean(h), 3)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", nargs="+", type=float, default=list(TIMES))
    args = ap.parse_args(argv)
    root = bench_dir()
    runs = load_all_runs(root)
    bks = load_bks(root / "bks.csv")
    sets = {i.name: i.set for i in discover(root)}
    times = tuple(args.times)
    run_rows = per_run_rows(runs, bks, sets, times)
    inst_rows = per_instance_rows(run_rows, times)
    set_rows = per_set_rows(inst_rows, run_rows, times)
    out = root / "results"
    write_csv(out / "summary_by_run.csv", run_rows)
    write_csv(out / "summary_by_instance.csv", inst_rows)
    write_csv(out / "summary_by_set.csv", set_rows)
    (out / "baseline_report.md").write_text(markdown_report(set_rows, inst_rows, times))
    bad = [r for r in run_rows if not r["verified"]]
    print(f"{len(runs)} runs -> {len(run_rows)} run rows, {len(inst_rows)} instance rows, {len(set_rows)} set rows; "
          f"{len(bad)} rows with an unverified solution")


if __name__ == "__main__":
    main()
