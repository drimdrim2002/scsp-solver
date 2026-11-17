"""Head-to-head comparison of two solver configs at fixed times (P1 pass criterion).

    python -m scsp.bench.compare --ours scsp --ref highs --times 5 10 30
    python -m scsp.bench.compare --ours scsp --ref gurobi --out benchmarks/results/p1_vs_gurobi

For every instance both configs ran, the objective at time t is averaged over
the seeds both have (a seed without a solution at t counts as "no solution").
An instance is a win if our mean is lower than the reference mean by more than
a relative 1e-9, a tie if they agree within that, else a loss. Values at t come
from `metrics.per_run_rows` (exact run, proven-optimal run, or trace readout).
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from .catalog import bench_dir, discover
from .metrics import load_all_runs, load_bks, per_run_rows, write_csv

SC_SETS = ("dms", "scp4", "scp5", "scp6", "scpa", "scpb", "scpc", "scpd", "scpe",
           "scpnre", "scpnrf", "scpnrg", "scpnrh", "rail")
TOL = 1e-9


def compare(run_rows: list, ours: str, ref: str, times, sets, threads: int = 4, warm: int = 0) -> list:
    by = defaultdict(dict)                               # (instance, set) -> config -> seed -> row
    for r in run_rows:
        if r["config"] in (ours, ref) and r["set"] in sets and r["threads"] == threads and r["warm_start"] == warm:
            by[(r["instance"], r["set"])].setdefault(r["config"], {})[r["seed"]] = r
    rows = []
    for (inst, st), cfg in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if ours not in cfg or ref not in cfg:
            continue
        seeds = sorted(set(cfg[ours]) & set(cfg[ref]))
        if not seeds:
            continue
        row = {"instance": inst, "set": st, "seeds": len(seeds), "bks": cfg[ours][seeds[0]]["bks"]}
        for t in times:
            tag = f"{t:g}"
            a = [cfg[ours][s][f"obj_{tag}"] for s in seeds]
            b = [cfg[ref][s][f"obj_{tag}"] for s in seeds]
            ma = None if any(v is None for v in a) else sum(a) / len(a)
            mb = None if any(v is None for v in b) else sum(b) / len(b)
            row[f"{ours}_{tag}"], row[f"{ref}_{tag}"] = ma, mb
            if ma is None and mb is None:
                res, diff = "tie", None
            elif mb is None:
                res, diff = "win", None
            elif ma is None:
                res, diff = "loss", None
            else:
                diff = 100.0 * (ma - mb) / abs(mb) if mb else math.inf
                tol = TOL * max(1.0, abs(mb))
                res = "win" if ma < mb - tol else ("tie" if ma <= mb + tol else "loss")
            row[f"res_{tag}"], row[f"diff_pct_{tag}"] = res, diff
        rows.append(row)
    return rows


def summarize(rows: list, times) -> list:
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["set"]].append(r)
        groups["ALL"].append(r)
    for st in sorted(groups, key=lambda s: (s == "ALL", s)):
        rs = groups[st]
        row = {"set": st, "instances": len(rs)}
        for t in times:
            tag = f"{t:g}"
            w = sum(r[f"res_{tag}"] == "win" for r in rs)
            ti = sum(r[f"res_{tag}"] == "tie" for r in rs)
            lo = len(rs) - w - ti
            diffs = [r[f"diff_pct_{tag}"] for r in rs if r[f"diff_pct_{tag}"] is not None and math.isfinite(r[f"diff_pct_{tag}"])]
            row.update({f"win_{tag}": w, f"tie_{tag}": ti, f"loss_{tag}": lo,
                        f"win_pct_{tag}": 100.0 * w / len(rs), f"not_worse_pct_{tag}": 100.0 * (w + ti) / len(rs),
                        f"mean_diff_pct_{tag}": sum(diffs) / len(diffs) if diffs else None})
        out.append(row)
    return out


def markdown(summary: list, ours: str, ref: str, times) -> str:
    lines = [f"| set | n | " + " | ".join(f"{t:g}s win/tie/loss | {t:g}s mean diff %" for t in times) + " |",
             "|---|---|" + "---|---|" * len(times)]
    for r in summary:
        cells = []
        for t in times:
            tag = f"{t:g}"
            d = r[f"mean_diff_pct_{tag}"]
            cells.append(f"{r[f'win_{tag}']}/{r[f'tie_{tag}']}/{r[f'loss_{tag}']}")
            cells.append("-" if d is None else f"{d:+.3f}")
        lines.append(f"| {r['set']} | {r['instances']} | " + " | ".join(cells) + " |")
    return f"{ours} vs {ref} (mean diff = ({ours} - {ref}) / |{ref}|, negative is better for {ours})\n\n" + "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="scsp")
    ap.add_argument("--ref", default="highs")
    ap.add_argument("--times", nargs="+", type=float, default=[5, 10, 30])
    ap.add_argument("--sets", nargs="*", default=list(SC_SETS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--out", default="", help="path prefix for <out>_by_instance.csv, <out>_by_set.csv, <out>.md")
    args = ap.parse_args(argv)

    root = bench_dir()
    sets = {i.name: i.set for i in discover()}
    run_rows = per_run_rows(load_all_runs(), load_bks(root / "bks.csv"), sets, times=tuple(args.times))
    rows = compare(run_rows, args.ours, args.ref, args.times, set(args.sets), args.threads, int(args.warm_start))
    summary = summarize(rows, args.times)
    text = markdown(summary, args.ours, args.ref, args.times)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(Path(str(out) + "_by_instance.csv"), rows)
        write_csv(Path(str(out) + "_by_set.csv"), summary)
        Path(str(out) + ".md").write_text(text)


if __name__ == "__main__":
    main()
