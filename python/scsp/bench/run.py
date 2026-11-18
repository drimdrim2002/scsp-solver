"""Run reference solvers over benchmark instances and append the results to
benchmarks/results/runs/<config>.jsonl (one JSON object per run).

    python -m scsp.bench.run --configs gurobi highs --sets scp4 scpe --time-limits 60 --seeds 1 2 3
    python -m scsp.bench.run --configs highs --instances scpnrh1 rail507 --time-limits 5 10 30 60

Runs are sequential (one solver at a time, so timings do not disturb each
other) and resumable: a (config, instance, time limit, seed, threads, warm
start) combination already present in the JSONL file is skipped.

`--skip-solved` drops a run when the same (config, instance, seed) was already
solved to proven optimality under a longer time limit in less time than the new
limit: the solver would do exactly the same work again.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .catalog import bench_dir, discover, gurobi_ok
from .runners import CONFIGS


def run_key(r: dict) -> tuple:
    return (r["config"], r["instance"], float(r["time_limit"]), int(r["seed"]), int(r["threads"]), bool(r["warm_start"]))


def load_runs(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def results_dir(root: Path | None = None) -> Path:
    return (root or bench_dir()) / "results" / "runs"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["gurobi", "highs"], choices=sorted(CONFIGS))
    ap.add_argument("--sets", nargs="*", default=[], help="instance sets (scp4, scpnre, sppnw, miplib, dms, sp, ...)")
    ap.add_argument("--instances", nargs="*", default=[], help="instance names")
    ap.add_argument("--time-limits", nargs="+", type=float, default=[5, 10, 30, 60])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--warm-kind", choices=["near", "indep"], default="near",
                    help="which stored start to pass (bench.warm); runs with 'indep' are recorded as <config>@indep")
    ap.add_argument("--skip-solved", action="store_true")
    ap.add_argument("--tag", default="", help="write to runs/<config><tag>.jsonl instead of runs/<config>.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    catalog = discover()
    chosen = [i for i in catalog if i.set in args.sets or i.name in args.instances]
    if not args.sets and not args.instances:
        chosen = catalog
    missing = set(args.instances) - {i.name for i in catalog}
    if missing:
        raise SystemExit(f"unknown instances: {sorted(missing)}")

    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    planned = done = 0
    t_start = time.time()
    for config in args.configs:
        label = config + ("@" + args.warm_kind if args.warm_start and args.warm_kind != "near" else "")
        out = out_dir / f"{label}{args.tag}.jsonl"
        runs = load_runs(out)
        seen = {run_key(r) for r in runs}
        solved = {}                                      # (instance, seed) -> fastest proven runtime
        for r in runs:
            if r["status"] == "optimal" and int(r["threads"]) == args.threads and bool(r["warm_start"]) == args.warm_start:
                k = (r["instance"], int(r["seed"]))
                solved[k] = min(solved.get(k, float("inf")), float(r["runtime"]))
        # longest limit first, so --skip-solved can use it for the shorter ones
        for tl in sorted(args.time_limits, reverse=True):
            for inst in chosen:
                if config.startswith("gurobi"):
                    if inst.name not in sizes:
                        m = inst.load()
                        sizes[inst.name] = (m.n_cols, m.n_cons)
                    if not gurobi_ok(*sizes[inst.name]):
                        continue
                for seed in args.seeds:
                    key = (label, inst.name, float(tl), seed, args.threads, args.warm_start)
                    if key in seen:
                        continue
                    if args.skip_solved and solved.get((inst.name, seed), float("inf")) < tl:
                        continue
                    planned += 1
                    if args.dry_run:
                        continue
                    cmd = [sys.executable, "-I", "-m", "scsp.bench.worker", "--config", config, "--instance", inst.name,
                           "--time-limit", str(tl), "--threads", str(args.threads), "--seed", str(seed)]
                    if args.warm_start:
                        cmd += ["--warm-start", "--warm-kind", args.warm_kind]
                    t0 = time.time()
                    try:
                        p = subprocess.run(cmd, capture_output=True, text=True, timeout=tl * 2 + 120)
                        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                        if p.returncode != 0 or not lines:
                            rec = {"config": label, "solver": config.split("-")[0], "instance": inst.name,
                                   "time_limit": tl, "threads": args.threads, "seed": seed,
                                   "warm_start": args.warm_start, "status": "error", "obj": None, "bound": None,
                                   "gap": None, "runtime": time.time() - t0, "trace": [],
                                   "note": (p.stderr or p.stdout)[-400:]}
                        else:
                            rec = json.loads(lines[-1])
                    except subprocess.TimeoutExpired:
                        rec = {"config": label, "solver": config.split("-")[0], "instance": inst.name,
                               "time_limit": tl, "threads": args.threads, "seed": seed, "warm_start": args.warm_start,
                               "status": "killed", "obj": None, "bound": None, "gap": None,
                               "runtime": time.time() - t0, "trace": [], "note": "worker exceeded 2*tl+120 s"}
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    seen.add(key)
                    if rec["status"] == "optimal":
                        k = (inst.name, seed)
                        solved[k] = min(solved.get(k, float("inf")), float(rec["runtime"]))
                    done += 1
                    obj = "-" if rec.get("obj") is None else f"{rec['obj']:.6g}"
                    bnd = "-" if rec.get("bound") is None else f"{rec['bound']:.6g}"
                    print(f"[{time.time() - t_start:7.0f}s] {config:9s} {inst.name:26s} tl={tl:<4g} s={seed} "
                          f"{rec['status']:10s} obj={obj:>12s} bound={bnd:>12s} t={rec['runtime']:.2f} "
                          f"ok={rec.get('verified')} {rec.get('note', '')[:80]}", flush=True)
    print(f"{'planned' if args.dry_run else 'finished'}: {planned if args.dry_run else done} runs "
          f"in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
