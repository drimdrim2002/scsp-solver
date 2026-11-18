"""One solver run in its own process; prints the RunResult as one JSON line.

    python -m scsp.bench.worker --config highs --instance scp41 --time-limit 10 --seed 1

A separate process per run keeps solver thread pools, memory and crashes
isolated, and lets the caller kill a run that ignores its time limit.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .catalog import find
from .runners import CONFIGS, run_config
from .warm import FILES, stored


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--instance", required=True)
    ap.add_argument("--time-limit", type=float, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warm-start", action="store_true", help="pass the stored warm start (bench.warm, else meta['warm_start']) as the MIP start")
    ap.add_argument("--warm-kind", choices=sorted(FILES), default="near", help="which stored start (see bench.warm)")
    args = ap.parse_args(argv)

    model = find(args.instance).load()
    warm = source = None
    if args.warm_start:
        warm, source = stored(args.instance, kind=args.warm_kind), FILES[args.warm_kind]   # a good start (bench.warm)
        if warm is None and args.warm_kind == "near":
            warm, source = model.meta.get("warm_start"), "meta"       # the generator's hidden solution
        if warm is None:
            raise SystemExit(f"{args.instance}: no warm start stored for the instance")
    res = run_config(args.config, model, args.time_limit, threads=args.threads, seed=args.seed, warm_start=warm)
    rec = res.to_dict()
    if warm is not None:
        x0 = np.zeros(model.n_cols)
        x0[np.asarray(warm, dtype=np.int64)] = 1.0
        rec["warm_obj"], rec["warm_source"] = model.objective(x0), source
        if args.warm_kind != "near":                     # a different start is a different experiment
            rec["config"] += "@" + args.warm_kind
    sys.stdout.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
