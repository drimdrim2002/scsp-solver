"""One solver run in its own process; prints the RunResult as one JSON line.

    python -m scsp.bench.worker --config highs --instance scp41 --time-limit 10 --seed 1

A separate process per run keeps solver thread pools, memory and crashes
isolated, and lets the caller kill a run that ignores its time limit.
"""
from __future__ import annotations

import argparse
import json
import sys

from .catalog import find
from .runners import CONFIGS, run_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--instance", required=True)
    ap.add_argument("--time-limit", type=float, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warm-start", action="store_true", help="pass meta['warm_start'] as the MIP start")
    args = ap.parse_args(argv)

    model = find(args.instance).load()
    warm = model.meta.get("warm_start") if args.warm_start else None
    if args.warm_start and warm is None:
        raise SystemExit(f"{args.instance}: no warm start stored in the instance")
    res = run_config(args.config, model, args.time_limit, threads=args.threads, seed=args.seed, warm_start=warm)
    sys.stdout.write(json.dumps(res.to_dict()) + "\n")


if __name__ == "__main__":
    main()
