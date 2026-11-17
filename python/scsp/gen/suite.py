"""The standard synthetic suite.

    python -m scsp.gen.suite            # write benchmarks/instances/synthetic/*.npz
                                        # and benchmarks/synthetic_manifest.csv

Instances are deterministic in (generator, parameters, seed). The manifest
records a SHA-256 of the model arrays so that a change in numpy's random
streams or in the generator code is noticed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time

from ..bench.catalog import bench_dir
from ..model import Model
from .dms import generate_dms
from .sp import generate_sp

# (orders K, bundles n, tight, seed). Gurobi's size-limited licence needs
# n <= 2000 and K + 3 <= 2000.
DMS_SUITE = [
    (50, 500, False, 1), (50, 2000, False, 1),
    (100, 1000, False, 1), (100, 2000, False, 1), (100, 2000, True, 1), (100, 2000, False, 2),
    (200, 2000, False, 1), (200, 2000, True, 1),
    (300, 2000, False, 1), (500, 2000, False, 1),
    (100, 5000, False, 1), (200, 10000, False, 1), (300, 20000, False, 1),
    (500, 30000, False, 1), (500, 30000, True, 1),
    (1000, 60000, False, 1), (2000, 60000, False, 1),
]
# (rows m, columns n, seed)
SP_SUITE = [
    (50, 500, 1), (100, 1000, 1), (100, 2000, 1), (200, 2000, 1), (200, 2000, 2),
    (300, 2000, 1), (400, 2000, 1), (300, 10000, 1),
]


def model_digest(m: Model) -> str:
    h = hashlib.sha256()
    for a in (m.cost, m.row_ptr, m.col_idx, m.side_ptr, m.side_idx, m.side_val, m.side_rhs):
        h.update(a.tobytes())
    h.update("".join(m.row_sense.tolist() + m.side_sense.tolist()).encode())
    return h.hexdigest()[:16]


def build_suite():
    for K, n, tight, seed in DMS_SUITE:
        yield "dms", dict(n_orders=K, n_bundles=n, tight=tight, seed=seed), lambda K=K, n=n, tight=tight, seed=seed: generate_dms(K, n, seed=seed, tight=tight)
    for m, n, seed in SP_SUITE:
        yield "sp", dict(n_rows=m, n_cols=n, seed=seed), lambda m=m, n=n, seed=seed: generate_sp(m, n, seed=seed)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="regenerate and compare digests with the manifest, write nothing")
    args = ap.parse_args(argv)

    root = bench_dir()
    out = root / "instances" / "synthetic"
    manifest = root / "synthetic_manifest.csv"
    old = {}
    if manifest.exists():
        with open(manifest, newline="") as f:
            old = {r["name"]: r["sha256_16"] for r in csv.DictReader(f)}
    if not args.check:
        out.mkdir(parents=True, exist_ok=True)

    rows, bad = [], 0
    for gen, params, make in build_suite():
        t0 = time.perf_counter()
        m = make()
        digest = model_digest(m)
        if args.check:
            status = "ok" if old.get(m.name) == digest else "MISMATCH"
            bad += status != "ok"
        else:
            m.save_npz(out / f"{m.name}.npz")
            status = "written"
        s = m.stats()
        rows.append({
            "name": m.name, "generator": gen, "params": ";".join(f"{k}={v}" for k, v in params.items()),
            "kind": s["kind"], "rows": s["rows"], "cols": s["cols"], "nnz": s["nnz"], "side_rows": s["side_rows"],
            "side_rhs": " ".join(f"{v:g}" for v in m.side_rhs), "warm_start_cols": len(m.meta["warm_start"]),
            "warm_start_obj": f"{m.meta['warm_start_obj']:.6f}", "sha256_16": digest,
        })
        print(f"{status:8s} {m.name:28s} {s['rows']:5d} x {s['cols']:6d} nnz={s['nnz']:7d} "
              f"warm={m.meta['warm_start_obj']:.2f}  {time.perf_counter() - t0:.1f}s", flush=True)
    if args.check:
        raise SystemExit(1 if bad else 0)
    with open(manifest, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} instances to {out} and {manifest}")


if __name__ == "__main__":
    main()
