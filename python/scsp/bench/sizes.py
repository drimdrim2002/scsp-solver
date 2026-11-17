"""Instance size table, measured from the files on disk.

    python -m scsp.bench.sizes        # writes benchmarks/instance_sizes.csv
"""
from __future__ import annotations

import csv

from .catalog import bench_dir, discover, gurobi_ok

FIELDS = ["instance", "set", "format", "file", "kind", "rows", "cols", "nnz", "density_pct", "side_rows",
          "row_len_min", "row_len_max", "col_len_min", "col_len_max", "cost_min", "cost_max", "cost_integral",
          "duplicates_removed", "gurobi_ok"]


def size_rows(root=None) -> list:
    rows = []
    for inst in discover(root):
        m = inst.load()
        s = m.stats()
        rows.append({
            "instance": inst.name, "set": inst.set, "format": inst.fmt, "file": inst.path.name, "kind": s["kind"],
            "rows": s["rows"], "cols": s["cols"], "nnz": s["nnz"], "density_pct": f"{100 * s['density']:.3f}",
            "side_rows": s["side_rows"], "row_len_min": s["row_len_min"], "row_len_max": s["row_len_max"],
            "col_len_min": s["col_len_min"], "col_len_max": s["col_len_max"],
            "cost_min": f"{s['cost_min']:g}", "cost_max": f"{s['cost_max']:g}", "cost_integral": int(s["cost_integral"]),
            "duplicates_removed": m.meta.get("duplicates_removed", 0),
            "gurobi_ok": int(gurobi_ok(m.n_cols, m.n_cons)),
        })
    return rows


def main() -> None:
    root = bench_dir()
    rows = size_rows(root)
    with open(root / "instance_sizes.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {root / 'instance_sizes.csv'}; "
          f"{sum(r['gurobi_ok'] for r in rows)} fit the Gurobi size limit")


if __name__ == "__main__":
    main()
