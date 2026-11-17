"""Best-known-solution table.

    python -m scsp.bench.bks          # merges benchmarks/bks_literature.csv with the run results
                                      # into benchmarks/bks.csv

Rules, per instance:
  * a run that proved optimality (status "optimal", solution re-checked) fixes
    the value: status "optimal", `verified_by` names the solver(s);
  * otherwise the smaller of the literature value and the best objective of any
    verified run is used, status "best_known" unless the literature says the
    value is a proven optimum;
  * `note` flags every disagreement between literature and runs.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict

from .catalog import bench_dir, discover
from .metrics import load_all_runs

FIELDS = ["instance", "set", "bks", "status", "source", "verified_by", "note"]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def build(root=None) -> list:
    root = root or bench_dir()
    lit = {}
    lit_path = root / "bks_literature.csv"
    if lit_path.exists():
        with open(lit_path, newline="") as f:
            lit = {r["instance"]: r for r in csv.DictReader(f)}

    proved = defaultdict(dict)       # instance -> solver label -> obj
    best = {}                        # instance -> (obj, solver label)
    for r in load_all_runs(root):
        if r.get("obj") is None or not r.get("verified"):
            continue
        label = f"{r['solver']} {r.get('solver_version', '')}".strip()
        if r["status"] == "optimal":
            proved[r["instance"]][label] = r["obj"]
        if r["instance"] not in best or r["obj"] < best[r["instance"]][0]:
            best[r["instance"]] = (r["obj"], label)

    rows = []
    for inst in discover(root):
        name = inst.name
        ref = lit.get(name)
        lit_val = float(ref["value"]) if ref else None
        notes = []
        row = {"instance": name, "set": inst.set, "bks": "", "status": "", "source": "", "verified_by": "", "note": ""}
        if proved.get(name):
            vals = sorted(set(round(v, 6) for v in proved[name].values()))
            val = min(proved[name].values())
            if len(vals) > 1 and not all(_close(v, val) for v in vals):
                notes.append(f"solvers disagree on the optimum: {proved[name]}")
            row.update(bks=val, status="optimal", verified_by="; ".join(sorted(proved[name])))
            row["source"] = ref["source"] if ref and _close(lit_val, val) else "proved in the benchmark runs"
            if ref and not _close(lit_val, val):
                notes.append(f"literature value {lit_val:g} ({ref['source']}) differs from the proven optimum")
        elif ref or name in best:
            run_val, run_label = best.get(name, (math.inf, ""))
            if ref and lit_val <= run_val or (ref and _close(lit_val, run_val)):
                row.update(bks=lit_val, status=ref["status"], source=ref["source"])
                if name in best and not _close(lit_val, run_val):
                    notes.append(f"best baseline run {run_val:g} ({run_label})")
            else:
                row.update(bks=run_val, status="best_known", source=f"best of the benchmark runs ({run_label})")
                if ref:
                    notes.append(f"better than the literature value {lit_val:g} ({ref['source']}) - check")
        else:
            notes.append("no value yet")
        if ref and ref.get("note"):
            notes.append(ref["note"])
        if row["bks"] != "":
            row["bks"] = f"{row['bks']:.10g}"
        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows


def main() -> None:
    root = bench_dir()
    rows = build(root)
    with open(root / "bks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    n_opt = sum(r["status"] == "optimal" for r in rows)
    n_ver = sum(bool(r["verified_by"]) for r in rows)
    flagged = [r for r in rows if "differs" in r["note"] or "disagree" in r["note"] or "check" in r["note"] or r["bks"] == ""]
    print(f"wrote {len(rows)} rows to {root / 'bks.csv'}: {n_opt} optimal ({n_ver} proved here), "
          f"{len(rows) - n_opt} best-known/open, {len(flagged)} flagged")
    for r in flagged:
        print("  FLAG", r["instance"], r["bks"], "|", r["note"])


if __name__ == "__main__":
    main()
