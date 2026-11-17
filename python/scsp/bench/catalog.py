"""Benchmark instance catalog: which files exist, which set they belong to and
how to read them.

The benchmark directory is `$SCSP_BENCH_DIR` if set, else `<repo>/benchmarks`
(found from this file's location, which works for an editable install).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..io import read_instance
from ..model import Model

# Gurobi size-limited licence: runs iff variables <= 2000 and constraints <= 2000
# on the original model (measured 2026-11-17: 2000 passes, 2001 fails).
GUROBI_MAX_VARS = 2000
GUROBI_MAX_CONS = 2000


def bench_dir() -> Path:
    env = os.environ.get("SCSP_BENCH_DIR")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[3]
    if (root / "benchmarks").is_dir():
        return root / "benchmarks"
    raise FileNotFoundError("benchmarks directory not found; set SCSP_BENCH_DIR")


@dataclass(frozen=True)
class Instance:
    name: str
    set: str          # scp4, scp5, scp6, scpa..scpe, scpnre..scpnrh, rail, sppnw, sppaa, sppus, sppkl, miplib, dms, sp
    fmt: str
    path: Path

    def load(self) -> Model:
        return read_instance(self.path, self.fmt, self.name)


def _strip(filename: str) -> str:
    for suffix in (".gz", ".txt", ".mps", ".npz"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
    return filename


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _scp_set(name: str) -> str:
    if name.startswith("rail"):
        return "rail"
    if name.startswith("scpnr"):
        return name[:6]                                  # scpnre .. scpnrh
    if name.startswith(("scpclr", "scpcyc")):
        return name[:6]
    m = re.fullmatch(r"scp([456])(\d+)", name)
    if m:
        return "scp" + m.group(1)
    return name[:4]                                      # scpa .. scpe


def discover(root: Path | None = None) -> list[Instance]:
    """All instances present on disk, in a stable order."""
    inst_dir = (root or bench_dir()) / "instances"
    found: dict[str, Instance] = {}

    def add(inst: Instance) -> None:
        old = found.get(inst.name)
        # prefer the uncompressed copy if both exist
        if old is None or (old.path.suffix == ".gz" and inst.path.suffix != ".gz"):
            found[inst.name] = inst

    for path in sorted((inst_dir / "orlib_scp").glob("*")):
        name = _strip(path.name)
        if path.suffix in (".txt", ".gz"):
            fmt = "orlib-scp-col" if name.startswith("rail") else "orlib-scp-row"
            add(Instance(name, _scp_set(name), fmt, path))
    for path in sorted((inst_dir / "orlib_spp").glob("*")):
        name = _strip(path.name)
        if path.suffix in (".txt", ".gz"):
            add(Instance(name, name[:5], "orlib-spp", path))
    for path in sorted((inst_dir / "miplib").glob("*.mps*")):
        add(Instance(_strip(path.name), "miplib", "mps", path))
    for path in sorted((inst_dir / "synthetic").glob("*.npz")):
        name = _strip(path.name)
        add(Instance(name, name.split("_", 1)[0], "npz", path))

    order = ["scp4", "scp5", "scp6", "scpa", "scpb", "scpc", "scpd", "scpe", "scpnre", "scpnrf", "scpnrg",
             "scpnrh", "rail", "sppnw", "sppaa", "sppus", "sppkl", "miplib", "dms", "sp"]
    rank = {s: i for i, s in enumerate(order)}
    return sorted(found.values(), key=lambda i: (rank.get(i.set, len(order)), _natural(i.name)))


def find(name: str, root: Path | None = None) -> Instance:
    for inst in discover(root):
        if inst.name == name:
            return inst
    raise KeyError(f"instance {name!r} not found under {(root or bench_dir()) / 'instances'}")


def gurobi_ok(n_cols: int, n_cons: int) -> bool:
    return n_cols <= GUROBI_MAX_VARS and n_cons <= GUROBI_MAX_CONS
