"""MPS reader for 0/1 models (air04, air05, nw04 and similar).

Reads the usual sections (NAME, OBJSENSE, ROWS, COLUMNS with INTORG/INTEND
markers, RHS, RANGES, BOUNDS, ENDATA). Fields are split on whitespace, so
names must not contain blanks (true for every instance we use).

`read_mps` returns the raw data; `mps_to_model` turns it into a `Model`:
  * every variable must be binary (integer with bounds [0, 1]); otherwise
    NotASetModel is raised. An integer variable without a BOUNDS entry is taken
    as binary (classic MPS convention) and counted in meta["implied_binary"].
  * a row whose coefficients are all 1 and whose right-hand side is 1 becomes a
    set row (>=, =, <=); every other row becomes a side constraint.
  * a ranged row becomes two side constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model import COVER, PACK, PARTITION, Model
from ._text import instance_name, open_bytes


class NotASetModel(ValueError):
    """The MPS file is valid but is not a pure 0/1 model."""


@dataclass
class MpsData:
    name: str = ""
    maximize: bool = False
    row_names: list = field(default_factory=list)       # constraint rows, in file order
    row_sense: list = field(default_factory=list)       # "L" / "G" / "E"
    obj_name: str | None = None
    col_names: list = field(default_factory=list)
    is_integer: list = field(default_factory=list)
    obj: dict = field(default_factory=dict)             # col -> coefficient
    entries: list = field(default_factory=list)         # (row, col, value)
    rhs: dict = field(default_factory=dict)             # row -> value
    obj_rhs: float = 0.0                                # RHS given for the objective row
    ranges: dict = field(default_factory=dict)          # row -> value
    lower: dict = field(default_factory=dict)           # col -> value (default 0)
    upper: dict = field(default_factory=dict)           # col -> value (default +inf)
    bounded: set = field(default_factory=set)           # columns that appear in BOUNDS


_SECTIONS = {"NAME", "OBJSENSE", "OBJSENSE:", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA",
             "OBJSENSE_MAX", "MAX", "MIN", "MAXIMIZE", "MINIMIZE"}
_NO_VALUE_BOUNDS = {"FR", "MI", "PL", "BV"}


def read_mps(path, name: str | None = None) -> MpsData:
    text = open_bytes(path).decode("ascii", errors="replace")
    d = MpsData(name=name or instance_name(path))
    row_id: dict = {}
    col_id: dict = {}
    free_rows: set = set()
    section = None
    in_int = False

    for lineno, line in enumerate(text.splitlines(), 1):
        if not line or line[0] == "*" or not line.strip():
            continue
        f = line.split()
        if line[0] not in " \t":                         # section header
            head = f[0].upper()
            if head == "NAME":
                if name is None and len(f) > 1:
                    d.name = f[1]
                section = "NAME"
            elif head in ("OBJSENSE", "OBJSENSE:"):
                section = "OBJSENSE"
                if len(f) > 1:
                    d.maximize = f[1].upper().startswith("MAX")
            elif head in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS"):
                section = head
            elif head == "ENDATA":
                break
            else:
                raise ValueError(f"{d.name}:{lineno}: unsupported MPS section {f[0]!r}")
            continue

        if section == "OBJSENSE":
            d.maximize = f[0].upper().startswith("MAX")
        elif section == "ROWS":
            kind, rname = f[0].upper(), f[1]
            if kind == "N":
                if d.obj_name is None:
                    d.obj_name = rname
                else:
                    free_rows.add(rname)
            elif kind in ("L", "G", "E"):
                row_id[rname] = len(d.row_names)
                d.row_names.append(rname)
                d.row_sense.append(kind)
            else:
                raise ValueError(f"{d.name}:{lineno}: bad row type {f[0]!r}")
        elif section == "COLUMNS":
            if len(f) >= 3 and f[1].strip("'").upper() == "MARKER":
                in_int = f[2].strip("'").upper() == "INTORG"
                continue
            cname = f[0]
            j = col_id.get(cname)
            if j is None:
                j = col_id[cname] = len(d.col_names)
                d.col_names.append(cname)
                d.is_integer.append(in_int)
            if len(f) not in (3, 5):
                raise ValueError(f"{d.name}:{lineno}: bad COLUMNS line")
            for k in range(1, len(f), 2):
                rname, val = f[k], float(f[k + 1])
                if rname == d.obj_name:
                    d.obj[j] = d.obj.get(j, 0.0) + val
                elif rname in row_id:
                    d.entries.append((row_id[rname], j, val))
                elif rname not in free_rows:
                    raise ValueError(f"{d.name}:{lineno}: unknown row {rname!r}")
        elif section in ("RHS", "RANGES"):
            start = len(f) % 2                           # odd count: first field is the set name
            for k in range(start, len(f), 2):
                rname, val = f[k], float(f[k + 1])
                if section == "RHS" and rname == d.obj_name:
                    d.obj_rhs = val
                elif rname in row_id:
                    (d.rhs if section == "RHS" else d.ranges)[row_id[rname]] = val
                elif rname not in free_rows:
                    raise ValueError(f"{d.name}:{lineno}: unknown row {rname!r}")
        elif section == "BOUNDS":
            kind = f[0].upper()
            need = 0 if kind in _NO_VALUE_BOUNDS else 1
            rest = f[1:]
            if len(rest) == need + 2:                    # set name present
                rest = rest[1:]
            elif len(rest) != need + 1:
                raise ValueError(f"{d.name}:{lineno}: bad BOUNDS line")
            if rest[0] not in col_id:
                raise ValueError(f"{d.name}:{lineno}: unknown column {rest[0]!r}")
            j = col_id[rest[0]]
            val = float(rest[1]) if need else 0.0
            d.bounded.add(j)
            if kind == "UP":
                d.upper[j] = val
                if val < 0 and j not in d.lower:
                    d.lower[j] = -np.inf
            elif kind == "LO":
                d.lower[j] = val
            elif kind == "FX":
                d.lower[j] = d.upper[j] = val
            elif kind == "FR":
                d.lower[j], d.upper[j] = -np.inf, np.inf
            elif kind == "MI":
                d.lower[j] = -np.inf
            elif kind == "PL":
                d.upper[j] = np.inf
            elif kind == "BV":
                d.lower[j], d.upper[j] = 0.0, 1.0
                d.is_integer[j] = True
            elif kind in ("LI", "UI"):
                d.is_integer[j] = True
                (d.lower if kind == "LI" else d.upper)[j] = val
            else:
                raise ValueError(f"{d.name}:{lineno}: bad bound type {f[0]!r}")
        else:
            raise ValueError(f"{d.name}:{lineno}: data line outside a section")

    if d.obj_name is None:
        raise ValueError(f"{d.name}: no objective (N) row")
    return d


def mps_to_model(d: MpsData) -> Model:
    n, m = len(d.col_names), len(d.row_names)
    if d.maximize:
        raise NotASetModel(f"{d.name}: maximization is not supported")

    implied = 0
    fixed = []                                           # (col, value)
    for j in range(n):
        if not d.is_integer[j]:
            raise NotASetModel(f"{d.name}: column {d.col_names[j]} is continuous")
        lo = d.lower.get(j, 0.0)
        up = d.upper.get(j, 1.0 if j not in d.bounded else np.inf)
        if j not in d.bounded:
            implied += 1
        if lo < 0 or up > 1 or lo > up or lo != round(lo) or up != round(up):
            raise NotASetModel(f"{d.name}: column {d.col_names[j]} has bounds [{lo}, {up}], not binary")
        if lo == up:
            fixed.append((j, lo))

    cost = np.zeros(n)
    for j, v in d.obj.items():
        cost[j] = v

    ent = np.asarray(d.entries, dtype=np.float64).reshape(-1, 3)
    row = ent[:, 0].astype(np.int64)
    col = ent[:, 1].astype(np.int64)
    val = ent[:, 2]
    keep = val != 0
    row, col, val = row[keep], col[keep], val[keep]
    order = np.lexsort((col, row))
    row, col, val = row[order], col[order], val[order]
    if row.size and np.any((row[1:] == row[:-1]) & (col[1:] == col[:-1])):
        raise ValueError(f"{d.name}: a column has two entries in the same row")
    ptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(row, minlength=m), out=ptr[1:])

    rhs = np.array([d.rhs.get(i, 0.0) for i in range(m)])
    all_ones = np.ones(m, dtype=bool)
    np.logical_and.at(all_ones, row, val == 1.0)
    ranged = np.zeros(m, dtype=bool)
    ranged[list(d.ranges)] = True
    nonempty = np.diff(ptr) > 0
    is_set = all_ones & (rhs == 1.0) & ~ranged & nonempty

    sense_map = {"G": COVER, "E": PARTITION, "L": PACK}
    set_rows = np.flatnonzero(is_set)
    side_rows = np.flatnonzero(~is_set)
    set_len = (ptr[1:] - ptr[:-1])[set_rows]
    row_ptr = np.zeros(set_rows.size + 1, dtype=np.int64)
    np.cumsum(set_len, out=row_ptr[1:])
    col_idx = col[is_set[row]].astype(np.int32)

    side = []
    for i in side_rows:
        idx, coef = col[ptr[i]:ptr[i + 1]], val[ptr[i]:ptr[i + 1]]
        s, b = d.row_sense[i], rhs[i]
        if i in d.ranges:
            r = d.ranges[i]
            if s == "G":
                lo, up = b, b + abs(r)
            elif s == "L":
                lo, up = b - abs(r), b
            else:
                lo, up = (b, b + r) if r >= 0 else (b + r, b)
            side.append((idx, coef, COVER, lo))
            side.append((idx, coef, PACK, up))
        else:
            side.append((idx, coef, sense_map[s], b))
    for j, v in fixed:
        side.append((np.array([j]), np.array([1.0]), PARTITION, v))

    side_ptr = np.zeros(len(side) + 1, dtype=np.int64)
    np.cumsum([len(s[0]) for s in side], out=side_ptr[1:])
    return Model(
        name=d.name, cost=cost, row_ptr=row_ptr, col_idx=col_idx,
        row_sense=np.array([sense_map[d.row_sense[i]] for i in set_rows], dtype="U1"),
        side_ptr=side_ptr,
        side_idx=np.concatenate([s[0] for s in side]).astype(np.int32) if side else np.zeros(0, dtype=np.int32),
        side_val=np.concatenate([s[1] for s in side]).astype(np.float64) if side else np.zeros(0),
        side_sense=np.array([s[2] for s in side], dtype="U1"),
        side_rhs=np.array([s[3] for s in side], dtype=np.float64),
        obj_offset=-d.obj_rhs,
        meta={"format": "mps", "implied_binary": implied, "fixed_columns": len(fixed), "mps_rows": m},
    )


def read_mps_model(path, name: str | None = None) -> Model:
    return mps_to_model(read_mps(path, name))
