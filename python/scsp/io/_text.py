"""Shared helpers for the text readers."""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np


def open_bytes(path) -> bytes:
    """Whole file as bytes; .gz files are decompressed transparently."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def numeric_tokens(path) -> np.ndarray:
    """All whitespace-separated numbers of a text file as float64.

    np.fromstring stops at the first token it cannot parse, so the readers
    always check that the token count matches the structure they decoded.
    """
    data = open_bytes(path)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return np.fromstring(data.decode("ascii", errors="replace"), dtype=np.float64, sep=" ")


def instance_name(path) -> str:
    name = os.path.basename(str(path))
    for suffix in (".gz", ".txt", ".mps", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def to_index(values: np.ndarray, upper: int, what: str, name: str) -> np.ndarray:
    """1-based float tokens -> validated 0-based int64 indices."""
    idx = values.astype(np.int64)
    if not np.array_equal(idx, values):
        raise ValueError(f"{name}: non-integer {what} index")
    idx -= 1
    if idx.size and (idx.min() < 0 or idx.max() >= upper):
        raise ValueError(f"{name}: {what} index out of range 1..{upper}")
    return idx
