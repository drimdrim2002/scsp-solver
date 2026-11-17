"""Instance readers. `read_instance` picks the reader from the format name."""
from __future__ import annotations

from ..model import Model
from .mps import MpsData, NotASetModel, mps_to_model, read_mps, read_mps_model
from .orlib import read_scp, read_scp_rail, read_spp

FORMATS = {
    "orlib-scp-row": read_scp,
    "orlib-scp-col": read_scp_rail,
    "orlib-spp": read_spp,
    "mps": read_mps_model,
    "npz": lambda path, name=None: Model.load_npz(path),
}


def read_instance(path, fmt: str, name: str | None = None) -> Model:
    try:
        reader = FORMATS[fmt]
    except KeyError:
        raise ValueError(f"unknown instance format {fmt!r}; known: {sorted(FORMATS)}") from None
    return reader(path, name)


__all__ = ["FORMATS", "MpsData", "NotASetModel", "mps_to_model", "read_instance", "read_mps",
           "read_mps_model", "read_scp", "read_scp_rail", "read_spp"]
