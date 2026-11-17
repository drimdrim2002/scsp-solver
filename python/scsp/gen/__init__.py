"""Synthetic instance generators."""
from .dms import DmsParams, generate_dms
from .sp import SpParams, generate_sp

__all__ = ["DmsParams", "SpParams", "generate_dms", "generate_sp"]
