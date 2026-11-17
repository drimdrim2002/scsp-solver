"""scsp: set covering / set partitioning MIP solver.

State: P0 harness (model, readers, generators, reference-solver benchmark) and
the P1 set covering heuristic (`scsp.solve`, `scsp.dms.solveSP`) and the P2 LP engine
(`scsp.solve_lp`).
"""
from ._core import __version__, build_info, hardware_threads, parallel_sum, version
from .lp import LpSolver, LPResult, solve_lp
from .model import Model
from .solve import Result, solve

__all__ = ["Model", "Result", "LPResult", "LpSolver", "solve", "solve_lp", "__version__", "build_info", "hardware_threads", "parallel_sum", "version"]
