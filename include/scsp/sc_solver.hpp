// Set covering solver: the P1 heuristic (Lagrangian three-phase engine, local search,
// packing side constraints, enumeration with Lagrangian bounds) plus, from P3 on, one
// thread that solves the root LP and runs LP-based branch and bound for the proof.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "scsp/problem.hpp"

namespace scsp {

struct SolveOptions {
  double time_limit = 10.0;
  int threads = 4;
  std::uint64_t seed = 1;
  double mip_gap = 0.0;      // stop when (obj - bound) <= mip_gap * |obj|
  bool presolve = true;
  bool local_search = true;
  bool enumerate = true;
  bool bnb = true;           // one thread runs the root LP and LP-based branch and bound (P3)
};

struct TracePoint {
  double t;      // seconds since the start of solve_sc
  double obj;    // incumbent objective (NaN if none yet)
  double bound;  // lower bound (NaN if none yet)
  int kind;      // 0 = new incumbent, 1 = bound sample
};

struct SolveResult {
  std::string status;        // optimal | time_limit | no_solution | infeasible
  std::vector<int> cols;     // original columns at 1
  double obj = 0.0;
  double bound = 0.0;        // NaN if no bound
  double runtime = 0.0;
  std::vector<TracePoint> trace;
  // statistics
  int red_cols = 0, red_rows = 0, fixed_cols = 0;
  double presolve_time = 0.0;
  long long sub_iters = 0, greedy_calls = 0, ls_moves = 0, enum_nodes = 0, passes = 0;
  bool warm_used = false;
  // LP-based branch and bound (P3)
  std::string bnb_status;    // "" (not run) | exhausted | stopped | failed | lp_infeasible
  double lp_root = 0.0, lp_root_time = 0.0;   // root LP value of the reduced problem (NaN if not solved)
  long long bnb_nodes = 0, bnb_lp_iters = 0, bnb_strong = 0;
  int bnb_cols = 0;
};

// `warm`: optional feasible solution (original column indices). The returned
// solution is never worse than it.
SolveResult solve_sc(const Problem& p, const SolveOptions& opt, const std::vector<int>* warm = nullptr);

}  // namespace scsp
