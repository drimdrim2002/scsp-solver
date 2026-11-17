// LP-based branch and bound for 0-1 problems (P3).
//
// Sources (public literature only): Land & Doig style LP branch and bound with best-bound
// node selection and depth-first plunging; reduced cost fixing (Beasley 1987; Atamturk,
// Nemhauser & Savelsbergh 2000); reliability branching on pseudocosts with strong branching
// (Achterberg, Koch & Martin 2005); Ryan-Foster branching for set partitioning rows
// (Ryan & Foster 1981).
#pragma once

#include <functional>
#include <vector>

#include "scsp/lp.hpp"

namespace scsp {

// How the search talks to the rest of the solver. Costs exclude LpProblem::offset.
struct BnbHooks {
  std::function<double()> thr;                                    // a solution counts only if cost <= thr() (inf: none yet)
  std::function<void(const std::vector<int>&, double)> offer;     // columns at one, cost
  std::function<void(double)> offer_lb;                           // proven lower bound on every solution that still counts
  std::function<bool()> stop;
  std::function<void()> yield;   // optional: called between nodes; may run other work on this thread
  // optional: called once with the root LP (row duals, reduced costs of all columns, value)
  std::function<void(const std::vector<double>&, const std::vector<double>&, double)> root;
};

struct BnbOptions {
  unsigned seed = 1;
  bool integral = false;       // all costs integral: bounds are rounded up
  double time_limit = 1e30;    // seconds from the call
  int strong_cands = 8;        // candidates tried by strong branching per node
  long strong_iters = 100;     // dual simplex iterations per strong branching LP
  int reliability = 2;         // pseudocost observations before a variable is trusted
  bool ryan_foster = true;     // use Ryan-Foster branching when every row is an equality with rhs 1
  long node_limit = -1;
};

struct BnbResult {
  // exhausted: the tree is finished and every column that could matter was in it (the incumbent, if any, is optimal;
  //            without an incumbent the problem is infeasible)
  // stopped:   time, node limit or stop request
  // failed:    the root LP could not be solved, or an LP failure left the tree incomplete
  enum Status { EXHAUSTED, STOPPED, FAILED, LP_INFEASIBLE } status = STOPPED;
  double root_lp = 0.0, root_time = 0.0;
  long nodes = 0, lp_iters = 0, strong_lps = 0, restarts = 0, dives = 0, dive_nodes = 0, pump_iters = 0, pump_found = 0, cover_nodes = 0;
  int lp_cols = 0;
  bool exact = false;
};

// P: every column is binary with bounds [0, 1].
BnbResult branch_and_bound(const LpProblem& P, const BnbHooks& H, const BnbOptions& opt);

}  // namespace scsp
