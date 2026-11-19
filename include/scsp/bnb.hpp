// LP-based branch and bound for 0-1 problems (P3).
//
// Sources (public literature only): Land & Doig style LP branch and bound with best-bound
// node selection and depth-first plunging; reduced cost fixing (Beasley 1987; Atamturk,
// Nemhauser & Savelsbergh 2000); reliability branching on pseudocosts with strong branching
// (Achterberg, Koch & Martin 2005); Ryan-Foster branching for set partitioning rows
// (Ryan & Foster 1981).
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

#include "scsp/lp.hpp"

namespace scsp {

// Cooperative parallel tree handle (P5): lets external threads (e.g. idle heuristic/LNS workers)
// temporarily join an active branch-and-bound tree and leave cleanly when the tree ends or
// `stay()` returns false.
class BnbTeam {
 public:
  bool active() const { return is_active_.load(std::memory_order_acquire); }
  bool help(const std::function<bool()>& stay);

  // Used internally by branch_and_bound.
  void activate(std::function<void(const std::function<bool()>&)> fn);
  void deactivate();

 private:
  mutable std::mutex mu_;
  std::condition_variable done_cv_;
  std::function<void(const std::function<bool()>&)> fn_;
  std::atomic<bool> is_active_{false};
  int in_flight_ = 0;
};

// How the search talks to the rest of the solver. Costs exclude LpProblem::offset.
struct BnbHooks {
  std::function<double()> thr;                                    // a solution counts only if cost <= thr() (inf: none yet)
  std::function<void(const std::vector<int>&, double)> offer;     // columns at one, cost
  std::function<void(double)> offer_lb;                           // proven lower bound on every solution that still counts
  std::function<bool()> stop;
  std::function<void()> yield;   // optional: called between nodes; may run other work on this thread
  // optional: called at every node with the tree's lower bound (raw LP value, not rounded) and the node count;
  // returning false gives the search up (BnbResult::gave_up)
  std::function<bool(double, long)> keep;
  // optional: called every few nodes with the node LP solution (loaded columns and their values)
  std::function<void(const std::vector<int>&, const std::vector<double>&)> lp_point;
  // optional: the columns of the incumbent (needed by BnbOptions::rins)
  std::function<void(std::vector<int>&)> incumbent;
  // optional: called once with the root LP (row duals, reduced costs of all columns, value)
  std::function<void(const std::vector<double>&, const std::vector<double>&, double)> root;
  // optional: external worker threads may call team->help(...) to join the active tree (P5)
  BnbTeam* team = nullptr;
};

struct BnbOptions {
  unsigned seed = 1;
  bool integral = false;       // all costs integral: bounds are rounded up
  double time_limit = 1e30;    // seconds from the call
  int strong_cands = 8;        // candidates tried by strong branching per node
  long strong_iters = 100;     // dual simplex iterations per strong branching LP
  int reliability = 2;         // pseudocost observations before a variable is trusted
  int strong_lookahead = 4;    // candidates in a row without a better score before strong branching stops
  double strong_share = 1.0 / 3.0;  // share of all LP iterations strong branching may use below depth 8
  bool node_basis = true;      // a queued node keeps its parent's optimal basis
  bool node_fixing = true;     // reduced cost fixing at every node (for the rest of the plunge)
  // cutting planes at the root of an exact search (cuts.hpp): 1 = zero-half on covering sides,
  // 2 = zero-half on packing sides (odd holes), 4 = cliques
  int cuts = 0;
  int cut_rounds = 30;
  int cut_size = 15;           // largest row set of a zero-half cut
  double cut_time = 0.1;       // share of the time limit the cut rounds may use
  // RINS inside the tree (Danna, Rothberg & Le Pape 2005): now and then the columns on which the node LP
  // and the incumbent agree are fixed and the rest is solved as a small problem of its own
  bool rins = false;
  bool ryan_foster = false;    // optional Ryan-Foster branching for pure partitioning; default off (reliability var branching is stronger)
  long node_limit = -1;
  double root_time_limit = 1e30;  // seconds for the root LP; the search is given up if it takes longer
  int threads = 1;                // internal parallel tree threads (1 = caller only; > 1 spawns helpers)
};

struct BnbResult {
  // exhausted: the tree is finished and every column that could matter was in it (the incumbent, if any, is optimal;
  //            without an incumbent the problem is infeasible)
  // stopped:   time, node limit or stop request
  // failed:    the root LP could not be solved, or an LP failure left the tree incomplete
  enum Status { EXHAUSTED, STOPPED, FAILED, LP_INFEASIBLE } status = STOPPED;
  double root_lp = 0.0, root_time = 0.0;
  long nodes = 0, lp_iters = 0, strong_lps = 0, restarts = 0, dives = 0, dive_nodes = 0, pump_iters = 0, pump_found = 0, cover_nodes = 0;
  long node_fixed = 0;  // columns fixed by reduced cost at nodes (counted every time)
  int lp_cols = 0;
  bool exact = false;
  bool gave_up = false;        // stopped by BnbHooks::keep or by the root LP time limit
  double root_cut = 0.0;       // LP value after the cut rounds of the last exact search (root_lp if there were none)
  int cuts = 0, cut_rounds = 0;  // cuts left in the LP after the rounds; rounds run
  double cut_time = 0.0;
  long rins_calls = 0, rins_found = 0;
  double rins_time = 0.0;
};

// The problem that is left when some columns are fixed at one and all but a few of the others at zero
// (neighbourhood searches solve it with branch_and_bound again).
struct SubProblem {
  LpProblem P;
  std::vector<int> col_of;  // column of the sub-problem -> column of the full problem
  double fixed_cost = 0.0;
  bool ok = false;          // false: a row is violated by the fixed columns and no free column touches it
};
// `ones` are fixed at one, `freec` stay free, every other column is zero. With `nonneg` (no negative
// coefficient anywhere) a free column that would push a row past its upper side is left out.
SubProblem make_sub(const LpProblem& L, const std::vector<int>& freec, const std::vector<int>& ones, bool nonneg);

// P: every column is binary with bounds [0, 1].
BnbResult branch_and_bound(const LpProblem& P, const BnbHooks& H, const BnbOptions& opt);

}  // namespace scsp
