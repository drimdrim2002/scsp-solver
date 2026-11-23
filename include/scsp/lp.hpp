// LP engine (P2): bounded dual simplex, primal simplex and sifting.
//
//   min  cost' x + offset
//   s.t. row_lo <= A x <= row_up,   col_lb <= x <= col_ub
//
// Sources (public literature only): Koberstein (2005) for the bounded dual simplex
// (dual steepest edge, bound flipping ratio test, cost perturbation and shifting);
// Forrest & Goldfarb (1992) for the steepest edge update; Harris (1973) for the
// tolerance ratio test and Devex pricing; Bixby, Gregory, Lustig, Marsten & Shanno
// (1992) for sifting.
#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace scsp {

struct LpProblem {
  int n = 0, m = 0;
  std::vector<double> cost, lb, ub;  // columns
  std::vector<double> rlo, rup;      // rows
  std::vector<int> cptr, cidx;       // column-wise matrix
  std::vector<double> cval;
  double offset = 0.0;

  // Builds the column-wise matrix from row-wise input.
  static LpProblem from_csr(int n, int m, const double* cost, const double* lb, const double* ub, const double* rlo,
                            const double* rup, const std::int64_t* row_ptr, const std::int32_t* col_idx,
                            const double* val, double offset);
};

struct LpOptions {
  std::string method = "auto";  // auto | dual | sifting
  double time_limit = std::numeric_limits<double>::infinity();
  long max_iter = std::numeric_limits<long>::max();
  double feas_tol = 1e-9;
  double opt_tol = 1e-9;
  int refactor = 100;   // basis updates between refactorizations
  bool perturb = true;  // cost perturbation in the dual simplex
  unsigned seed = 1;
  int sift_cap = 0;     // columns added per sifting round (0 = automatic)
  const std::atomic<bool>* stop = nullptr;  // optional early-stop flag checked inside simplex
};

struct LpResult {
  // optimal | infeasible | unbounded | time_limit | iteration_limit | failed
  std::string status = "failed";
  double obj = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> x;         // n
  std::vector<double> row_dual;  // m
  std::vector<double> red_cost;  // n
  std::string method;
  long dual_iters = 0, primal_iters = 0, refactors = 0, flips = 0;
  int sift_rounds = 0, work_cols = 0;
  double max_primal_infeas = 0.0, max_dual_infeas = 0.0;
  double runtime = 0.0;
};

LpResult solve_lp(const LpProblem& p, const LpOptions& opt);

// An LP that stays loaded: change bounds, solve again from the last basis with the
// dual simplex (the basis stays dual feasible under bound changes), save and restore
// a basis. Used by branch and bound.
class LpSolver {
 public:
  enum Status { OPTIMAL = 0, INFEASIBLE = 1, UNBOUNDED = 2, LIMIT = 3, FAILED = 4, CUTOFF = 5 };

  // Loads the columns `cols` of p (all of them when empty). Column index k below is the position in `cols`.
  LpSolver(const LpProblem& p, const std::vector<int>& cols, const LpOptions& opt);
  ~LpSolver();
  LpSolver(const LpSolver&) = delete;
  LpSolver& operator=(const LpSolver&) = delete;

  // Independent copy of the current loaded LP (including added/removed rows, column bounds,
  // and optimal basis/weights) for parallel tree workers (P5).
  std::unique_ptr<LpSolver> clone(unsigned seed, double time_limit,
                                  const std::atomic<bool>* stop = nullptr) const;

  // Stops with CUTOFF as soon as the dual objective exceeds the cutoff (a proven lower bound of the LP value),
  // with LIMIT after max_iter dual iterations (-1 = no limit). solve() uses the cutoff of set_cutoff().
  Status solve();
  Status solve(double cutoff, long max_iter = -1);
  void set_cutoff(double cutoff);               // infinity switches it off
  double cutoff() const;
  void set_col_bounds(int k, double lb, double ub);
  // Cutting planes (P4). Rows are given row-wise over the loaded columns. A new row starts with its row
  // variable basic, so the next solve() continues with the dual simplex from the last basis. Removing rows
  // keeps the basis when every removed row has its row variable basic (see row_status), else the next
  // solve() starts from scratch. Both throw std::invalid_argument on bad input.
  void add_rows(const std::vector<int>& row_ptr, const std::vector<int>& col_idx, const std::vector<double>& val,
                const std::vector<double>& row_lo, const std::vector<double>& row_up);
  void remove_rows(const std::vector<int>& rows);
  void remove_rows(int first_row);              // this row and every later one
  void get_row_activity(std::vector<double>& a) const;   // A x of the last solve
  std::vector<char> row_status() const;         // 0 = row variable basic (the row is not tight), else at a side
  void set_col_cost(int k, double cost);
  std::vector<char> get_basis() const;          // one status per row, then per column
  bool set_basis(const std::vector<char>& basis);
  // A basis together with the pricing weights (and factored LU snapshot when valid) that go with it;
  // restoring it is cheaper to continue from than set_basis alone (strong branching returns to the node this way).
  struct Snapshot;
  struct State {
    std::vector<char> basis;
    std::vector<double> weights;
    std::shared_ptr<const Snapshot> snap;
  };
  void save_state(State& s) const;
  bool restore_state(const State& s);
  double obj() const;                           // objective of the last solve, offset included
  void get_x(std::vector<double>& x) const;
  void get_red_cost(std::vector<double>& d) const;
  bool get_row_dual(std::vector<double>& y);    // refactorizes
  // Tableau / row inspection for Gomory mixed-integer (GMI) cut generation (P6).
  bool is_col_basic(int k) const;
  bool get_tableau_row(int k, std::vector<int>& nb_var, std::vector<double>& nb_val,
                       std::vector<char>& nb_st, std::vector<double>& nb_lo,
                       std::vector<double>& nb_up);
  void get_row(int i, std::vector<int>& col_idx, std::vector<double>& col_val) const;
  int cols() const;
  int rows() const;
  long iterations() const;

  // Dynamic working column set for in-tree sifting on wide models (P7-3).
  // External column indices 0..cols()-1 never change; inactive columns stay at 0 inside Simplex.
  void set_working_cols(const std::vector<char>& active);
  void activate_cols(const std::vector<int>& add_cols);
  void activate_all_unfixed();
  bool price_inactive(std::vector<int>& neg_cols, double tol = 1e-7);
  int active_cols_count() const;
  bool has_inactive_cols() const;

 private:
  struct Impl;
  explicit LpSolver(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace scsp
