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

#include <cstdint>
#include <limits>
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

}  // namespace scsp
