// Cutting planes for set covering / packing / partitioning rows (P4).
//
// A "set row" has only coefficients 1. Its >= 1 side makes it a covering row, its <= 1 side a packing
// row; an equality row is both.
//
//   zero-half cuts (Caprara & Fischetti 1996): add an odd number of rows and divide by two.
//     covering side: sum_j ceil(c_j / 2) x_j >= (|S| + 1) / 2     (c_j = number of rows of S that column j is in)
//     packing side:  sum_j floor(c_j / 2) x_j <= (|S| - 1) / 2    (the odd hole inequalities of the conflict
//                    graph are of this kind: Hoffman & Padberg 1993)
//   clique cuts (Hoffman & Padberg 1993): columns that pairwise share a packing row, sum <= 1.
//
// Separation is heuristic. All cuts are valid for every 0-1 solution of the rows they were built from.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "scsp/lp.hpp"

namespace scsp {

struct CutPool {
  std::vector<int> ptr{0}, idx;   // row-wise, over the loaded columns
  std::vector<double> val, lo, up, efficacy;
  std::vector<char> is_int;       // 1 if all cut coefficients are integer (zero-half / clique), 0 otherwise (GMI)
  int size() const { return static_cast<int>(lo.size()); }
  void clear();
};

// The set rows of the problem restricted to the loaded columns, row-wise.
struct SetRows {
  int m = 0, nk = 0;
  std::vector<int> rptr, rcol;       // row -> loaded columns
  std::vector<int> kptr, krow;       // loaded column -> set rows
  std::vector<char> cover, pack;     // the row has a >= 1 side / a <= 1 side
  SetRows(const LpProblem& P, const std::vector<int>& cols);
};

// Nonnegative <= b knapsack rows (not pure unit packing rows) restricted to loaded columns (P8).
struct KnapsackRows {
  int nr = 0, nk = 0;
  std::vector<int> rptr, rcol;       // row -> loaded columns
  std::vector<double> rval, rhs;     // positive coefficients and finite upper bound
  KnapsackRows(const LpProblem& P, const std::vector<int>& cols);
};

// x: LP values of the loaded columns. Cuts violated by more than min_viol are appended; returns how many.
int separate_zero_half(const SetRows& R, const std::vector<double>& x, bool covering, int max_size, double min_viol,
                       int max_cuts, std::mt19937& rng, CutPool& out);
// Generalized Chvatal-Gomory {0, 1/d} covering cuts (d >= 3) over covering rows (Caprara & Fischetti 1996).
int separate_cg_cover(const SetRows& R, const std::vector<double>& x, int denom, int max_size, double min_viol,
                      int max_cuts, std::mt19937& rng, CutPool& out);
// Clique cuts from packing rows plus optional learned binary conflicts (x_a + x_b <= 1, Atamturk et al. 2000).
int separate_cliques(const SetRows& R, const std::vector<double>& x, double min_viol, int max_cuts, CutPool& out,
                     const std::vector<std::vector<int>>* extra_conf = nullptr);
// Lifted 0-1 knapsack cover inequalities (Balas 1975; Gu, Nemhauser & Savelsbergh 1998).
int separate_knapsack_covers(const KnapsackRows& K, const std::vector<double>& x, double min_viol, int max_cuts,
                             CutPool& out);
// Single-row c-MIR inequalities on knapsack constraints (Marchand & Wolsey 2001).
int separate_knapsack_mir(const KnapsackRows& K, const std::vector<double>& x, double min_viol, int max_cuts,
                          CutPool& out);
// Gomory mixed-integer (GMI) cuts from simplex tableau rows of fractional basic binary variables
// (Gomory 1960; Balas, Ceria, Cornuejols & Natraj 1996).
// row_int[i] == 1 iff row i of `lp` has all integer coefficients (so its row activity is integer).
int separate_gmi(LpSolver& lp, const std::vector<char>& row_int, const std::vector<double>& x, double min_viol,
                 int max_cuts, int max_nnz, CutPool& out);

}  // namespace scsp
