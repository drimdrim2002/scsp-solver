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

// x: LP values of the loaded columns. Cuts violated by more than min_viol are appended; returns how many.
int separate_zero_half(const SetRows& R, const std::vector<double>& x, bool covering, int max_size, double min_viol,
                       int max_cuts, std::mt19937& rng, CutPool& out);
int separate_cliques(const SetRows& R, const std::vector<double>& x, double min_viol, int max_cuts, CutPool& out);

}  // namespace scsp
