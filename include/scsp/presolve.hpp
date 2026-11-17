// Basic presolve for set covering with packing side constraints.
//
// Reductions (Balas & Ho 1980; Beasley 1987), all recorded so that a solution
// of the reduced problem maps back to the original one:
//   * a row covered by a single column fixes that column to 1;
//   * a column that cannot fit a side constraint, or covers no remaining row, is dropped;
//   * column domination: rows(j) subset of rows(k), cost(k) <= cost(j) and the side
//     coefficients of k are not larger than those of j  ->  drop j;
//   * without side constraints: cost(j) > sum over its rows of the cheapest column
//     covering that row  ->  drop j;
//   * row domination: columns(i) subset of columns(k)  ->  drop row k.
#pragma once

#include <vector>

#include "scsp/problem.hpp"

namespace scsp {

struct PresolveResult {
  Problem red;                 // reduced problem; red.offset includes the fixed columns
  std::vector<int> col_orig;   // reduced column -> original column
  std::vector<int> to_red;     // original column -> reduced column, or -1
  std::vector<int> fixed_one;  // original columns fixed to 1
  std::vector<int> rep;        // original column -> column that replaces it (itself if kept), -1 if none
  bool infeasible = false;
  int rounds = 0;
};

// `enable = false` returns the problem unchanged (identity maps).
PresolveResult presolve(const Problem& p, bool enable = true, double work_limit = 4e7);

}  // namespace scsp
