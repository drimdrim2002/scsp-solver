// Set covering problem with a few packing-type side constraints (P1 scope).
//
//   min  sum_j cost[j] x_j + offset
//   s.t. sum_{j in row i} x_j >= 1            for every set row i
//        sum_j a[t][j] x_j <= side_rhs[t]     for every side row t   (a >= 0)
//        x binary,  cost >= 0
#pragma once

#include <cstdint>
#include <vector>

namespace scsp {

struct Problem {
  int n = 0;   // columns
  int m = 0;   // set (cover) rows
  int ns = 0;  // side rows
  std::vector<double> cost;
  std::vector<int> col_ptr, col_row;  // column -> set rows, sorted
  std::vector<int> row_ptr, row_col;  // set row -> columns, sorted
  std::vector<int> cs_ptr, cs_row;    // column -> side rows
  std::vector<double> cs_val;
  std::vector<double> side_rhs;
  double offset = 0.0;
  bool integral = false;  // every cost and the offset are integers

  // Builds the column-wise copies and `integral` from the row-wise data.
  // rows: row_ptr/row_col; side rows: side_ptr/side_idx/side_val.
  static Problem from_rows(int n, const double* cost, int m, const std::int64_t* row_ptr,
                           const std::int32_t* col_idx, int ns, const std::int64_t* side_ptr,
                           const std::int32_t* side_idx, const double* side_val,
                           const double* side_rhs, double offset);
};

}  // namespace scsp
