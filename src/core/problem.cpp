#include "scsp/problem.hpp"

#include <cmath>
#include <stdexcept>

namespace scsp {

Problem Problem::from_rows(int n, const double* cost, int m, const std::int64_t* row_ptr,
                           const std::int32_t* col_idx, int ns, const std::int64_t* side_ptr,
                           const std::int32_t* side_idx, const double* side_val,
                           const double* side_rhs, double offset) {
  Problem p;
  p.n = n;
  p.m = m;
  p.ns = ns;
  p.offset = offset;
  p.cost.assign(cost, cost + n);
  const std::int64_t nnz = m ? row_ptr[m] : 0;
  const std::int64_t snz = ns ? side_ptr[ns] : 0;
  if (nnz > 2000000000LL || snz > 2000000000LL) throw std::invalid_argument("too many nonzeros");

  p.integral = std::floor(offset) == offset;
  for (int j = 0; j < n; ++j) {
    if (!(cost[j] >= 0.0) || !std::isfinite(cost[j])) throw std::invalid_argument("costs must be finite and >= 0");
    if (std::floor(cost[j]) != cost[j]) p.integral = false;
  }

  p.row_ptr.resize(m + 1);
  p.row_col.resize(static_cast<size_t>(nnz));
  p.col_ptr.assign(n + 1, 0);
  for (int i = 0; i <= m; ++i) p.row_ptr[i] = static_cast<int>(row_ptr[i]);
  for (std::int64_t q = 0; q < nnz; ++q) {
    const int j = col_idx[q];
    if (j < 0 || j >= n) throw std::invalid_argument("column index out of range");
    p.row_col[static_cast<size_t>(q)] = j;
    ++p.col_ptr[j + 1];
  }
  for (int j = 0; j < n; ++j) p.col_ptr[j + 1] += p.col_ptr[j];
  p.col_row.resize(static_cast<size_t>(nnz));
  {
    std::vector<int> pos(p.col_ptr.begin(), p.col_ptr.end() - 1);
    for (int i = 0; i < m; ++i)
      for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q) p.col_row[pos[p.row_col[q]]++] = i;
  }

  p.side_rhs.assign(side_rhs, side_rhs + ns);
  p.cs_ptr.assign(n + 1, 0);
  for (std::int64_t q = 0; q < snz; ++q) {
    const int j = side_idx[q];
    if (j < 0 || j >= n) throw std::invalid_argument("side column index out of range");
    if (!(side_val[q] >= 0.0)) throw std::invalid_argument("side coefficients must be >= 0");
    ++p.cs_ptr[j + 1];
  }
  for (int j = 0; j < n; ++j) p.cs_ptr[j + 1] += p.cs_ptr[j];
  p.cs_row.resize(static_cast<size_t>(snz));
  p.cs_val.resize(static_cast<size_t>(snz));
  {
    std::vector<int> pos(p.cs_ptr.begin(), p.cs_ptr.end() - 1);
    for (int t = 0; t < ns; ++t)
      for (std::int64_t q = side_ptr[t]; q < side_ptr[t + 1]; ++q) {
        if (side_val[q] == 0.0) continue;
        const int k = pos[side_idx[q]]++;
        p.cs_row[k] = t;
        p.cs_val[k] = side_val[q];
      }
    // zero coefficients were skipped: close the gaps
    std::vector<int> ptr(n + 1, 0);
    int w = 0;
    for (int j = 0; j < n; ++j) {
      for (int k = p.cs_ptr[j]; k < pos[j]; ++k) {
        p.cs_row[w] = p.cs_row[k];
        p.cs_val[w] = p.cs_val[k];
        ++w;
      }
      ptr[j + 1] = w;
    }
    p.cs_ptr.swap(ptr);
    p.cs_row.resize(w);
    p.cs_val.resize(w);
  }
  return p;
}

}  // namespace scsp
