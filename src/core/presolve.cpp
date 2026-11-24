#include "scsp/presolve.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace scsp {

namespace {

Problem compact(const Problem& p, const std::vector<char>& calive, const std::vector<char>& ralive,
                const std::vector<double>& rhs, double fixed_cost, std::vector<int>& col_orig,
                std::vector<int>& to_red) {
  Problem r;
  to_red.assign(p.n, -1);
  col_orig.clear();
  for (int j = 0; j < p.n; ++j)
    if (calive[j]) {
      to_red[j] = static_cast<int>(col_orig.size());
      col_orig.push_back(j);
    }
  r.n = static_cast<int>(col_orig.size());
  r.ns = p.ns;
  r.offset = p.offset + fixed_cost;
  r.integral = p.integral;
  r.side_rhs = rhs;
  r.cost.resize(r.n);
  r.row_ptr.push_back(0);
  for (int i = 0; i < p.m; ++i) {
    if (!ralive[i]) continue;
    for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q)
      if (calive[p.row_col[q]]) r.row_col.push_back(to_red[p.row_col[q]]);
    r.row_ptr.push_back(static_cast<int>(r.row_col.size()));
  }
  r.m = static_cast<int>(r.row_ptr.size()) - 1;
  r.col_ptr.assign(r.n + 1, 0);
  for (int j : r.row_col) ++r.col_ptr[j + 1];
  for (int j = 0; j < r.n; ++j) r.col_ptr[j + 1] += r.col_ptr[j];
  r.col_row.resize(r.row_col.size());
  std::vector<int> pos(r.col_ptr.begin(), r.col_ptr.end() - 1);
  for (int i = 0; i < r.m; ++i)
    for (int q = r.row_ptr[i]; q < r.row_ptr[i + 1]; ++q) r.col_row[pos[r.row_col[q]]++] = i;
  r.cs_ptr.push_back(0);
  for (int k = 0; k < r.n; ++k) {
    const int j = col_orig[k];
    r.cost[k] = p.cost[j];
    for (int q = p.cs_ptr[j]; q < p.cs_ptr[j + 1]; ++q) {
      r.cs_row.push_back(p.cs_row[q]);
      r.cs_val.push_back(p.cs_val[q]);
    }
    r.cs_ptr.push_back(static_cast<int>(r.cs_row.size()));
  }
  return r;
}

}  // namespace

PresolveResult presolve(const Problem& p, bool enable, double work_limit) {
  PresolveResult out;
  const int n = p.n, m = p.m, ns = p.ns;
  std::vector<char> calive(n, 1), ralive(m, 1);
  std::vector<double> rhs = p.side_rhs;
  out.rep.resize(n);
  std::iota(out.rep.begin(), out.rep.end(), 0);
  double fixed_cost = 0.0;

  if (!enable) {
    for (int i = 0; i < m; ++i)
      if (p.row_ptr[i] == p.row_ptr[i + 1]) out.infeasible = true;
    out.red = compact(p, calive, ralive, rhs, 0.0, out.col_orig, out.to_red);
    return out;
  }

  std::vector<int> csize(n, 0), rsize(m, 0), stamp(m, -1), cstamp(n, -1), sstamp(ns, -1);
  std::vector<double> sval(ns, 0.0), dmin(m, 0.0);
  double work = 0.0;

  auto recount = [&]() {
    std::fill(rsize.begin(), rsize.end(), 0);
    for (int j = 0; j < n; ++j) {
      csize[j] = 0;
      if (!calive[j]) continue;
      for (int q = p.col_ptr[j]; q < p.col_ptr[j + 1]; ++q)
        if (ralive[p.col_row[q]]) {
          ++csize[j];
          ++rsize[p.col_row[q]];
        }
    }
  };

  for (int round = 0; round < 6; ++round) {
    out.rounds = round + 1;
    bool changed = false;

    // columns that no longer fit a side constraint
    for (int j = 0; j < n; ++j) {
      if (!calive[j]) continue;
      for (int q = p.cs_ptr[j]; q < p.cs_ptr[j + 1]; ++q)
        if (p.cs_val[q] > rhs[p.cs_row[q]] + 1e-9) {
          calive[j] = 0;
          out.rep[j] = -1;
          changed = true;
          break;
        }
    }
    recount();

    // rows with no column (infeasible) or a single column (fix it)
    bool fixed_any = false;
    for (int i = 0; i < m; ++i) {
      if (!ralive[i]) continue;
      int only = -1, cnt = 0;
      for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1] && cnt < 2; ++q)
        if (calive[p.row_col[q]]) {
          only = p.row_col[q];
          ++cnt;
        }
      if (cnt == 0) {
        out.infeasible = true;
        out.red = compact(p, calive, ralive, rhs, fixed_cost, out.col_orig, out.to_red);
        return out;
      }
      if (cnt == 1) {
        calive[only] = 0;
        out.fixed_one.push_back(only);
        fixed_cost += p.cost[only];
        for (int q = p.col_ptr[only]; q < p.col_ptr[only + 1]; ++q) ralive[p.col_row[q]] = 0;
        for (int q = p.cs_ptr[only]; q < p.cs_ptr[only + 1]; ++q) rhs[p.cs_row[q]] -= p.cs_val[q];
        fixed_any = true;
      }
    }
    if (fixed_any) {
      changed = true;
      for (int t = 0; t < ns; ++t)
        if (rhs[t] < -1e-9) out.infeasible = true;
      if (out.infeasible) {
        out.red = compact(p, calive, ralive, rhs, fixed_cost, out.col_orig, out.to_red);
        return out;
      }
      recount();
    }

    // empty columns
    for (int j = 0; j < n; ++j)
      if (calive[j] && csize[j] == 0) {
        calive[j] = 0;
        out.rep[j] = -1;
        changed = true;
      }

    // cost test for pure set covering: cheaper to cover each row of j separately
    if (ns == 0) {
      for (int i = 0; i < m; ++i) {
        if (!ralive[i]) continue;
        double d = std::numeric_limits<double>::infinity();
        for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q)
          if (calive[p.row_col[q]]) d = std::min(d, p.cost[p.row_col[q]]);
        dmin[i] = d;
      }
      for (int j = 0; j < n; ++j) {
        if (!calive[j]) continue;
        double s = 0.0;
        for (int q = p.col_ptr[j]; q < p.col_ptr[j + 1]; ++q)
          if (ralive[p.col_row[q]]) s += dmin[p.col_row[q]];
        if (p.cost[j] > s + 1e-9 * (1.0 + s)) {
          calive[j] = 0;
          out.rep[j] = -1;  // replaced by the cheapest column of each of its rows
          changed = true;
        }
      }
    } else if (work < work_limit) {
      // Mandatory side-capacity probing: if every alive column covering row i uses >= mu > 0 of
      // side resource t, any column k not covering row i with a_{tk} + mu > rhs[t] is infeasible.
      std::vector<double> col_a(n, 0.0);
      std::vector<int> side_cols;
      int ptick = 0;
      std::vector<int> pmark(n, 0);
      for (int t = 0; t < ns && work < work_limit; ++t) {
        side_cols.clear();
        double max_a = 0.0;
        for (int j = 0; j < n; ++j) {
          if (!calive[j]) continue;
          for (int q = p.cs_ptr[j]; q < p.cs_ptr[j + 1]; ++q) {
            if (p.cs_row[q] == t && p.cs_val[q] > 1e-9) {
              col_a[j] = p.cs_val[q];
              side_cols.push_back(j);
              if (p.cs_val[q] > max_a) max_a = p.cs_val[q];
              break;
            }
          }
        }
        if (side_cols.empty()) continue;
        for (int i = 0; i < m && work < work_limit; ++i) {
          if (!ralive[i]) continue;
          double mu = std::numeric_limits<double>::infinity();
          for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q) {
            const int j = p.row_col[q];
            if (!calive[j]) continue;
            mu = std::min(mu, col_a[j]);
            if (mu <= 1e-9) break;
          }
          if (mu <= 1e-9 || !std::isfinite(mu) || max_a + mu <= rhs[t] + 1e-9) continue;
          ++ptick;
          for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q)
            if (calive[p.row_col[q]]) pmark[p.row_col[q]] = ptick;
          work += static_cast<double>(side_cols.size());
          for (int k : side_cols) {
            if (calive[k] && pmark[k] != ptick && col_a[k] + mu > rhs[t] + 1e-9) {
              calive[k] = 0;
              out.rep[k] = -1;
              changed = true;
            }
          }
        }
        for (int k : side_cols) col_a[k] = 0.0;
      }
    }

    // pairwise column domination
    for (int j = 0; j < n && work < work_limit; ++j) {
      if (!calive[j] || csize[j] == 0) continue;
      int rare = -1;
      for (int q = p.col_ptr[j]; q < p.col_ptr[j + 1]; ++q) {
        const int i = p.col_row[q];
        if (!ralive[i]) continue;
        stamp[i] = j;
        if (rare < 0 || rsize[i] < rsize[rare]) rare = i;
      }
      const int nsj = p.cs_ptr[j + 1] - p.cs_ptr[j];
      for (int q = p.cs_ptr[j]; q < p.cs_ptr[j + 1]; ++q) {
        sstamp[p.cs_row[q]] = j;
        sval[p.cs_row[q]] = p.cs_val[q];
      }
      for (int q = p.row_ptr[rare]; q < p.row_ptr[rare + 1]; ++q) {
        const int k = p.row_col[q];
        if (k == j || !calive[k] || p.cost[k] > p.cost[j] || csize[k] < csize[j]) continue;
        bool ok = true, side_equal = (p.cs_ptr[k + 1] - p.cs_ptr[k]) == nsj;
        for (int s = p.cs_ptr[k]; s < p.cs_ptr[k + 1] && ok; ++s) {
          const int t = p.cs_row[s];
          if (sstamp[t] != j || p.cs_val[s] > sval[t] + 1e-12) ok = false;
          else if (p.cs_val[s] != sval[t]) side_equal = false;
        }
        if (!ok) continue;
        int hit = 0;
        work += p.col_ptr[k + 1] - p.col_ptr[k];
        for (int s = p.col_ptr[k]; s < p.col_ptr[k + 1]; ++s)
          if (stamp[p.col_row[s]] == j) ++hit;
        if (hit != csize[j]) continue;
        if (p.cost[k] == p.cost[j] && csize[k] == csize[j] && side_equal && k > j) continue;  // identical: keep the first
        calive[j] = 0;
        out.rep[j] = k;
        changed = true;
        break;
      }
    }

    // row domination: columns(i) subset of columns(k) makes row k redundant
    recount();
    for (int i = 0; i < m && work < work_limit; ++i) {
      if (!ralive[i]) continue;
      int thin = -1;
      for (int q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q) {
        const int j = p.row_col[q];
        if (!calive[j]) continue;
        cstamp[j] = i;
        if (thin < 0 || csize[j] < csize[thin]) thin = j;
      }
      for (int q = p.col_ptr[thin]; q < p.col_ptr[thin + 1]; ++q) {
        const int k = p.col_row[q];
        if (k == i || !ralive[k] || rsize[k] < rsize[i]) continue;
        if (rsize[k] == rsize[i] && k < i) continue;  // identical rows: the lower index removes the higher
        int hit = 0;
        work += p.row_ptr[k + 1] - p.row_ptr[k];
        for (int s = p.row_ptr[k]; s < p.row_ptr[k + 1]; ++s)
          if (calive[p.row_col[s]] && cstamp[p.row_col[s]] == i) ++hit;
        if (hit == rsize[i]) {
          ralive[k] = 0;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }

  // final consistency: drop columns left without rows, fix nothing else
  recount();
  for (int j = 0; j < n; ++j)
    if (calive[j] && csize[j] == 0) {
      calive[j] = 0;
      out.rep[j] = -1;
    }
  for (int i = 0; i < m; ++i)
    if (ralive[i] && rsize[i] == 0) out.infeasible = true;
  out.red = compact(p, calive, ralive, rhs, fixed_cost, out.col_orig, out.to_red);
  return out;
}

}  // namespace scsp
