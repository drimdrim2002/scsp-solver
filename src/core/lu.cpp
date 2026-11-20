#include "scsp/lu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scsp {

namespace {
const double kAbsPivot = 1e-11;  // smallest acceptable pivot
const double kThreshold = 0.1;   // pivot >= kThreshold * largest entry of its column
}  // namespace

void BasisLU::u_row_push(int id, int c, double v) {
  if (u_rlen[id] == u_rcap[id]) {
    const int old_start = u_rstart[id];
    const int len = u_rlen[id];
    const int ncap = std::max(4, u_rcap[id] * 2);
    const int nstart = static_cast<int>(u_rcol.size());
    u_rcol.resize(static_cast<size_t>(nstart + ncap));
    u_rval.resize(static_cast<size_t>(nstart + ncap));
    for (int t = 0; t < len; ++t) {
      u_rcol[nstart + t] = u_rcol[old_start + t];
      u_rval[nstart + t] = u_rval[old_start + t];
    }
    u_rstart[id] = nstart;
    u_rcap[id] = ncap;
  }
  const int pos = u_rstart[id] + u_rlen[id]++;
  u_rcol[pos] = c;
  u_rval[pos] = v;
}

int BasisLU::factor(int m, const std::function<Col(int)>& get, std::vector<std::pair<int, int>>* bad) {
  m_ = m;
  nupd_ = 0;
  u_nnz_ = 0;
  if (!lf_ || lf_.use_count() > 1) lf_ = std::make_shared<LFactor>();
  LFactor& L = *lf_;
  L.Lrow.clear();
  L.Lstart.assign(1, 0);
  L.Lidx.clear();
  L.Lval.clear();
  L.lcol_of_row.assign(m, -1);
  L.Lt_start.clear();
  L.Lt_idx.clear();
  L.Lt_val.clear();
  Rrow.clear();
  Rstart.assign(1, 0);
  Ridx.clear();
  Rval.clear();
  u_rstart.assign(m, 0);
  u_rlen.assign(m, 0);
  u_rcap.assign(m, 0);
  u_rcol.clear();
  u_rval.clear();
  u_cstart.assign(m, 0);
  u_clen.assign(m, 0);
  u_ccol.clear();
  u_cval.clear();
  diag.assign(m, 0.0);
  prow.assign(m, -1);
  pcol.assign(m, -1);
  rowpiv.assign(m, -1);
  colpiv.assign(m, -1);
  order.clear();
  order.reserve(m);
  opos.assign(m, -1);
  sc_.ensure(m);
  sc_.reset_adaptive();
  if (m == 0) {
    factor_nnz_ = 0;
    return 0;
  }

  // Flat dynamic storage for active rows (with values) and active columns (row indices).
  auto& rstart = fw_.rstart;
  auto& rlen = fw_.rlen;
  auto& rcap = fw_.rcap;
  auto& rcol = fw_.rcol;
  auto& rval = fw_.rval;
  auto& cstart = fw_.cstart;
  auto& clen = fw_.clen;
  auto& ccap = fw_.ccap;
  auto& crow = fw_.crow;
  auto& rhead_cnt = fw_.rhead_cnt;
  auto& rprev = fw_.rprev;
  auto& rnext = fw_.rnext;
  auto& ractive = fw_.ractive;
  auto& cactive = fw_.cactive;
  auto& posin = fw_.posin;
  auto& rstack = fw_.rstack;
  auto& cstack = fw_.cstack;

  rstart.assign(m, 0);
  rlen.assign(m, 0);
  rcap.assign(m, 0);
  cstart.assign(m, 0);
  clen.assign(m, 0);
  ccap.assign(m, 0);

  // First pass: count row & column lengths to pack active submatrix contiguously with elbow room.
  int total_col_cap = 0;
  for (int p = 0; p < m; ++p) {
    const Col c = get(p);
    int cnt = 0;
    for (int k = 0; k < c.len; ++k) {
      if (c.val[k] == 0.0) continue;
      ++rlen[c.idx[k]];
      ++cnt;
    }
    cstart[p] = total_col_cap;
    const int cap = std::max(4, cnt + (cnt >> 1) + 2);
    ccap[p] = cap;
    total_col_cap += cap;
  }
  int total_row_cap = 0;
  for (int i = 0; i < m; ++i) {
    rstart[i] = total_row_cap;
    const int cap = std::max(4, rlen[i] + (rlen[i] >> 1) + 2);
    rcap[i] = cap;
    total_row_cap += cap;
    rlen[i] = 0;
  }
  rcol.resize(static_cast<size_t>(total_row_cap));
  rval.resize(static_cast<size_t>(total_row_cap));
  crow.resize(static_cast<size_t>(total_col_cap));

  // Second pass: populate active rows and columns.
  for (int p = 0; p < m; ++p) {
    const Col c = get(p);
    const int cb = cstart[p];
    int cn = 0;
    for (int k = 0; k < c.len; ++k) {
      const double v = c.val[k];
      if (v == 0.0) continue;
      const int i = c.idx[k];
      const int rp = rstart[i] + rlen[i]++;
      rcol[rp] = p;
      rval[rp] = v;
      crow[cb + cn++] = i;
    }
    clen[p] = cn;
  }

  auto row_reserve = [&](int i, int need) {
    if (rcap[i] >= need) return;
    const int old_s = rstart[i];
    const int len = rlen[i];
    const int ncap = std::max(need, rcap[i] * 2);
    const int ns = static_cast<int>(rcol.size());
    rcol.resize(static_cast<size_t>(ns + ncap));
    rval.resize(static_cast<size_t>(ns + ncap));
    for (int t = 0; t < len; ++t) {
      rcol[ns + t] = rcol[old_s + t];
      rval[ns + t] = rval[old_s + t];
    }
    rstart[i] = ns;
    rcap[i] = ncap;
  };
  auto col_push = [&](int c, int row) {
    if (clen[c] == ccap[c]) {
      const int old_s = cstart[c];
      const int len = clen[c];
      const int ncap = std::max(4, ccap[c] * 2);
      const int ns = static_cast<int>(crow.size());
      crow.resize(static_cast<size_t>(ns + ncap));
      for (int t = 0; t < len; ++t) crow[ns + t] = crow[old_s + t];
      cstart[c] = ns;
      ccap[c] = ncap;
    }
    crow[cstart[c] + clen[c]++] = row;
  };

  // Doubly-linked row-count buckets (Suhl & Suhl 1990) for O(1) min-count row lookup.
  // Searching rows avoids linear scans because rcol/rval are stored contiguously per row.
  rhead_cnt.assign(m + 1, -1);
  rprev.assign(m, -1);
  rnext.assign(m, -1);
  int min_rcnt = m + 1;
  auto rbucket_insert = [&](int i, int cnt) {
    const int h = rhead_cnt[cnt];
    rprev[i] = -1;
    rnext[i] = h;
    if (h >= 0) rprev[h] = i;
    rhead_cnt[cnt] = i;
    if (cnt < min_rcnt) min_rcnt = cnt;
  };
  auto rbucket_remove = [&](int i, int cnt) {
    const int pr = rprev[i];
    const int nx = rnext[i];
    if (pr >= 0) rnext[pr] = nx;
    else rhead_cnt[cnt] = nx;
    if (nx >= 0) rprev[nx] = pr;
    rprev[i] = rnext[i] = -1;
  };

  ractive.assign(m, 1);
  cactive.assign(m, 1);
  posin.assign(m, 0);
  rstack.clear();
  cstack.clear();
  for (int i = 0; i < m; ++i) {
    if (rlen[i] == 1) rstack.push_back(i);
    if (clen[i] == 1) cstack.push_back(i);
    if (rlen[i] > 0) rbucket_insert(i, rlen[i]);
  }

  auto value = [&](int i, int c) {
    const int beg = rstart[i], end = beg + rlen[i];
    for (int q = beg; q < end; ++q)
      if (rcol[q] == c) return rval[q];
    return 0.0;
  };

  u_rcol.reserve(static_cast<size_t>(total_row_cap));
  u_rval.reserve(static_cast<size_t>(total_row_cap));

  int npiv = 0;
  while (npiv < m) {
    int pi = -1, pc = -1;
    double pv = 0.0;
    // column singletons need no elimination
    while (pi < 0 && !cstack.empty()) {
      const int c = cstack.back();
      cstack.pop_back();
      if (!cactive[c] || clen[c] != 1) continue;
      const int i = crow[cstart[c]];
      const double v = value(i, c);
      if (std::fabs(v) > kAbsPivot) pi = i, pc = c, pv = v;
    }
    // row singletons create no fill in U and must pivot in their unique column
    while (pi < 0 && !rstack.empty()) {
      const int i = rstack.back();
      rstack.pop_back();
      if (!ractive[i] || rlen[i] != 1) continue;
      const int c = rcol[rstart[i]];
      const double v = rval[rstart[i]];
      if (std::fabs(v) > kAbsPivot) pi = i, pc = c, pv = v;
    }
    // Row-oriented Markowitz search with row-threshold pivoting (Suhl & Suhl 1990):
    // all entries of row i are contiguous, so zero cross-row value scans are needed.
    if (pi < 0) {
      while (min_rcnt <= m && rhead_cnt[min_rcnt] < 0) ++min_rcnt;
      if (min_rcnt > m) break;
      long best = std::numeric_limits<long>::max();
      auto inspect_row = [&](int i) {
        const int rl = rlen[i];
        const int rb = rstart[i];
        double rmx = 0.0;
        for (int t = 0; t < rl; ++t) rmx = std::max(rmx, std::fabs(rval[rb + t]));
        if (rmx <= kAbsPivot) return;
        const double thresh_v = kThreshold * rmx;
        const long rc = static_cast<long>(rl) - 1;
        for (int t = 0; t < rl; ++t) {
          const double v = rval[rb + t];
          const double av = std::fabs(v);
          if (av < thresh_v || av <= kAbsPivot) continue;
          const int c = rcol[rb + t];
          const long merit = rc * (static_cast<long>(clen[c]) - 1);
          if (merit < best || (merit == best && av > std::fabs(pv))) best = merit, pi = i, pc = c, pv = v;
        }
      };
      int examined = 0;
      for (int cnt = min_rcnt; cnt <= std::min(m, min_rcnt + 1) && examined < 8; ++cnt) {
        for (int i = rhead_cnt[cnt]; i >= 0 && examined < 8; i = rnext[i]) {
          ++examined;
          inspect_row(i);
        }
      }
      if (pi < 0) {
        examined = 0;
        for (int cnt = min_rcnt; cnt <= m && examined < 8; ++cnt) {
          for (int i = rhead_cnt[cnt]; i >= 0 && examined < 8; i = rnext[i]) {
            ++examined;
            inspect_row(i);
          }
        }
      }
      if (pi < 0) {
        for (int cnt = min_rcnt; cnt <= m && pi < 0; ++cnt) {
          for (int i = rhead_cnt[cnt]; i >= 0 && pi < 0; i = rnext[i]) {
            const int rb = rstart[i], re = rb + rlen[i];
            double rmx = 0.0;
            for (int q = rb; q < re; ++q) rmx = std::max(rmx, std::fabs(rval[q]));
            if (rmx <= kAbsPivot) continue;
            const double thresh_v = kThreshold * rmx;
            for (int q = rb; q < re; ++q) {
              if (std::fabs(rval[q]) >= thresh_v) {
                pi = i, pc = rcol[q], pv = rval[q];
                break;
              }
            }
          }
        }
        if (pi < 0) break;
      }
    }

    const int k = npiv++;
    prow[k] = pi;
    pcol[k] = pc;
    rowpiv[pi] = k;
    colpiv[pc] = k;
    diag[k] = pv;
    opos[k] = static_cast<int>(order.size());
    order.push_back(k);

    if (rlen[pi] > 0) rbucket_remove(pi, rlen[pi]);

    // Copy pivot row (excluding pc) directly into flat Urows with elbow room for FT updates.
    const int p_rb = rstart[pi], p_re = p_rb + rlen[pi];
    const int ulen = std::max(0, rlen[pi] - 1);
    const int ucap = std::max(4, ulen + 4);
    const int u0 = static_cast<int>(u_rcol.size());
    u_rcol.resize(static_cast<size_t>(u0 + ucap));
    u_rval.resize(static_cast<size_t>(u0 + ucap));
    int un = 0;
    for (int q = p_rb; q < p_re; ++q) {
      if (rcol[q] != pc) {
        u_rcol[u0 + un] = rcol[q];
        u_rval[u0 + un] = rval[q];
        ++un;
      }
    }
    u_rstart[k] = u0;
    u_rlen[k] = un;
    u_rcap[k] = ucap;
    u_nnz_ += un;
    const int u1 = u0 + un;

    bool any = false;
    const int pcb = cstart[pc];
    const int pcn = clen[pc];
    for (int ct = 0; ct < pcn; ++ct) {
      const int i = crow[pcb + ct];
      if (i == pi) continue;
      const int old_rl = rlen[i];
      if (old_rl > 0) rbucket_remove(i, old_rl);
      int rb = rstart[i];
      int& rl = rlen[i];
      double v = 0.0;
      for (int t = 0; t < rl; ++t) {
        if (rcol[rb + t] == pc) {
          v = rval[rb + t];
          rcol[rb + t] = rcol[rb + rl - 1];
          rval[rb + t] = rval[rb + rl - 1];
          --rl;
          break;
        }
      }
      const double mu = v / pv;
      if (mu != 0.0) {
        if (!any) {
          L.lcol_of_row[pi] = static_cast<int>(L.Lrow.size());
          L.Lrow.push_back(pi);
          any = true;
        }
        L.Lidx.push_back(i);
        L.Lval.push_back(mu);
        if (un > 0) {
          row_reserve(i, rl + un);
          rb = rstart[i];
          for (int t = 0; t < rl; ++t) posin[rcol[rb + t]] = t + 1;
          for (int q = u0; q < u1; ++q) {
            const int ec = u_rcol[q];
            const double ev = u_rval[q];
            const int t = posin[ec];
            if (t) {
              rval[rb + t - 1] -= mu * ev;
            } else {
              rcol[rb + rl] = ec;
              rval[rb + rl] = -mu * ev;
              ++rl;
              col_push(ec, i);
            }
          }
          for (int t = 0; t < rl; ++t) posin[rcol[rb + t]] = 0;
        }
      }
      if (rl > 0) rbucket_insert(i, rl);
      if (rl == 1) rstack.push_back(i);
    }
    if (any) L.Lstart.push_back(static_cast<int>(L.Lidx.size()));

    for (int q = u0; q < u1; ++q) {
      const int ec = u_rcol[q];
      const int old_cn = clen[ec];
      const int cb = cstart[ec];
      for (int t = 0; t < old_cn; ++t) {
        if (crow[cb + t] == pi) {
          crow[cb + t] = crow[cb + old_cn - 1];
          --clen[ec];
          if (clen[ec] == 1) cstack.push_back(ec);
          break;
        }
      }
    }
    ractive[pi] = 0;
    cactive[pc] = 0;
    rlen[pi] = 0;
    clen[pc] = 0;
  }

  factor_nnz_ = nnz();
  if (npiv != m) {
    if (bad) {
      bad->clear();
      std::vector<int> fr, fc;
      for (int i = 0; i < m; ++i) {
        if (ractive[i]) fr.push_back(i);
        if (cactive[i]) fc.push_back(i);
      }
      for (size_t t = 0; t < fr.size() && t < fc.size(); ++t) bad->push_back({fc[t], fr[t]});
    }
    return m - npiv;
  }

  // Build L^T (row-wise transpose of L) in O(m + nnz(L)) for hypersparse BTRAN.
  L.Lt_start.assign(m + 1, 0);
  for (int r : L.Lidx) ++L.Lt_start[r + 1];
  for (int i = 0; i < m; ++i) L.Lt_start[i + 1] += L.Lt_start[i];
  L.Lt_idx.resize(L.Lidx.size());
  L.Lt_val.resize(L.Lval.size());
  posin.assign(L.Lt_start.begin(), L.Lt_start.end() - 1);
  for (size_t e = 0; e < L.Lrow.size(); ++e) {
    const int r0 = L.Lrow[e];
    for (int q = L.Lstart[e]; q < L.Lstart[e + 1]; ++q) {
      const int p = posin[L.Lidx[q]]++;
      L.Lt_idx[p] = r0;
      L.Lt_val[p] = L.Lval[q];
    }
  }

  // Build Ucols (column-wise U) in O(m + nnz(U)) for hypersparse FTRAN and fast update().
  auto& ucnt = fw_.ucol_cnt;
  ucnt.assign(m, 0);
  for (int id = 0; id < m; ++id) {
    const int beg = u_rstart[id], end = beg + u_rlen[id];
    for (int q = beg; q < end; ++q) ++ucnt[u_rcol[q]];
  }
  int total_u = 0;
  for (int c = 0; c < m; ++c) {
    u_cstart[c] = total_u;
    u_clen[c] = ucnt[c];
    total_u += ucnt[c];
  }
  u_ccol.resize(static_cast<size_t>(total_u));
  u_cval.resize(static_cast<size_t>(total_u));
  posin.assign(u_cstart.begin(), u_cstart.end());
  for (int id = 0; id < m; ++id) {
    const int target_c = pcol[id];
    const int beg = u_rstart[id], end = beg + u_rlen[id];
    for (int q = beg; q < end; ++q) {
      const int p = posin[u_rcol[q]]++;
      u_ccol[p] = target_c;
      u_cval[p] = u_rval[q];
    }
  }
  return 0;
}

bool BasisLU::dfs_l(const LFactor& L, const std::vector<int>& seeds, int thresh, int& out_gen) const {
  const int g = sc_.next_gen();
  out_gen = g;
  int* const mark = sc_.mark.data();
  int* const dnode = sc_.dfs_node.data();
  int* const dedge = sc_.dfs_edge.data();
  const int* const lcol = L.lcol_of_row.data();
  const int* const lstart = L.Lstart.data();
  const int* const lidx = L.Lidx.data();
  auto& topo = sc_.topo;
  topo.clear();
  int visited = 0;
  for (int s : seeds) {
    if (mark[s] == g) continue;
    mark[s] = g;
    if (++visited > thresh) return false;
    const int e0 = lcol[s];
    if (e0 < 0) {
      topo.push_back(s);
      continue;
    }
    int top = 0;
    dnode[0] = s;
    dedge[0] = lstart[e0];
    while (top >= 0) {
      const int u = dnode[top];
      int& ep = dedge[top];
      const int end = lstart[lcol[u] + 1];
      bool pushed = false;
      while (ep < end) {
        const int v = lidx[ep++];
        if (mark[v] != g) {
          mark[v] = g;
          if (++visited > thresh) return false;
          const int ev = lcol[v];
          if (ev < 0) {
            topo.push_back(v);
          } else {
            ++top;
            dnode[top] = v;
            dedge[top] = lstart[ev];
            pushed = true;
            break;
          }
        }
      }
      if (!pushed) {
        topo.push_back(u);
        --top;
      }
    }
  }
  return true;
}

bool BasisLU::dfs_lt(const LFactor& L, const std::vector<int>& seeds, int thresh) const {
  const int g = sc_.next_gen();
  int* const mark = sc_.mark.data();
  int* const dnode = sc_.dfs_node.data();
  int* const dedge = sc_.dfs_edge.data();
  const int* const lts = L.Lt_start.data();
  const int* const lti = L.Lt_idx.data();
  auto& topo = sc_.topo;
  topo.clear();
  int visited = 0;
  for (int s : seeds) {
    if (mark[s] == g) continue;
    mark[s] = g;
    if (++visited > thresh) return false;
    const int b0 = lts[s], e0 = lts[s + 1];
    if (b0 == e0) {
      topo.push_back(s);
      continue;
    }
    int top = 0;
    dnode[0] = s;
    dedge[0] = b0;
    while (top >= 0) {
      const int u = dnode[top];
      int& ep = dedge[top];
      const int end = lts[u + 1];
      bool pushed = false;
      while (ep < end) {
        const int v = lti[ep++];
        if (mark[v] != g) {
          mark[v] = g;
          if (++visited > thresh) return false;
          const int bv = lts[v], ev = lts[v + 1];
          if (bv == ev) {
            topo.push_back(v);
          } else {
            ++top;
            dnode[top] = v;
            dedge[top] = bv;
            pushed = true;
            break;
          }
        }
      }
      if (!pushed) {
        topo.push_back(u);
        --top;
      }
    }
  }
  return true;
}

bool BasisLU::dfs_ucols(const std::vector<int>& seeds, int thresh) const {
  const int g = sc_.next_gen();
  int* const mark = sc_.mark.data();
  int* const dnode = sc_.dfs_node.data();
  int* const dedge = sc_.dfs_edge.data();
  const int* const ucs = u_cstart.data();
  const int* const ucl = u_clen.data();
  const int* const ucc = u_ccol.data();
  auto& topo = sc_.topo;
  topo.clear();
  int visited = 0;
  for (int s : seeds) {
    if (mark[s] == g) continue;
    mark[s] = g;
    if (++visited > thresh) return false;
    if (ucl[s] == 0) {
      topo.push_back(s);
      continue;
    }
    int top = 0;
    dnode[0] = s;
    dedge[0] = ucs[s];
    while (top >= 0) {
      const int u = dnode[top];
      int& ep = dedge[top];
      const int end = ucs[u] + ucl[u];
      bool pushed = false;
      while (ep < end) {
        const int v = ucc[ep++];
        if (mark[v] != g) {
          mark[v] = g;
          if (++visited > thresh) return false;
          if (ucl[v] == 0) {
            topo.push_back(v);
          } else {
            ++top;
            dnode[top] = v;
            dedge[top] = ucs[v];
            pushed = true;
            break;
          }
        }
      }
      if (!pushed) {
        topo.push_back(u);
        --top;
      }
    }
  }
  return true;
}

bool BasisLU::dfs_urows(const std::vector<int>& seeds, int thresh) const {
  const int g = sc_.next_gen();
  int* const mark = sc_.mark.data();
  int* const dnode = sc_.dfs_node.data();
  int* const dedge = sc_.dfs_edge.data();
  const int* const cp = colpiv.data();
  const int* const urs = u_rstart.data();
  const int* const url = u_rlen.data();
  const int* const urc = u_rcol.data();
  auto& topo = sc_.topo;
  topo.clear();
  int visited = 0;
  for (int s : seeds) {
    if (mark[s] == g) continue;
    mark[s] = g;
    if (++visited > thresh) return false;
    const int id0 = cp[s];
    if (url[id0] == 0) {
      topo.push_back(s);
      continue;
    }
    int top = 0;
    dnode[0] = s;
    dedge[0] = urs[id0];
    while (top >= 0) {
      const int u = dnode[top];
      const int id = cp[u];
      int& ep = dedge[top];
      const int end = urs[id] + url[id];
      bool pushed = false;
      while (ep < end) {
        const int v = urc[ep++];
        if (mark[v] != g) {
          mark[v] = g;
          if (++visited > thresh) return false;
          const int idv = cp[v];
          if (url[idv] == 0) {
            topo.push_back(v);
          } else {
            ++top;
            dnode[top] = v;
            dedge[top] = urs[idv];
            pushed = true;
            break;
          }
        }
      }
      if (!pushed) {
        topo.push_back(u);
        --top;
      }
    }
  }
  return true;
}

bool BasisLU::ftran_sparse(std::vector<double>& y, std::vector<int>& y_idx, bool y_sparse, std::vector<double>& x,
                           std::vector<int>& x_idx, std::vector<double>* spike, std::vector<int>* spike_idx,
                           bool* spike_sparse) const {
  if (m_ == 0) {
    x_idx.clear();
    if (spike_idx) spike_idx->clear();
    if (spike_sparse) *spike_sparse = true;
    return true;
  }
  sc_.ensure(m_);
  if (y_sparse && y_idx.empty()) {
    x_idx.clear();
    if (spike_idx) spike_idx->clear();
    if (spike_sparse) *spike_sparse = true;
    return true;
  }

  const LFactor& L = *lf_;
  const int thresh = m_ >= 24 ? std::max(8, m_ / 8) : 0;
  bool try_sparse = y_sparse && static_cast<int>(y_idx.size()) <= thresh;
  if (try_sparse && sc_.ftran_skip > 0) {
    --sc_.ftran_skip;
    try_sparse = false;
  }
  bool sparse = try_sparse;
  int ygen = 0;

  // Step 1: L-FTRAN (forward solve)
  if (sparse) {
    if (dfs_l(L, y_idx, thresh, ygen)) {
      const auto& topo = sc_.topo;
      double* const yp = y.data();
      const int* const lcol = L.lcol_of_row.data();
      const int* const ls = L.Lstart.data();
      const int* const li = L.Lidx.data();
      const double* const lv = L.Lval.data();
      for (size_t k = topo.size(); k-- > 0;) {
        const int r = topo[k];
        const double yr = yp[r];
        if (yr == 0.0) continue;
        const int e = lcol[r];
        if (e >= 0) {
          const int beg = ls[e], end = ls[e + 1];
          for (int s = beg; s < end; ++s) yp[li[s]] -= lv[s] * yr;
        }
      }
      y_idx = topo;
    } else {
      sparse = false;
    }
  }
  if (!sparse) {
    double* const yp = y.data();
    const int* const lr = L.Lrow.data();
    const int* const ls = L.Lstart.data();
    const int* const li = L.Lidx.data();
    const double* const lv = L.Lval.data();
    const size_t nl = L.Lrow.size();
    for (size_t e = 0; e < nl; ++e) {
      const double yv = yp[lr[e]];
      if (yv == 0.0) continue;
      const int beg = ls[e], end = ls[e + 1];
      for (int s = beg; s < end; ++s) yp[li[s]] -= lv[s] * yv;
    }
  }

  // Step 2: R-FTRAN (Forrest-Tomlin row etas)
  if (!Rrow.empty()) {
    double* const yp = y.data();
    const int* const rr = Rrow.data();
    const int* const rs = Rstart.data();
    const int* const ri = Ridx.data();
    const double* const rv = Rval.data();
    const size_t nr = Rrow.size();
    if (sparse) {
      int* const mark = sc_.mark.data();
      for (size_t e = 0; e < nr; ++e) {
        double s = 0.0;
        const int beg = rs[e], end = rs[e + 1];
        for (int q = beg; q < end; ++q) s += rv[q] * yp[ri[q]];
        if (s != 0.0) {
          const int r = rr[e];
          yp[r] -= s;
          if (mark[r] != ygen) {
            mark[r] = ygen;
            y_idx.push_back(r);
          }
        }
      }
      if (static_cast<int>(y_idx.size()) > thresh) sparse = false;
    } else {
      for (size_t e = 0; e < nr; ++e) {
        double s = 0.0;
        const int beg = rs[e], end = rs[e + 1];
        for (int q = beg; q < end; ++q) s += rv[q] * yp[ri[q]];
        yp[rr[e]] -= s;
      }
    }
  }

  // Optional spike output (before U solve)
  if (spike) {
    if (sparse && spike_idx && spike_sparse) {
      spike_idx->clear();
      double* const sp = spike->data();
      const double* const yp = y.data();
      for (int r : y_idx) {
        const double v = yp[r];
        if (v != 0.0) {
          sp[r] = v;
          spike_idx->push_back(r);
        }
      }
      *spike_sparse = true;
    } else {
      *spike = y;
      if (spike_sparse) *spike_sparse = false;
    }
  }

  // Step 3: U-FTRAN
  if (sparse) {
    auto& seeds = sc_.seed;
    seeds.clear();
    double* const yp = y.data();
    double* const xp = x.data();
    for (int r : y_idx) {
      const double v = yp[r];
      yp[r] = 0.0;
      if (v != 0.0) {
        const int c = pcol[rowpiv[r]];
        xp[c] = v;
        seeds.push_back(c);
      }
    }
    if (dfs_ucols(seeds, thresh)) {
      sc_.ftran_streak = 0;
      const auto& topo = sc_.topo;
      x_idx.clear();
      const int* const cp = colpiv.data();
      const double* const dg = diag.data();
      const int* const ucs = u_cstart.data();
      const int* const ucl = u_clen.data();
      const int* const ucc = u_ccol.data();
      const double* const ucv = u_cval.data();
      for (size_t k = topo.size(); k-- > 0;) {
        const int c = topo[k];
        double xc = xp[c];
        if (xc == 0.0) continue;
        xc /= dg[cp[c]];
        xp[c] = xc;
        x_idx.push_back(c);
        const int beg = ucs[c], end = beg + ucl[c];
        for (int q = beg; q < end; ++q) xp[ucc[q]] -= ucv[q] * xc;
      }
      return true;
    }
    // U-DFS went dense: record backoff and run gather-mode dense U-FTRAN (x already holds permuted y).
    ++sc_.ftran_streak;
    sc_.ftran_skip = std::min(32, 1 << std::min(5, sc_.ftran_streak));
    const int* const ord = order.data();
    const int* const pc = pcol.data();
    const double* const dg = diag.data();
    const int* const urs = u_rstart.data();
    const int* const url = u_rlen.data();
    const int* const urc = u_rcol.data();
    const double* const urv = u_rval.data();
    for (int idx = m_ - 1; idx >= 0; --idx) {
      const int id = ord[idx];
      const int c = pc[id];
      double s = xp[c];
      const int beg = urs[id], end = beg + url[id];
      for (int q = beg; q < end; ++q) s -= urv[q] * xp[urc[q]];
      xp[c] = s / dg[id];
    }
    return false;
  }

  if (try_sparse) {
    ++sc_.ftran_streak;
    sc_.ftran_skip = std::min(32, 1 << std::min(5, sc_.ftran_streak));
  }

  // Dense gather-mode U-FTRAN: reads flat u_rcol/u_rval sequentially, zero random writes, clears y.
  double* const yp = y.data();
  double* const xp = x.data();
  const int* const ord = order.data();
  const int* const pr = prow.data();
  const int* const pc = pcol.data();
  const double* const dg = diag.data();
  const int* const urs = u_rstart.data();
  const int* const url = u_rlen.data();
  const int* const urc = u_rcol.data();
  const double* const urv = u_rval.data();
  for (int idx = m_ - 1; idx >= 0; --idx) {
    const int id = ord[idx];
    const int r = pr[id];
    double s = yp[r];
    yp[r] = 0.0;
    const int beg = urs[id], end = beg + url[id];
    for (int q = beg; q < end; ++q) s -= urv[q] * xp[urc[q]];
    xp[pc[id]] = s / dg[id];
  }
  return false;
}

bool BasisLU::btran_sparse(std::vector<double>& w, const std::vector<int>& w_idx, bool w_sparse,
                           std::vector<double>& z, std::vector<int>& z_idx) const {
  if (m_ == 0) {
    z_idx.clear();
    return true;
  }
  sc_.ensure(m_);
  if (w_sparse && w_idx.empty()) {
    z_idx.clear();
    return true;
  }

  const LFactor& L = *lf_;
  const int thresh = m_ >= 24 ? std::max(8, m_ / 8) : 0;
  bool try_sparse = w_sparse && static_cast<int>(w_idx.size()) <= thresh;
  if (try_sparse && sc_.btran_skip > 0) {
    --sc_.btran_skip;
    try_sparse = false;
  }
  bool sparse = try_sparse;

  // Step 1: U-BTRAN (row-oriented forward solve)
  if (sparse) {
    if (dfs_urows(w_idx, thresh)) {
      const auto& topo = sc_.topo;
      z_idx.clear();
      double* const wp = w.data();
      double* const zp = z.data();
      const int* const cp = colpiv.data();
      const int* const pr = prow.data();
      const double* const dg = diag.data();
      const int* const urs = u_rstart.data();
      const int* const url = u_rlen.data();
      const int* const urc = u_rcol.data();
      const double* const urv = u_rval.data();
      for (size_t k = topo.size(); k-- > 0;) {
        const int c = topo[k];
        const double wc = wp[c];
        wp[c] = 0.0;
        if (wc == 0.0) continue;
        const int id = cp[c];
        const double t = wc / dg[id];
        const int r = pr[id];
        zp[r] = t;
        z_idx.push_back(r);
        const int beg = urs[id], end = beg + url[id];
        for (int q = beg; q < end; ++q) wp[urc[q]] -= urv[q] * t;
      }
    } else {
      sparse = false;
    }
  }
  if (!sparse) {
    double* const wp = w.data();
    double* const zp = z.data();
    const int* const ord = order.data();
    const int* const pr = prow.data();
    const int* const pc = pcol.data();
    const double* const dg = diag.data();
    const int* const urs = u_rstart.data();
    const int* const url = u_rlen.data();
    const int* const urc = u_rcol.data();
    const double* const urv = u_rval.data();
    for (int idx = 0; idx < m_; ++idx) {
      const int id = ord[idx];
      const int c = pc[id];
      const double wc = wp[c];
      wp[c] = 0.0;
      if (wc == 0.0) {
        zp[pr[id]] = 0.0;
        continue;
      }
      const double t = wc / dg[id];
      zp[pr[id]] = t;
      const int beg = urs[id], end = beg + url[id];
      for (int q = beg; q < end; ++q) wp[urc[q]] -= urv[q] * t;
    }
  }

  // Step 2: R-BTRAN (Forrest-Tomlin row etas in reverse)
  if (!Rrow.empty()) {
    double* const zp = z.data();
    const int* const rr = Rrow.data();
    const int* const rs = Rstart.data();
    const int* const ri = Ridx.data();
    const double* const rv = Rval.data();
    if (sparse) {
      const int zgen = sc_.next_gen();
      int* const mark = sc_.mark.data();
      for (int r : z_idx) mark[r] = zgen;
      for (size_t e = Rrow.size(); e-- > 0;) {
        const double zt = zp[rr[e]];
        if (zt == 0.0) continue;
        const int beg = rs[e], end = rs[e + 1];
        for (int q = beg; q < end; ++q) {
          const int r = ri[q];
          zp[r] -= rv[q] * zt;
          if (mark[r] != zgen) {
            mark[r] = zgen;
            z_idx.push_back(r);
          }
        }
      }
      if (static_cast<int>(z_idx.size()) > thresh) sparse = false;
    } else {
      for (size_t e = Rrow.size(); e-- > 0;) {
        const double zt = zp[rr[e]];
        if (zt == 0.0) continue;
        const int beg = rs[e], end = rs[e + 1];
        for (int q = beg; q < end; ++q) zp[ri[q]] -= rv[q] * zt;
      }
    }
  }

  // Step 3: L-BTRAN
  if (sparse) {
    if (dfs_lt(L, z_idx, thresh)) {
      sc_.btran_streak = 0;
      const auto& topo = sc_.topo;
      z_idx.clear();
      double* const zp = z.data();
      const int* const lts = L.Lt_start.data();
      const int* const lti = L.Lt_idx.data();
      const double* const ltv = L.Lt_val.data();
      for (size_t k = topo.size(); k-- > 0;) {
        const int r = topo[k];
        const double zr = zp[r];
        if (zr == 0.0) continue;
        z_idx.push_back(r);
        const int beg = lts[r], end = lts[r + 1];
        for (int q = beg; q < end; ++q) zp[lti[q]] -= ltv[q] * zr;
      }
      return true;
    }
  }

  if (try_sparse) {
    ++sc_.btran_streak;
    sc_.btran_skip = std::min(32, 1 << std::min(5, sc_.btran_streak));
  }

  // Dense gather-mode L-BTRAN: sequential Lidx/Lval dot products, zero random writes.
  double* const zp = z.data();
  const int* const lr = L.Lrow.data();
  const int* const ls = L.Lstart.data();
  const int* const li = L.Lidx.data();
  const double* const lv = L.Lval.data();
  for (size_t e = L.Lrow.size(); e-- > 0;) {
    double s = 0.0;
    const int beg = ls[e], end = ls[e + 1];
    for (int q = beg; q < end; ++q) s += lv[q] * zp[li[q]];
    zp[lr[e]] -= s;
  }
  return false;
}

void BasisLU::ftran(std::vector<double>& y, std::vector<double>& x, std::vector<double>* spike) const {
  std::vector<int> dummy_in, dummy_out;
  ftran_sparse(y, dummy_in, false, x, dummy_out, spike, nullptr, nullptr);
}

void BasisLU::btran(std::vector<double>& w, std::vector<double>& z) const {
  std::vector<int> dummy_in, dummy_out;
  btran_sparse(w, dummy_in, false, z, dummy_out);
}

bool BasisLU::update(int p, const std::vector<double>& spike, double alpha, const std::vector<int>* spike_idx) {
  sc_.ensure(m_);
  const int t = colpiv[p];
  const int it = opos[t];

  // 1. Remove old column p from earlier rows using Ucols[p] directly.
  {
    const int cb = u_cstart[p], ce = cb + u_clen[p];
    for (int q = cb; q < ce; ++q) {
      const int id = colpiv[u_ccol[q]];
      const int rb = u_rstart[id];
      int& rl = u_rlen[id];
      for (int s = 0; s < rl; ++s) {
        if (u_rcol[rb + s] == p) {
          u_rcol[rb + s] = u_rcol[rb + rl - 1];
          u_rval[rb + s] = u_rval[rb + rl - 1];
          --rl;
          --u_nnz_;
          break;
        }
      }
    }
    u_clen[p] = 0;
  }

  // 2. Row t moves to the end: gather its entries into work_ and remove p (= pcol[t]) from Ucols.
  auto& seeds = sc_.seed;
  seeds.clear();
  double* const work = sc_.work.data();
  {
    const int rb = u_rstart[t], rl = u_rlen[t];
    for (int s = 0; s < rl; ++s) {
      const int c = u_rcol[rb + s];
      work[c] = u_rval[rb + s];
      seeds.push_back(c);
      const int cb = u_cstart[c];
      int& cl = u_clen[c];
      for (int q = 0; q < cl; ++q) {
        if (u_ccol[cb + q] == p) {
          u_ccol[cb + q] = u_ccol[cb + cl - 1];
          u_cval[cb + q] = u_cval[cb + cl - 1];
          --cl;
          break;
        }
      }
    }
    u_nnz_ -= rl;
    u_rlen[t] = 0;
  }

  // 3. Eliminate row t's off-diagonal entries with later pivot rows (adaptive hypersparse).
  double nd = spike[prow[t]];
  const size_t r0 = Ridx.size();
  if (!seeds.empty()) {
    const int thresh = m_ >= 24 ? std::max(8, m_ / 8) : 0;
    bool try_sp = static_cast<int>(seeds.size()) <= thresh;
    if (try_sp && sc_.upd_skip > 0) {
      --sc_.upd_skip;
      try_sp = false;
    }
    if (try_sp && dfs_urows(seeds, thresh)) {
      sc_.upd_streak = 0;
      const auto& topo = sc_.topo;
      for (size_t k = topo.size(); k-- > 0;) {
        const int c = topo[k];
        const double w = work[c];
        work[c] = 0.0;
        if (w == 0.0) continue;
        const int id = colpiv[c];
        const double mu = w / diag[id];
        const int beg = u_rstart[id], end = beg + u_rlen[id];
        for (int q = beg; q < end; ++q) work[u_rcol[q]] -= mu * u_rval[q];
        nd -= mu * spike[prow[id]];
        Ridx.push_back(prow[id]);
        Rval.push_back(mu);
      }
    } else {
      if (try_sp) {
        ++sc_.upd_streak;
        sc_.upd_skip = std::min(32, 1 << std::min(5, sc_.upd_streak));
      }
      for (int idx = it + 1; idx < m_; ++idx) {
        const int id = order[idx];
        const int c = pcol[id];
        const double w = work[c];
        if (w == 0.0) continue;
        work[c] = 0.0;
        const double mu = w / diag[id];
        const int beg = u_rstart[id], end = beg + u_rlen[id];
        for (int q = beg; q < end; ++q) work[u_rcol[q]] -= mu * u_rval[q];
        nd -= mu * spike[prow[id]];
        Ridx.push_back(prow[id]);
        Rval.push_back(mu);
      }
    }
  }
  if (Ridx.size() > r0) {
    Rrow.push_back(prow[t]);
    Rstart.push_back(static_cast<int>(Ridx.size()));
  }

  const double expect = alpha * diag[t];
  const bool ok = std::fabs(nd) > kAbsPivot &&
                  std::fabs(nd - expect) <= 1e-6 * std::max(std::fabs(nd), std::fabs(expect)) + 1e-10;
  diag[t] = nd;

  // 4. Insert new spike column p into Urows and as a fresh contiguous slice of Ucols[p].
  u_cstart[p] = static_cast<int>(u_ccol.size());
  u_clen[p] = 0;
  if (spike_idx) {
    for (int r : *spike_idx) {
      const int id = rowpiv[r];
      if (id == t) continue;
      const double v = spike[r];
      if (std::fabs(v) > 1e-14) {
        u_row_push(id, p, v);
        u_ccol.push_back(pcol[id]);
        u_cval.push_back(v);
        ++u_clen[p];
        ++u_nnz_;
      }
    }
  } else {
    for (int id = 0; id < m_; ++id) {
      if (id == t) continue;
      const double v = spike[prow[id]];
      if (std::fabs(v) > 1e-14) {
        u_row_push(id, p, v);
        u_ccol.push_back(pcol[id]);
        u_cval.push_back(v);
        ++u_clen[p];
        ++u_nnz_;
      }
    }
  }

  order.erase(order.begin() + it);
  order.push_back(t);
  for (int idx = it; idx < m_; ++idx) opos[order[idx]] = idx;
  ++nupd_;
  return ok;
}

}  // namespace scsp
