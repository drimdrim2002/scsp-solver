#include "scsp/lu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace scsp {

namespace {
const double kAbsPivot = 1e-11;  // smallest acceptable pivot
const double kThreshold = 0.1;   // pivot >= kThreshold * largest entry of its column
}  // namespace

int BasisLU::factor(int m, const std::function<Col(int)>& get, std::vector<std::pair<int, int>>* bad) {
  m_ = m;
  nupd_ = 0;
  Lrow.clear();
  Lstart.assign(1, 0);
  Lidx.clear();
  Lval.clear();
  Rrow.clear();
  Rstart.assign(1, 0);
  Ridx.clear();
  Rval.clear();
  Urows.resize(m);
  for (auto& r : Urows) r.clear();
  diag.assign(m, 0.0);
  prow.assign(m, -1);
  pcol.assign(m, -1);
  order.clear();
  opos.assign(m, -1);
  colpiv.assign(m, -1);
  work_.assign(m, 0.0);

  // active submatrix: rows with values, columns with the row pattern
  std::vector<std::vector<Ent>> rows(m);
  std::vector<std::vector<int>> cols(m);
  for (int p = 0; p < m; ++p) {
    const Col c = get(p);
    cols[p].reserve(c.len);
    for (int k = 0; k < c.len; ++k) {
      if (c.val[k] == 0.0) continue;
      rows[c.idx[k]].push_back({p, c.val[k]});
      cols[p].push_back(c.idx[k]);
    }
  }
  std::vector<char> ractive(m, 1), cactive(m, 1);
  std::vector<int> posin(m, 0), rstack, cstack;
  for (int i = 0; i < m; ++i) {
    if (rows[i].size() == 1) rstack.push_back(i);
    if (cols[i].size() == 1) cstack.push_back(i);
  }
  auto value = [&](int i, int c) {
    for (const Ent& e : rows[i])
      if (e.c == c) return e.v;
    return 0.0;
  };
  auto colmax = [&](int c) {
    double mx = 0.0;
    for (int i : cols[c]) mx = std::max(mx, std::fabs(value(i, c)));
    return mx;
  };

  int npiv = 0;
  while (npiv < m) {
    int pi = -1, pc = -1;
    double pv = 0.0;
    // column singletons need no elimination
    while (pi < 0 && !cstack.empty()) {
      const int c = cstack.back();
      cstack.pop_back();
      if (!cactive[c] || cols[c].size() != 1) continue;
      const int i = cols[c][0];
      const double v = value(i, c);
      if (std::fabs(v) > kAbsPivot) pi = i, pc = c, pv = v;
    }
    // row singletons create no fill
    while (pi < 0 && !rstack.empty()) {
      const int i = rstack.back();
      rstack.pop_back();
      if (!ractive[i] || rows[i].size() != 1) continue;
      const int c = rows[i][0].c;
      const double v = rows[i][0].v;
      if (std::fabs(v) > kAbsPivot && std::fabs(v) >= kThreshold * colmax(c)) pi = i, pc = c, pv = v;
    }
    // Markowitz search over the columns with the smallest counts
    if (pi < 0) {
      size_t mincnt = std::numeric_limits<size_t>::max();
      for (int c = 0; c < m; ++c)
        if (cactive[c] && !cols[c].empty()) mincnt = std::min(mincnt, cols[c].size());
      if (mincnt == std::numeric_limits<size_t>::max()) break;
      long best = std::numeric_limits<long>::max();
      for (int pass = 0; pass < 2 && pi < 0; ++pass) {
        int examined = 0;
        for (int c = 0; c < m && examined < 8; ++c) {
          if (!cactive[c] || cols[c].empty()) continue;
          if (pass == 0 ? cols[c].size() > mincnt + 1 : false) continue;
          ++examined;
          const double mx = colmax(c);
          if (mx <= kAbsPivot) continue;
          const long cc = static_cast<long>(cols[c].size()) - 1;
          for (int i : cols[c]) {
            const double v = value(i, c);
            if (std::fabs(v) < kThreshold * mx || std::fabs(v) <= kAbsPivot) continue;
            const long merit = (static_cast<long>(rows[i].size()) - 1) * cc;
            if (merit < best || (merit == best && std::fabs(v) > std::fabs(pv))) best = merit, pi = i, pc = c, pv = v;
          }
        }
      }
      if (pi < 0) {
        // every remaining column is numerically empty in the columns examined: look at all of them
        for (int c = 0; c < m && pi < 0; ++c) {
          if (!cactive[c] || cols[c].empty()) continue;
          const double mx = colmax(c);
          if (mx <= kAbsPivot) continue;
          for (int i : cols[c]) {
            const double v = value(i, c);
            if (std::fabs(v) >= kThreshold * mx) {
              pi = i, pc = c, pv = v;
              break;
            }
          }
        }
        if (pi < 0) break;
      }
    }

    const int k = npiv++;
    prow[k] = pi;
    pcol[k] = pc;
    diag[k] = pv;
    colpiv[pc] = k;
    opos[k] = static_cast<int>(order.size());
    order.push_back(k);
    std::vector<Ent>& urow = Urows[k];
    for (const Ent& e : rows[pi])
      if (e.c != pc) urow.push_back(e);

    bool any = false;
    for (int i : cols[pc]) {
      if (i == pi) continue;
      std::vector<Ent>& rw = rows[i];
      double v = 0.0;
      for (size_t t = 0; t < rw.size(); ++t)
        if (rw[t].c == pc) {
          v = rw[t].v;
          rw[t] = rw.back();
          rw.pop_back();
          break;
        }
      const double mu = v / pv;
      if (mu != 0.0) {
        if (!any) Lrow.push_back(pi), any = true;
        Lidx.push_back(i);
        Lval.push_back(mu);
        if (!urow.empty()) {
          for (size_t t = 0; t < rw.size(); ++t) posin[rw[t].c] = static_cast<int>(t) + 1;
          for (const Ent& e : urow) {
            const int t = posin[e.c];
            if (t) {
              rw[t - 1].v -= mu * e.v;
            } else {
              rw.push_back({e.c, -mu * e.v});
              cols[e.c].push_back(i);
            }
          }
          for (const Ent& e : rw) posin[e.c] = 0;
        }
      }
      if (rw.size() == 1) rstack.push_back(i);
    }
    if (any) Lstart.push_back(static_cast<int>(Lidx.size()));
    for (const Ent& e : urow) {
      std::vector<int>& cl = cols[e.c];
      for (size_t t = 0; t < cl.size(); ++t)
        if (cl[t] == pi) {
          cl[t] = cl.back();
          cl.pop_back();
          break;
        }
      if (cl.size() == 1) cstack.push_back(e.c);
    }
    ractive[pi] = 0;
    cactive[pc] = 0;
    rows[pi].clear();
    cols[pc].clear();
  }

  factor_nnz_ = nnz();
  if (npiv == m) return 0;
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

long BasisLU::nnz() const {
  long s = static_cast<long>(Lidx.size() + Ridx.size()) + m_;
  for (const auto& r : Urows) s += static_cast<long>(r.size());
  return s;
}

void BasisLU::ftran(std::vector<double>& y, std::vector<double>& x, std::vector<double>* spike) const {
  for (size_t e = 0; e < Lrow.size(); ++e) {
    const double yp = y[Lrow[e]];
    if (yp == 0.0) continue;
    for (int s = Lstart[e]; s < Lstart[e + 1]; ++s) y[Lidx[s]] -= Lval[s] * yp;
  }
  for (size_t e = 0; e < Rrow.size(); ++e) {
    double s = 0.0;
    for (int q = Rstart[e]; q < Rstart[e + 1]; ++q) s += Rval[q] * y[Ridx[q]];
    y[Rrow[e]] -= s;
  }
  if (spike) *spike = y;
  for (int idx = m_ - 1; idx >= 0; --idx) {
    const int id = order[idx];
    double s = y[prow[id]];
    for (const Ent& e : Urows[id]) s -= e.v * x[e.c];
    x[pcol[id]] = s / diag[id];
  }
}

void BasisLU::btran(std::vector<double>& w, std::vector<double>& z) const {
  for (int idx = 0; idx < m_; ++idx) {
    const int id = order[idx];
    const double t = w[pcol[id]] / diag[id];
    z[prow[id]] = t;
    if (t != 0.0)
      for (const Ent& e : Urows[id]) w[e.c] -= e.v * t;
  }
  for (size_t e = Rrow.size(); e-- > 0;) {
    const double zt = z[Rrow[e]];
    if (zt == 0.0) continue;
    for (int q = Rstart[e]; q < Rstart[e + 1]; ++q) z[Ridx[q]] -= Rval[q] * zt;
  }
  for (size_t e = Lrow.size(); e-- > 0;) {
    double s = 0.0;
    for (int q = Lstart[e]; q < Lstart[e + 1]; ++q) s += Lval[q] * z[Lidx[q]];
    z[Lrow[e]] -= s;
  }
}

bool BasisLU::update(int p, const std::vector<double>& spike, double alpha) {
  const int t = colpiv[p];
  const int it = opos[t];
  // the old column p can only appear in rows of earlier pivots
  for (int idx = 0; idx < it; ++idx) {
    std::vector<Ent>& rw = Urows[order[idx]];
    for (size_t s = 0; s < rw.size(); ++s)
      if (rw[s].c == p) {
        rw[s] = rw.back();
        rw.pop_back();
        break;
      }
  }
  // row t moves to the end: eliminate its entries with the rows of the later pivots
  for (const Ent& e : Urows[t]) work_[e.c] = e.v;
  Urows[t].clear();
  double nd = spike[prow[t]];
  const size_t r0 = Ridx.size();
  for (int idx = it + 1; idx < m_; ++idx) {
    const int id = order[idx];
    const int c = pcol[id];
    const double w = work_[c];
    if (w == 0.0) continue;
    work_[c] = 0.0;
    const double mu = w / diag[id];
    for (const Ent& e : Urows[id]) work_[e.c] -= mu * e.v;
    nd -= mu * spike[prow[id]];
    Ridx.push_back(prow[id]);
    Rval.push_back(mu);
  }
  if (Ridx.size() > r0) {
    Rrow.push_back(prow[t]);
    Rstart.push_back(static_cast<int>(Ridx.size()));
  }
  const double expect = alpha * diag[t];
  const bool ok = std::fabs(nd) > kAbsPivot &&
                  std::fabs(nd - expect) <= 1e-6 * std::max(std::fabs(nd), std::fabs(expect)) + 1e-10;
  diag[t] = nd;
  for (int id = 0; id < m_; ++id) {
    if (id == t) continue;
    const double v = spike[prow[id]];
    if (std::fabs(v) > 1e-14) Urows[id].push_back({p, v});
  }
  order.erase(order.begin() + it);
  order.push_back(t);
  for (int idx = it; idx < m_; ++idx) opos[order[idx]] = idx;
  ++nupd_;
  return ok;
}

}  // namespace scsp
