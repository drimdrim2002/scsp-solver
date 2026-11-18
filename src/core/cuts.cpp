#include "scsp/cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace scsp {

namespace {
const double INF = std::numeric_limits<double>::infinity();
const double EPS = 1e-6;
}  // namespace

void CutPool::clear() {
  ptr.assign(1, 0);
  idx.clear();
  val.clear();
  lo.clear();
  up.clear();
  efficacy.clear();
}

SetRows::SetRows(const LpProblem& P, const std::vector<int>& cols) : m(P.m), nk(static_cast<int>(cols.size())) {
  std::vector<char> unit(m, 1);
  for (int k = 0; k < nk; ++k)
    for (int q = P.cptr[cols[k]]; q < P.cptr[cols[k] + 1]; ++q)
      if (P.cval[q] != 1.0) unit[P.cidx[q]] = 0;
  cover.assign(m, 0);
  pack.assign(m, 0);
  for (int i = 0; i < m; ++i) {
    cover[i] = unit[i] && P.rlo[i] == 1.0;
    pack[i] = unit[i] && P.rup[i] == 1.0;
  }
  rptr.assign(m + 1, 0);
  kptr.assign(1, 0);
  for (int k = 0; k < nk; ++k) {
    for (int q = P.cptr[cols[k]]; q < P.cptr[cols[k] + 1]; ++q) {
      const int i = P.cidx[q];
      if (cover[i] || pack[i]) {
        krow.push_back(i);
        ++rptr[i + 1];
      }
    }
    kptr.push_back(static_cast<int>(krow.size()));
  }
  for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
  rcol.resize(krow.size());
  std::vector<int> pos(rptr.begin(), rptr.end() - 1);
  for (int k = 0; k < nk; ++k)
    for (int q = kptr[k]; q < kptr[k + 1]; ++q) rcol[pos[krow[q]]++] = k;
}

// Greedy search for an odd row set S with  sum of slacks + (LP weight of the columns in an odd number of
// rows of S)  below one: from every row, add the row that lowers that sum most, and keep the best odd set
// met on the way.
int separate_zero_half(const SetRows& R, const std::vector<double>& x, bool covering, int max_size, double min_viol,
                       int max_cuts, std::mt19937& rng, CutPool& out) {
  const int m = R.m;
  const std::vector<char>& ok = covering ? R.cover : R.pack;
  // support of x, row-wise
  std::vector<int> sptr(m + 1, 0), scol;
  std::vector<double> slack(m, INF);
  {
    std::vector<int> cnt(m, 0);
    for (int k = 0; k < R.nk; ++k)
      if (x[k] > EPS)
        for (int q = R.kptr[k]; q < R.kptr[k + 1]; ++q) ++cnt[R.krow[q]];
    for (int i = 0; i < m; ++i) sptr[i + 1] = sptr[i] + cnt[i];
    scol.resize(sptr[m]);
    std::vector<int> pos(sptr.begin(), sptr.end() - 1);
    std::vector<double> act(m, 0.0);
    for (int k = 0; k < R.nk; ++k)
      if (x[k] > EPS)
        for (int q = R.kptr[k]; q < R.kptr[k + 1]; ++q) {
          scol[pos[R.krow[q]]++] = k;
          act[R.krow[q]] += x[k];
        }
    for (int i = 0; i < m; ++i)
      if (ok[i]) slack[i] = std::max(0.0, covering ? act[i] - 1.0 : 1.0 - act[i]);
  }
  std::vector<int> seeds;
  for (int i = 0; i < m; ++i) {
    if (!ok[i] || slack[i] > 0.5) continue;
    bool frac = false;
    for (int q = sptr[i]; q < sptr[i + 1] && !frac; ++q) frac = x[scol[q]] < 1.0 - EPS;
    if (frac) seeds.push_back(i);
  }
  std::shuffle(seeds.begin(), seeds.end(), rng);

  std::vector<char> par(R.nk, 0), in_s(m, 0), in_f(m, 0);
  std::vector<int> S, bestS, frontier, cnt(R.nk, 0), touched;
  std::set<std::vector<int>> seen;
  struct Found {
    double viol;
    std::vector<int> rows;
  };
  std::vector<Found> found;
  auto add_row = [&](int r, double& f) {
    f += slack[r];
    for (int q = sptr[r]; q < sptr[r + 1]; ++q) {
      const int k = scol[q];
      f += par[k] ? -x[k] : x[k];
      par[k] ^= 1;
      for (int a = R.kptr[k]; a < R.kptr[k + 1]; ++a) {
        const int i = R.krow[a];
        if (!in_s[i] && !in_f[i] && ok[i] && slack[i] <= 0.5) {
          in_f[i] = 1;
          frontier.push_back(i);
        }
      }
    }
    in_s[r] = 1;
    S.push_back(r);
  };
  for (int seed : seeds) {
    if (static_cast<int>(found.size()) >= 4 * max_cuts) break;
    S.clear();
    frontier.clear();
    double f = 0.0, best_f = INF;
    add_row(seed, f);
    while (static_cast<int>(S.size()) < max_size) {
      int br = -1;
      double bd = INF;
      for (int r : frontier) {
        if (in_s[r]) continue;
        double d = slack[r];
        for (int q = sptr[r]; q < sptr[r + 1]; ++q) d += par[scol[q]] ? -x[scol[q]] : x[scol[q]];
        if (d < bd) bd = d, br = r;
      }
      if (br < 0) break;
      add_row(br, f);
      if ((S.size() & 1) && f < best_f - 1e-9) {
        best_f = f;
        bestS = S;
      }
    }
    for (int r : S) {
      in_s[r] = 0;
      for (int q = sptr[r]; q < sptr[r + 1]; ++q) par[scol[q]] = 0;
    }
    for (int r : frontier) in_f[r] = 0;
    if (best_f < 1.0 - 2.0 * min_viol && bestS.size() >= 3) {  // the cut is violated by (1 - f) / 2
      std::sort(bestS.begin(), bestS.end());
      if (seen.insert(bestS).second) found.push_back({(1.0 - best_f) / 2.0, bestS});
    }
  }
  std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.viol > b.viol; });
  int added = 0;
  for (const Found& fd : found) {
    if (added >= max_cuts) break;
    touched.clear();
    for (int r : fd.rows)
      for (int q = R.rptr[r]; q < R.rptr[r + 1]; ++q) {
        const int k = R.rcol[q];
        if (cnt[k]++ == 0) touched.push_back(k);
      }
    std::sort(touched.begin(), touched.end());
    const int s = static_cast<int>(fd.rows.size());
    const double rhs = covering ? (s + 1) / 2 : (s - 1) / 2;
    double lhs = 0.0, norm = 0.0;
    const size_t start = out.idx.size();
    for (int k : touched) {
      double c = covering ? (cnt[k] + 1) / 2 : cnt[k] / 2;
      if (covering) c = std::min(c, rhs);  // x is binary: a coefficient above the right-hand side can come down to it
      cnt[k] = 0;
      if (c == 0.0) continue;
      out.idx.push_back(k);
      out.val.push_back(c);
      lhs += c * x[k];
      norm += c * c;
    }
    const double viol = covering ? rhs - lhs : lhs - rhs;
    if (viol < min_viol || norm == 0.0) {
      out.idx.resize(start);
      out.val.resize(start);
      continue;
    }
    out.ptr.push_back(static_cast<int>(out.idx.size()));
    out.lo.push_back(covering ? rhs : -INF);
    out.up.push_back(covering ? INF : rhs);
    out.efficacy.push_back(viol / std::sqrt(norm));
    ++added;
  }
  return added;
}

// Greedy cliques of the conflict graph (two columns conflict when they share a packing row), grown from
// the fractional columns by decreasing LP value and then extended by columns at zero.
int separate_cliques(const SetRows& R, const std::vector<double>& x, double min_viol, int max_cuts, CutPool& out) {
  std::vector<int> sup;
  for (int k = 0; k < R.nk; ++k)
    if (x[k] > EPS) sup.push_back(k);
  std::sort(sup.begin(), sup.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
  std::vector<int> stamp(R.m, -1), cstamp(R.nk, -1);
  int tick = 0;
  auto conflict = [&](int a, int b) {  // rows of a are stamped by the caller with `tick`
    (void)a;
    for (int q = R.kptr[b]; q < R.kptr[b + 1]; ++q)
      if (stamp[R.krow[q]] == tick && R.pack[R.krow[q]]) return true;
    return false;
  };
  auto share = [&](int a, int b) {
    ++tick;
    for (int q = R.kptr[a]; q < R.kptr[a + 1]; ++q) stamp[R.krow[q]] = tick;
    return conflict(a, b);
  };
  std::set<std::vector<int>> seen;
  int added = 0, ctick = 0;
  std::vector<int> C, cand;
  for (int s : sup) {
    if (added >= max_cuts) break;
    if (x[s] > 1.0 - EPS) continue;
    // candidates: support columns in a packing row of s
    ++ctick;
    cand.clear();
    cstamp[s] = ctick;
    for (int q = R.kptr[s]; q < R.kptr[s + 1]; ++q) {
      const int i = R.krow[q];
      if (!R.pack[i]) continue;
      for (int a = R.rptr[i]; a < R.rptr[i + 1]; ++a) {
        const int k = R.rcol[a];
        if (cstamp[k] != ctick && x[k] > EPS) {
          cstamp[k] = ctick;
          cand.push_back(k);
        }
      }
    }
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    C.assign(1, s);
    double sum = x[s];
    for (int k : cand) {
      bool all = true;
      for (int c : C)
        if (!share(c, k)) {
          all = false;
          break;
        }
      if (all) {
        C.push_back(k);
        sum += x[k];
      }
    }
    if (sum < 1.0 + min_viol || C.size() < 3) continue;
    // a clique inside one packing row is that row's own inequality
    bool in_row = false;
    for (int q = R.kptr[s]; q < R.kptr[s + 1] && !in_row; ++q) {
      const int i = R.krow[q];
      if (!R.pack[i]) continue;
      bool all = true;
      for (int c : C) {
        bool has = false;
        for (int a = R.kptr[c]; a < R.kptr[c + 1] && !has; ++a) has = R.krow[a] == i;
        if (!has) {
          all = false;
          break;
        }
      }
      in_row = all;
    }
    if (in_row) continue;
    // extend by columns at zero that conflict with every member (candidates: the packing rows of s)
    for (int q = R.kptr[s]; q < R.kptr[s + 1]; ++q) {
      const int i = R.krow[q];
      if (!R.pack[i]) continue;
      for (int a = R.rptr[i]; a < R.rptr[i + 1]; ++a) {
        const int k = R.rcol[a];
        if (cstamp[k] == ctick) continue;
        cstamp[k] = ctick;
        bool all = true;
        for (int c : C)
          if (!share(c, k)) {
            all = false;
            break;
          }
        if (all) C.push_back(k);
      }
    }
    std::sort(C.begin(), C.end());
    if (!seen.insert(C).second) continue;
    for (int k : C) {
      out.idx.push_back(k);
      out.val.push_back(1.0);
    }
    out.ptr.push_back(static_cast<int>(out.idx.size()));
    out.lo.push_back(-INF);
    out.up.push_back(1.0);
    out.efficacy.push_back((sum - 1.0) / std::sqrt(static_cast<double>(C.size())));
    ++added;
  }
  return added;
}

}  // namespace scsp
