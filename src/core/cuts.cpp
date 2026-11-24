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
  is_int.clear();
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
    out.is_int.push_back(1);
    ++added;
  }
  return added;
}

// Greedy cliques of the conflict graph (two columns conflict when they share a packing row or a learned
// binary conflict x_a + x_b <= 1), grown from fractional columns by decreasing LP value and extended by zeros.
int separate_cliques(const SetRows& R, const std::vector<double>& x, double min_viol, int max_cuts, CutPool& out,
                     const std::vector<std::vector<int>>* extra_conf) {
  const bool has_extra = extra_conf && static_cast<int>(extra_conf->size()) == R.nk;
  std::vector<int> sup;
  for (int k = 0; k < R.nk; ++k)
    if (x[k] > EPS) sup.push_back(k);
  std::sort(sup.begin(), sup.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
  std::vector<int> stamp(R.m, -1), cstamp(R.nk, -1), estmp(has_extra ? R.nk : 0, -1);
  int tick = 0;
  auto conflict = [&](int a, int b) {  // rows & extra conflicts of a are stamped by the caller with `tick`
    (void)a;
    if (has_extra && estmp[b] == tick) return true;
    for (int q = R.kptr[b]; q < R.kptr[b + 1]; ++q)
      if (stamp[R.krow[q]] == tick && R.pack[R.krow[q]]) return true;
    return false;
  };
  auto share = [&](int a, int b) {
    ++tick;
    for (int q = R.kptr[a]; q < R.kptr[a + 1]; ++q) stamp[R.krow[q]] = tick;
    if (has_extra)
      for (int nb : (*extra_conf)[a]) estmp[nb] = tick;
    return conflict(a, b);
  };
  std::set<std::vector<int>> seen;
  int added = 0, ctick = 0;
  std::vector<int> C, cand;
  for (int s : sup) {
    if (added >= max_cuts) break;
    if (x[s] > 1.0 - EPS) continue;
    // candidates: support columns conflicting with s
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
    if (has_extra) {
      for (int k : (*extra_conf)[s]) {
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
    // extend by columns at zero that conflict with every member
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
    if (has_extra) {
      for (int k : (*extra_conf)[s]) {
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
    out.is_int.push_back(1);
    ++added;
  }
  return added;
}

// Generalized Chvatal-Gomory {0, 1/d} covering cuts (d >= 3):
// For a subset S of s covering rows with s = 1 (mod d), sum_j min(ceil(c_j / d), rhs) x_j >= rhs
// where rhs = (s + d - 1) / d is valid for every 0-1 solution.
int separate_cg_cover(const SetRows& R, const std::vector<double>& x, int denom, int max_size, double min_viol,
                      int max_cuts, std::mt19937& rng, CutPool& out) {
  if (denom < 3 || max_size < denom + 1 || max_cuts <= 0) return 0;
  const int m = R.m;
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
      if (R.cover[i]) slack[i] = std::max(0.0, act[i] - 1.0);
  }
  std::vector<int> seeds;
  for (int i = 0; i < m; ++i) {
    if (!R.cover[i] || slack[i] > 0.25) continue;
    bool frac = false;
    for (int q = sptr[i]; q < sptr[i + 1] && !frac; ++q) frac = x[scol[q]] < 1.0 - EPS;
    if (frac) seeds.push_back(i);
  }
  if (seeds.empty()) return 0;
  std::shuffle(seeds.begin(), seeds.end(), rng);
  if (static_cast<int>(seeds.size()) > 200) seeds.resize(200);

  std::vector<int> c_sup(R.nk, 0), cnt(R.nk, 0), touched, S, bestS, frontier;
  std::vector<char> in_s(m, 0), in_f(m, 0);
  std::set<std::vector<int>> seen;
  struct Found {
    double viol;
    std::vector<int> rows;
  };
  std::vector<Found> found;

  const double d_dbl = static_cast<double>(denom);
  for (int seed : seeds) {
    if (static_cast<int>(found.size()) >= max_cuts * 2) break;
    S.clear();
    frontier.clear();
    double sum_ceil = 0.0;
    auto add_r = [&](int r) {
      in_s[r] = 1;
      S.push_back(r);
      for (int q = sptr[r]; q < sptr[r + 1]; ++q) {
        const int k = scol[q];
        const int prev = c_sup[k]++;
        if (prev % denom == 0) sum_ceil += x[k];
        for (int a = R.kptr[k]; a < R.kptr[k + 1]; ++a) {
          const int nb = R.krow[a];
          if (R.cover[nb] && !in_s[nb] && !in_f[nb] && slack[nb] <= 0.35) {
            in_f[nb] = 1;
            frontier.push_back(nb);
          }
        }
      }
    };
    add_r(seed);
    double best_viol = min_viol;
    bestS.clear();
    while (static_cast<int>(S.size()) < max_size && !frontier.empty()) {
      int pick = -1;
      double best_inc = INF;
      for (int r : frontier) {
        if (in_s[r]) continue;
        double inc = 0.0;
        for (int q = sptr[r]; q < sptr[r + 1]; ++q) {
          const int k = scol[q];
          if (c_sup[k] % denom == 0) inc += x[k];
        }
        // prefer rows that add minimal new ceil-weight
        if (inc < best_inc) best_inc = inc, pick = r;
      }
      if (pick < 0) break;
      add_r(pick);
      const int s = static_cast<int>(S.size());
      if (s % denom == 1) {
        const double rhs = static_cast<double>((s + denom - 1) / denom);
        const double v = rhs - sum_ceil;
        if (v > best_viol) {
          best_viol = v;
          bestS = S;
        }
      }
      if (sum_ceil > static_cast<double>((s + denom - 1) / denom) + (d_dbl - 1.0) / d_dbl) break;
    }
    for (int r : S) {
      in_s[r] = 0;
      for (int q = sptr[r]; q < sptr[r + 1]; ++q) c_sup[scol[q]] = 0;
    }
    for (int r : frontier) in_f[r] = 0;
    if (!bestS.empty()) {
      std::sort(bestS.begin(), bestS.end());
      if (seen.insert(bestS).second) found.push_back({best_viol, bestS});
    }
  }
  std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.viol > b.viol; });
  int added = 0;
  for (const auto& fd : found) {
    if (added >= max_cuts) break;
    touched.clear();
    for (int r : fd.rows)
      for (int q = R.rptr[r]; q < R.rptr[r + 1]; ++q) {
        const int k = R.rcol[q];
        if (cnt[k]++ == 0) touched.push_back(k);
      }
    std::sort(touched.begin(), touched.end());
    const int s = static_cast<int>(fd.rows.size());
    const double rhs = static_cast<double>((s + denom - 1) / denom);
    double lhs = 0.0, norm2 = 0.0;
    const size_t start = out.idx.size();
    for (int k : touched) {
      const double c = std::min(static_cast<double>((cnt[k] + denom - 1) / denom), rhs);
      cnt[k] = 0;
      if (c == 0.0) continue;
      out.idx.push_back(k);
      out.val.push_back(c);
      lhs += c * x[k];
      norm2 += c * c;
    }
    const double viol = rhs - lhs;
    if (viol < min_viol || norm2 == 0.0) {
      out.idx.resize(start);
      out.val.resize(start);
      continue;
    }
    out.ptr.push_back(static_cast<int>(out.idx.size()));
    out.lo.push_back(rhs);
    out.up.push_back(INF);
    out.efficacy.push_back(viol / std::sqrt(norm2));
    out.is_int.push_back(1);
    ++added;
  }
  return added;
}

KnapsackRows::KnapsackRows(const LpProblem& P, const std::vector<int>& cols) : nk(static_cast<int>(cols.size())) {
  const int m = P.m;
  std::vector<char> ok(m, 0), all_unit(m, 1);
  for (int i = 0; i < m; ++i)
    if (P.rlo[i] == -INF && P.rup[i] < INF && P.rup[i] > 1e-9) ok[i] = 1;
  for (int k = 0; k < nk; ++k) {
    const int j = cols[k];
    for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
      const int i = P.cidx[q];
      if (!ok[i]) continue;
      const double v = P.cval[q];
      if (v < 0.0) ok[i] = 0;
      else if (v != 1.0) all_unit[i] = 0;
    }
  }
  std::vector<int> row_map(m, -1);
  for (int i = 0; i < m; ++i) {
    if (!ok[i]) continue;
    if (all_unit[i] && P.rup[i] <= 1.0 + 1e-9) continue;  // handled by SetRows packing/clique cuts
    row_map[i] = nr++;
    rhs.push_back(P.rup[i]);
  }
  if (nr == 0) return;
  rptr.assign(nr + 1, 0);
  for (int k = 0; k < nk; ++k) {
    const int j = cols[k];
    for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
      const int ri = row_map[P.cidx[q]];
      if (ri >= 0 && P.cval[q] > 1e-12) ++rptr[ri + 1];
    }
  }
  for (int i = 0; i < nr; ++i) rptr[i + 1] += rptr[i];
  rcol.resize(rptr[nr]);
  rval.resize(rptr[nr]);
  std::vector<int> pos(rptr.begin(), rptr.end() - 1);
  for (int k = 0; k < nk; ++k) {
    const int j = cols[k];
    for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
      const int ri = row_map[P.cidx[q]];
      if (ri >= 0 && P.cval[q] > 1e-12) {
        const int p = pos[ri]++;
        rcol[p] = k;
        rval[p] = P.cval[q];
      }
    }
  }
}

// Lifted 0-1 knapsack cover inequalities (Balas 1975; Gu, Nemhauser & Savelsbergh 1998).
// Uses exact sequence-independent up-lifting via descending cover prefix sums S_h = sum_{r=1}^h a_(r):
// since sum_{r in C} a_r = b + lambda > b, setting alpha_j = max{h < |C| : a_j >= S_h} is valid for
// all j not in C simultaneously (concavity of S_h guarantees sum a_j >= S_K > S_K - lambda).
int separate_knapsack_covers(const KnapsackRows& K, const std::vector<double>& x, double min_viol, int max_cuts,
                             CutPool& out) {
  if (K.nr == 0 || max_cuts <= 0 || static_cast<int>(x.size()) != K.nk) return 0;
  int added = 0;
  std::vector<char> in_c(K.nk, 0);
  std::set<std::vector<std::pair<int, int>>> seen;
  struct Item {
    int k;
    double a, xv;
  };
  std::vector<Item> sup, cover;
  std::vector<double> pref;
  std::vector<std::pair<int, int>> cut_entries;

  for (int ri = 0; ri < K.nr && added < max_cuts; ++ri) {
    const double b = K.rhs[ri];
    const double b_tol = 1e-7 * std::max(1.0, b);
    const int beg = K.rptr[ri], end = K.rptr[ri + 1];
    double act = 0.0;
    sup.clear();
    for (int q = beg; q < end; ++q) {
      const int k = K.rcol[q];
      const double xv = x[k];
      if (xv > 1e-4) {
        act += K.rval[q] * xv;
        sup.push_back({k, K.rval[q], xv});
      }
    }
    if (act < 0.70 * b || sup.size() < 2) continue;

    for (int mode = 0; mode < 2 && added < max_cuts; ++mode) {
      if (mode == 0) {
        std::sort(sup.begin(), sup.end(), [](const Item& u, const Item& v) {
          return (1.0 - u.xv) / u.a < (1.0 - v.xv) / v.a;
        });
      } else {
        std::sort(sup.begin(), sup.end(), [](const Item& u, const Item& v) {
          return u.xv > v.xv || (u.xv == v.xv && u.a > v.a);
        });
      }
      cover.clear();
      double csum = 0.0;
      for (const auto& it : sup) {
        cover.push_back(it);
        csum += it.a;
        if (csum > b + b_tol) break;
      }
      if (csum <= b + b_tol || cover.size() < 2) continue;
      // Minimize cover: drop smallest coefficients that still leave sum > b + b_tol.
      std::sort(cover.begin(), cover.end(), [](const Item& u, const Item& v) { return u.a < v.a; });
      std::vector<Item> min_c;
      min_c.reserve(cover.size());
      for (const auto& it : cover) {
        if (csum - it.a > b + b_tol) csum -= it.a;
        else min_c.push_back(it);
      }
      if (min_c.size() < 2 || csum <= b + b_tol) continue;

      const int cs = static_cast<int>(min_c.size());
      const double rhs = static_cast<double>(cs - 1);
      double base_lhs = 0.0;
      for (const auto& it : min_c) base_lhs += it.xv;
      if (base_lhs < rhs - 0.5) continue;

      // Sort descending for exact sequence-independent prefix sums S_1..S_{cs-1}.
      std::sort(min_c.begin(), min_c.end(), [](const Item& u, const Item& v) { return u.a > v.a; });
      pref.resize(static_cast<size_t>(cs));
      pref[0] = 0.0;
      for (int h = 1; h < cs; ++h) pref[static_cast<size_t>(h)] = pref[static_cast<size_t>(h - 1)] + min_c[h - 1].a;

      cut_entries.clear();
      double lhs = 0.0, norm2 = 0.0;
      for (const auto& it : min_c) {
        in_c[it.k] = 1;
        cut_entries.push_back({it.k, 1});
        lhs += it.xv;
        norm2 += 1.0;
      }
      const double s1 = pref[1];
      for (int q = beg; q < end; ++q) {
        const int k = K.rcol[q];
        if (in_c[k]) continue;
        const double ak = K.rval[q];
        if (ak < s1) continue;
        // Largest h in 1..cs-1 with ak >= pref[h].
        auto it = std::upper_bound(pref.begin() + 1, pref.end(), ak);
        const int h = static_cast<int>(it - pref.begin()) - 1;
        if (h >= 1) {
          cut_entries.push_back({k, h});
          lhs += static_cast<double>(h) * x[k];
          norm2 += static_cast<double>(h) * static_cast<double>(h);
        }
      }
      for (const auto& it : min_c) in_c[it.k] = 0;

      const double viol = lhs - rhs;
      if (viol < min_viol || norm2 == 0.0) continue;
      std::sort(cut_entries.begin(), cut_entries.end());
      if (!seen.insert(cut_entries).second) continue;

      for (const auto& e : cut_entries) {
        out.idx.push_back(e.first);
        out.val.push_back(static_cast<double>(e.second));
      }
      out.ptr.push_back(static_cast<int>(out.idx.size()));
      out.lo.push_back(-INF);
      out.up.push_back(rhs);
      out.efficacy.push_back(viol / std::sqrt(norm2));
      out.is_int.push_back(1);
      ++added;
    }
  }
  return added;
}

// Single-row c-MIR cuts on nonnegative knapsack rows sum a_j x_j <= b (Marchand & Wolsey 2001).
// For any divisor d > 0 with beta = floor(b/d), f = b/d - beta in (0, 1):
//   sum_j (floor(a_j/d) + max(0, frac(a_j/d) - f) / (1 - f)) x_j <= beta.
int separate_knapsack_mir(const KnapsackRows& K, const std::vector<double>& x, double min_viol, int max_cuts,
                          CutPool& out) {
  if (K.nr == 0 || max_cuts <= 0 || static_cast<int>(x.size()) != K.nk) return 0;
  int added = 0;
  std::vector<std::pair<double, double>> divs;  // (|x - 0.5|, a_k)
  std::vector<double> tried_d;
  std::vector<std::pair<int, double>> entries;

  for (int ri = 0; ri < K.nr && added < max_cuts; ++ri) {
    const double b = K.rhs[ri];
    const int beg = K.rptr[ri], end = K.rptr[ri + 1];
    double act = 0.0;
    divs.clear();
    for (int q = beg; q < end; ++q) {
      const int k = K.rcol[q];
      const double xv = x[k];
      if (xv > 1e-6) act += K.rval[q] * xv;
      if (xv >= 0.05 && xv <= 0.95 && K.rval[q] > 1e-9) divs.push_back({std::fabs(xv - 0.5), K.rval[q]});
    }
    if (act < 0.85 * b || divs.empty()) continue;
    std::sort(divs.begin(), divs.end());
    tried_d.clear();
    int per_row = 0;
    for (const auto& pr : divs) {
      if (added >= max_cuts || per_row >= 3 || tried_d.size() >= 12) break;
      const double d = pr.second;
      bool dup = false;
      for (double td : tried_d)
        if (std::fabs(td - d) <= 1e-6 * std::max(1.0, td)) dup = true;
      if (dup) continue;
      tried_d.push_back(d);

      const double bd = b / d;
      const double beta = std::floor(bd);
      const double f = bd - beta;
      if (beta < 0.0 || f < 0.02 || f > 0.98) continue;
      const double inv_1mf = 1.0 / (1.0 - f);

      double max_pi = 0.0;
      entries.clear();
      for (int q = beg; q < end; ++q) {
        const double ad = K.rval[q] / d;
        const double u = std::floor(ad);
        const double fj = ad - u;
        const double pi = u + (fj > f ? (fj - f) * inv_1mf : 0.0);
        if (pi > 1e-12) {
          entries.push_back({K.rcol[q], pi});
          if (pi > max_pi) max_pi = pi;
        }
      }
      if (entries.empty() || max_pi < 1e-9 || !std::isfinite(max_pi)) continue;
      // Dropping tiny positive coefficients on a <= inequality with x >= 0 strictly relaxes (safe).
      const double tiny = 1e-6 * max_pi;
      double min_pi = INF, lhs = 0.0, norm2 = 0.0;
      size_t w = 0;
      for (size_t e = 0; e < entries.size(); ++e) {
        if (entries[e].second < tiny) continue;
        entries[w++] = entries[e];
        min_pi = std::min(min_pi, entries[e].second);
        lhs += entries[e].second * x[entries[e].first];
        norm2 += entries[e].second * entries[e].second;
      }
      entries.resize(w);
      if (entries.empty() || max_pi / min_pi > 1e6) continue;

      const double inv = 1.0 / max_pi;
      const double b_norm = beta * inv;
      const double viol = lhs * inv - b_norm;
      const double norm = std::sqrt(norm2) * inv;
      const double eff = norm > 0.0 ? viol / norm : 0.0;
      if (viol < min_viol || eff < 0.002) continue;

      std::sort(entries.begin(), entries.end());
      for (const auto& e : entries) {
        out.idx.push_back(e.first);
        out.val.push_back(e.second * inv);
      }
      out.ptr.push_back(static_cast<int>(out.idx.size()));
      out.lo.push_back(-INF);
      out.up.push_back(b_norm + 1e-9 * (1.0 + std::fabs(b_norm)));
      out.efficacy.push_back(eff);
      out.is_int.push_back(0);
      ++added;
      ++per_row;
    }
  }
  return added;
}

// Gomory mixed-integer (GMI) cuts (Balas, Ceria, Cornuejols & Natraj 1996).
// From tableau row  x_B + sum_{v in N} a_v (x_v - x_bar_v) = x_bar_B  with x_B binary and
// f0 = x_bar_B - floor(x_bar_B) in [0.05, 0.95], shifts each nonbasic v to s_v >= 0, forms
// sum pi_v s_v >= 1, and substitutes row logicals r_i = sum_k A_{ik} x_k back into x-space.
int separate_gmi(LpSolver& lp, const std::vector<char>& row_int, const std::vector<double>& x, double min_viol,
                 int max_cuts, int max_nnz, CutPool& out) {
  const int m = lp.rows(), nk = lp.cols();
  if (m <= 0 || nk <= 0 || max_cuts <= 0 || static_cast<int>(x.size()) != nk) return 0;

  std::vector<int> cands;
  for (int k = 0; k < nk; ++k) {
    const double f0 = x[k] - std::floor(x[k]);
    if (f0 >= 0.05 && f0 <= 0.95 && lp.is_col_basic(k)) cands.push_back(k);
  }
  if (cands.empty()) return 0;
  std::sort(cands.begin(), cands.end(), [&](int a, int b) {
    const double fa = std::fabs((x[a] - std::floor(x[a])) - 0.5);
    const double fb = std::fabs((x[b] - std::floor(x[b])) - 0.5);
    return fa < fb || (fa == fb && a < b);
  });
  const int try_limit = std::min<int>(static_cast<int>(cands.size()), std::max(max_cuts * 3, 60));

  std::vector<int> nb_var, rcols, gtouched;
  std::vector<double> nb_val, nb_lo, nb_up, rvals, gamma(nk, 0.0);
  std::vector<char> nb_st, gmark(nk, 0);
  std::vector<std::pair<int, double>> entries;

  auto clear_gamma = [&]() {
    for (int col : gtouched) gamma[col] = 0.0, gmark[col] = 0;
    gtouched.clear();
  };
  auto add_gamma = [&](int col, double v) {
    if (!gmark[col]) {
      gmark[col] = 1;
      gtouched.push_back(col);
    }
    gamma[col] += v;
  };

  int added = 0;
  for (int ci = 0; ci < try_limit && added < max_cuts; ++ci) {
    const int k = cands[ci];
    const double f0 = x[k] - std::floor(x[k]);
    if (f0 < 0.05 || f0 > 0.95) continue;
    if (!lp.get_tableau_row(k, nb_var, nb_val, nb_st, nb_lo, nb_up)) continue;

    double beta = 1.0;
    bool ok = true;
    for (size_t t = 0; t < nb_var.size(); ++t) {
      const int v = nb_var[t];
      const double av = nb_val[t];
      const char sv = nb_st[t];
      const double lv = nb_lo[t], uv = nb_up[t];
      if (lv == uv) continue;  // fixed variable (e.g., equality row logical or globally fixed col): s_v == 0
      if (sv == 3) {           // free nonbasic variable with nonzero tableau coefficient: skip row
        ok = false;
        break;
      }
      const bool at_lower = (sv == 1);
      const double bnd = at_lower ? lv : uv;
      if (!std::isfinite(bnd)) {
        ok = false;
        break;
      }
      const double alpha = at_lower ? av : -av;
      const bool v_int = (v >= m) ? (std::floor(bnd) == bnd)
                                  : (v < static_cast<int>(row_int.size()) && row_int[v] != 0 && std::floor(bnd) == bnd);
      double pi = 0.0;
      if (v_int) {
        const double fv = alpha - std::floor(alpha);
        if (fv < 1e-12 || fv > 1.0 - 1e-12) continue;
        pi = (fv <= f0) ? (fv / f0) : ((1.0 - fv) / (1.0 - f0));
      } else {
        if (std::fabs(alpha) < 1e-12) continue;
        pi = (alpha > 0.0) ? (alpha / f0) : (-alpha / (1.0 - f0));
      }
      if (pi <= 1e-12 || !std::isfinite(pi)) continue;
      if (v >= m) {
        const int col = v - m;
        if (at_lower) {
          add_gamma(col, pi);
          beta += pi * lv;
        } else {
          add_gamma(col, -pi);
          beta -= pi * uv;
        }
      } else {
        lp.get_row(v, rcols, rvals);
        if (at_lower) {
          for (size_t q = 0; q < rcols.size(); ++q) add_gamma(rcols[q], pi * rvals[q]);
          beta += pi * lv;
        } else {
          for (size_t q = 0; q < rcols.size(); ++q) add_gamma(rcols[q], -pi * rvals[q]);
          beta -= pi * uv;
        }
      }
    }
    if (!ok || gtouched.empty() || !std::isfinite(beta)) {
      clear_gamma();
      continue;
    }

    double max_abs = 0.0;
    for (int col : gtouched) max_abs = std::max(max_abs, std::fabs(gamma[col]));
    if (max_abs < 1e-9 || !std::isfinite(max_abs)) {
      clear_gamma();
      continue;
    }
    // Safe small-coefficient relaxation: for 0 <= x_col <= 1, dropping tiny positive gamma[col] lowers beta by gamma[col].
    const double tiny = 1e-6 * max_abs;
    entries.clear();
    double min_abs = INF, lhs = 0.0, norm2 = 0.0;
    for (int col : gtouched) {
      const double g = gamma[col];
      if (std::fabs(g) < tiny) {
        if (g > 0.0) beta -= g;
        continue;
      }
      entries.push_back({col, g});
      min_abs = std::min(min_abs, std::fabs(g));
      lhs += g * x[col];
      norm2 += g * g;
    }
    clear_gamma();
    if (entries.empty() || static_cast<int>(entries.size()) > max_nnz) continue;
    if (max_abs / min_abs > 1e6) continue;  // dynamism safeguard

    const double inv = 1.0 / max_abs;
    const double b_norm = beta * inv;
    const double viol = b_norm - lhs * inv;
    const double norm = std::sqrt(norm2) * inv;
    const double eff = norm > 0.0 ? viol / norm : 0.0;
    if (viol < min_viol || eff < 0.002) continue;

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& e : entries) {
      out.idx.push_back(e.first);
      out.val.push_back(e.second * inv);
    }
    out.ptr.push_back(static_cast<int>(out.idx.size()));
    out.lo.push_back(b_norm - 1e-9 * (1.0 + std::fabs(b_norm)));
    out.up.push_back(INF);
    out.efficacy.push_back(eff);
    out.is_int.push_back(0);
    ++added;
  }
  return added;
}

}  // namespace scsp
