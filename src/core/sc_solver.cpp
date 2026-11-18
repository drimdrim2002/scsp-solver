// Set covering heuristic (P1).
//
// Sources (papers only, see docs/references.md):
//   * Caprara, Fischetti & Toth 1999: three-phase Lagrangian heuristic (subgradient
//     phase with a core problem and pricing, heuristic phase with a greedy on
//     Lagrangian costs, column fixing) and the refinement loop around it.
//   * Beasley 1987, 1990: Lagrangian lower bound, reduced-cost fixing, tree search
//     over an uncovered row with Lagrangian bounds; Balas & Ho 1980: row branching.
//   * Yagiura, Kishida & Ibaraki 2006: flip neighbourhoods for local search. Only the
//     "add one column, drop what became redundant" move is implemented here.
// Side constraints (sum a x <= b, a >= 0) get their own multipliers in the
// relaxation; the greedy, the local search and the enumeration never exceed b.
#include "scsp/sc_solver.hpp"

#include "scsp/bnb.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <utility>

#include "scsp/presolve.hpp"

namespace scsp {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static const bool DEBUG_SUB = std::getenv("SCSP_DEBUG_SUB") != nullptr;
static const bool DEBUG_TREE = std::getenv("SCSP_DEBUG_TREE") != nullptr;
static const bool NO_LP_ADOPT = std::getenv("SCSP_NO_LP_ADOPT") != nullptr;  // experiment: workers ignore the root LP

struct Shared {
  const Problem& P;
  const SolveOptions& opt;
  Clock::time_point t0;
  std::mutex mu;
  std::vector<int> best;  // reduced columns
  std::vector<TracePoint> trace;
  std::atomic<double> ub{INF}, lb{-INF};
  std::atomic<bool> stop{false}, proven{false};
  std::atomic<long long> sub_iters{0}, greedy_calls{0}, ls_moves{0}, enum_nodes{0}, passes{0};
  double last_lb_trace = -1.0;
  // root LP published by the branch-and-bound thread (written once, then lp_ready is set)
  std::vector<double> lp_u, lp_v, lp_rc;
  double lp_bound = -INF;
  std::atomic<bool> lp_ready{false}, infeasible{false};
  BnbResult bnb;
  bool bnb_ran = false;
  double tree_gave_up = -1.0;  // time at which the tree thread became a heuristic worker (-1: it did not)

  Shared(const Problem& p, const SolveOptions& o, Clock::time_point t) : P(p), opt(o), t0(t) {}

  double elapsed() const { return std::chrono::duration<double>(Clock::now() - t0).count(); }
  bool timeup() const { return stop.load(std::memory_order_relaxed) || elapsed() >= opt.time_limit; }

  // A solution only counts as an improvement if its cost is <= thr().
  double thr() const {
    const double u = ub.load(std::memory_order_relaxed);
    if (u == INF) return INF;
    return P.integral ? u - 1.0 + 1e-6 : u - 1e-9 * std::max(1.0, std::fabs(u));
  }
  double shown_bound(double l, double u) const {
    if (l == -INF) return NaN;
    double b = l + P.offset;
    if (P.integral) b = std::ceil(b - 1e-6);
    return u == INF ? b : std::min(b, u + P.offset);
  }
  void check_done() {  // call with mu held
    const double u = ub.load(), l = lb.load();
    if (u == INF || l == -INF) return;
    const double tol = std::max(opt.mip_gap * std::fabs(u + P.offset), 1e-9 * std::max(1.0, std::fabs(u)));
    const bool done = (P.integral && std::ceil(l - 1e-6) >= u - 1e-9) || (u - l <= tol);
    if (done) {
      proven = true;
      stop = true;
    }
  }
  bool offer(const std::vector<int>& sol, double cost) {
    if (cost >= ub.load(std::memory_order_relaxed)) return false;
    std::lock_guard<std::mutex> g(mu);
    const double u = ub.load();
    if (cost >= u - 1e-12 * std::max(1.0, std::fabs(cost))) return false;
    best = sol;
    ub = cost;
    trace.push_back({elapsed(), cost + P.offset, shown_bound(lb.load(), cost), 0});
    check_done();
    return true;
  }
  void offer_lb(double l) {
    if (l <= lb.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> g(mu);
    if (l <= lb.load()) return;
    lb = l;
    const double t = elapsed(), u = ub.load();
    if (t - last_lb_trace >= 0.1) {
      last_lb_trace = t;
      trace.push_back({t, u == INF ? NaN : u + P.offset, shown_bound(l, u), 1});
    }
    check_done();
  }
  void set_optimal() {
    std::lock_guard<std::mutex> g(mu);
    if (ub.load() == INF) return;
    lb = ub.load();
    proven = true;
    stop = true;
  }
};

class Worker {
 public:
  Worker(Shared& s, int id_)
      : S(s), P(s.P), id(id_), rng(s.opt.seed * 7919ULL + 104729ULL * static_cast<unsigned>(id_) + 1ULL),
        n(P.n), m(P.m), ns(P.ns), explorer(id_ % 2 == 1) {
    u.assign(m, 0.0);
    v.assign(ns, 0.0);
    rc.assign(n, INF);
    cap = P.side_rhs;
    ract.assign(m, 1);
    mact = m;
    dead.assign(n, 0);
    infix.assign(n, 0);
    cpos.assign(n, -1);
    cr_ptr.assign(m + 1, 0);
    sub.assign(m, 0.0);
    ssub.assign(ns, 0.0);
    unc.assign(m, 0);
    cnt.assign(m, 0);
    sumid.assign(m, 0);
    uniq.assign(n, 0);
    dec.assign(n, 0);
    insol.assign(n, 0);
    usage.assign(ns, 0.0);
  }

  void run() {
    begin();
    while (!S.timeup() && pass()) {
    }
  }

  void begin() {
    init_multipliers();
    pi = pi_min();
    noise = 0.15 * id;
  }

  // One round of the Lagrangian phases and the neighbourhood search. False when the worker cannot go on.
  bool pass() {
    if (S.timeup()) return false;
    const double t_pass = S.elapsed(), ub_pass = S.ub.load();
    pass_cost = INF;
    pass_best.clear();
    three_phase();
    if (explorer && !pass_best.empty() && (own.empty() || own_stall >= 60)) {
      own = pass_best;  // restart the explorer from a fresh Lagrangian solution
      own_cost = pass_cost;
      own_stall = 0;
      phi = 0.3;
    }
    ++S.passes;
    if (S.timeup() || !have_root) return false;
    const bool cft_improved = S.ub.load() < ub_pass;
    bool lns_improved = false;
    if (S.opt.enumerate && S.ub.load() < INF) {
      // a wide gap makes the bound-driven searches unproductive: leave the time to the Lagrangian passes
      const double rel_gap = (S.ub.load() - lb_root) / std::max(std::fabs(S.ub.load()), 1e-12);
      const bool wide = rel_gap > 0.05;
      if (pass_no == 0) share = wide ? 0.15 : 0.5;
      if (id == 0 && pass_no >= next_enum && (!wide || enum_attempt == 0)) {
        enumerate(100000ULL << (2 * std::min(enum_attempt, 12)));
        ++enum_attempt;
        next_enum = 2 * pass_no + 1;
      }
      const double t0 = S.elapsed();
      const double budget = std::max(0.02, share / (1.0 - share) * (t0 - t_pass));
      while (!S.timeup() && S.elapsed() - t0 < budget) lns_improved |= lns_step();
      if (S.timeup()) return false;
      if (lns_improved && !cft_improved) share = std::min(0.9, share + 0.1);
      if (cft_improved && !lns_improved) share = std::max(0.1, share - 0.1);
    }
    ++pass_no;
    const double ubn = S.ub.load();
    if (ubn < seen_ub - 1e-12 * std::max(1.0, std::fabs(ubn))) {
      pi = pi_min();
    } else {
      pi *= 1.1;
      if (pi > 0.95) {
        pi = pi_min();
        noise = std::min(2.0, noise + 0.25);
      }
    }
    seen_ub = ubn;
    refine_fix(pi, noise);
    return true;
  }

 private:
  Shared& S;
  const Problem& P;
  const int id;
  std::mt19937_64 rng;
  const int n, m, ns;

  std::vector<double> u, v, rc, cap;
  std::vector<char> ract, dead, infix;
  std::vector<int> fixed;
  double fixed_cost = 0.0;
  int mact = 0;
  std::vector<int> core, cpos, cr_ptr, cr_col;
  bool core_full = false;
  int n_elig = 0;
  double lambda = 0.1;
  double est_ub = INF;

  std::vector<double> u_root, v_root, rc_root;
  double lb_root = -INF;
  bool have_root = false, lp_adopted = false;
  // state of run() between passes
  double pi = 0.3, noise = 0.0, seen_ub = INF;
  double share = 0.5;  // part of the time given to the neighbourhood search
  int enum_attempt = 0;
  long long pass_no = 0, next_enum = 0;
  std::vector<int> ls_cand;

  std::vector<double> sub, ssub, gam, capl;
  std::vector<int> mu;
  std::vector<char> unc;
  std::vector<std::pair<double, int>> heap;

  // local-search state (all zero between calls)
  std::vector<int> cnt, uniq, dec, cur, touched, removable;
  std::vector<long long> sumid;
  std::vector<char> insol;
  std::vector<double> usage;
  double cur_cost = 0.0;

  double pi_min() const { return 0.3 + 0.1 * (id % 3); }
  double rnd() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

  void init_multipliers() {
    for (int i = 0; i < m; ++i) {
      double best = INF;
      for (int q = P.row_ptr[i]; q < P.row_ptr[i + 1]; ++q) {
        const int j = P.row_col[q];
        const int len = P.col_ptr[j + 1] - P.col_ptr[j];
        best = std::min(best, P.cost[j] / std::max(1, len));
      }
      u[i] = best == INF ? 0.0 : best;
      if (id > 0) u[i] *= 0.9 + 0.2 * rnd();
    }
  }

  // ---------------------------------------------------------------- pricing
  // Reduced costs of every eligible column, a new core, and the Lagrangian value
  // over all columns (a valid lower bound of the current residual problem).
  double price() {
    double L = 0.0, usum = 0.0;
    for (int i = 0; i < m; ++i)
      if (ract[i]) usum += u[i];
    L = usum;
    for (int t = 0; t < ns; ++t) L -= v[t] * cap[t];
    n_elig = 0;
    for (int j = 0; j < n; ++j) {
      rc[j] = INF;
      if (dead[j] || infix[j]) continue;
      double r = P.cost[j];
      int k = 0;
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
        const int i = P.col_row[q];
        if (ract[i]) {
          r -= u[i];
          ++k;
        }
      }
      if (!k) continue;
      bool ok = true;
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) {
        if (P.cs_val[q] > cap[P.cs_row[q]] + 1e-9) {
          ok = false;
          break;
        }
        r += v[P.cs_row[q]] * P.cs_val[q];
      }
      if (!ok) continue;
      rc[j] = r;
      ++n_elig;
      if (r < 0) L += r;
    }

    for (int j : core) cpos[j] = -1;
    core.clear();
    if (n_elig <= 15 * mact) {
      core_full = true;
      for (int j = 0; j < n; ++j)
        if (rc[j] < INF) core.push_back(j);
    } else {
      core_full = false;
      for (int i = 0; i < m; ++i) {
        if (!ract[i]) continue;
        int bj[5];
        double br[5];
        int nb = 0;
        for (int q = P.row_ptr[i]; q < P.row_ptr[i + 1]; ++q) {
          const int j = P.row_col[q];
          const double r = rc[j];
          if (r == INF) continue;
          if (nb == 5 && r >= br[4]) continue;
          int a = nb < 5 ? nb++ : 4;
          while (a > 0 && br[a - 1] > r) {
            br[a] = br[a - 1];
            bj[a] = bj[a - 1];
            --a;
          }
          br[a] = r;
          bj[a] = j;
        }
        for (int a = 0; a < nb; ++a)
          if (cpos[bj[a]] == -1) {
            cpos[bj[a]] = -2;
            core.push_back(bj[a]);
          }
      }
      const double thr = mact > 0 ? 0.05 * usum / mact : 0.0;
      std::vector<int> extra;
      for (int j = 0; j < n; ++j)
        if (rc[j] < thr && cpos[j] == -1) extra.push_back(j);
      const size_t lim = static_cast<size_t>(5) * static_cast<size_t>(mact);
      if (extra.size() > lim) {
        std::nth_element(extra.begin(), extra.begin() + static_cast<long>(lim), extra.end(),
                         [&](int a, int b) { return rc[a] < rc[b]; });
        extra.resize(lim);
      }
      core.insert(core.end(), extra.begin(), extra.end());
      std::sort(core.begin(), core.end());
    }
    for (size_t p = 0; p < core.size(); ++p) cpos[core[p]] = static_cast<int>(p);

    // rows -> core positions (active rows only)
    std::fill(cr_ptr.begin(), cr_ptr.end(), 0);
    for (int j : core)
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q)
        if (ract[P.col_row[q]]) ++cr_ptr[P.col_row[q] + 1];
    for (int i = 0; i < m; ++i) cr_ptr[i + 1] += cr_ptr[i];
    cr_col.resize(static_cast<size_t>(cr_ptr[m]));
    std::vector<int> pos(cr_ptr.begin(), cr_ptr.end() - 1);
    for (size_t p = 0; p < core.size(); ++p) {
      const int j = core[p];
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q)
        if (ract[P.col_row[q]]) cr_col[static_cast<size_t>(pos[P.col_row[q]]++)] = static_cast<int>(p);
    }
    return L;
  }

  void save_root(double L) {
    lb_root = L;
    u_root = u;
    v_root = v;
    rc_root = rc;
    have_root = true;
    S.offer_lb(L);
  }

  double target_ub(double L) const {
    const double g = S.ub.load(std::memory_order_relaxed);
    if (g < INF) return g - fixed_cost;
    if (est_ub < INF) return est_ub - fixed_cost;
    return L + 0.1 * std::max(std::fabs(L), 1.0);
  }

  // ------------------------------------------------------------ subgradient
  // heur = false: subgradient phase, returns the bound over all columns at the best multipliers.
  // heur = true: heuristic phase, a greedy solution is built at every multiplier vector.
  double subgradient(bool root, int window, int max_iter, bool heur) {
    std::vector<double> ubest = u, vbest = v;
    const std::vector<double> u0 = u, v0 = v;
    double best = -INF, ref = -INF, Lf = -INF;
    int T = 10, since = 0, noimp = 0, it = 0;
    const int patience = root ? 20 : 10;
    const double lam_min = root ? 0.005 : 0.02;
    if (!heur) lambda = root ? 1.0 : 0.3;
    else lambda = std::max(lambda, 0.05);
    std::vector<int> chosen;
    long long iters = 0;
    for (int round = 0;; ++round) {
      while (it < max_iter) {
        ++it;
        if (S.timeup()) break;
        ++iters;
        double L = 0.0;
        for (int i = 0; i < m; ++i)
          if (ract[i]) {
            L += u[i];
            sub[i] = 1.0;
          }
        for (int t = 0; t < ns; ++t) {
          L -= v[t] * cap[t];
          ssub[t] = -cap[t];
        }
        for (int j : core) {
          double r = P.cost[j];
          for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) r -= u[P.col_row[q]];
          for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) r += v[P.cs_row[q]] * P.cs_val[q];
          if (r < 0) {
            L += r;
            for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) sub[P.col_row[q]] -= 1.0;
            for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) ssub[P.cs_row[q]] += P.cs_val[q];
          }
        }
        if (best == -INF || L > best + 1e-10 * std::fabs(best)) {
          best = L;
          ubest = u;
          vbest = v;
          noimp = 0;
          if (root && core_full && !heur && (it % 25 == 0)) S.offer_lb(L);
        } else if (!heur && ++noimp >= patience) {
          // step control of Beasley 1990: halve the step factor when the bound stalls
          lambda *= 0.5;
          noimp = 0;
          u = ubest;
          v = vbest;
          if (lambda < lam_min) break;
          continue;
        }
        if (!heur && !core_full && ++since >= T) {
          // pricing of Caprara, Fischetti & Toth 1999: rebuild the core from all columns
          since = 0;
          const double Lp = price();
          if (root && Lp > lb_root) save_root(Lp);
          const double d = (L - Lp) / std::max(std::fabs(target_ub(L)), 1e-12);
          if (d <= 1e-6) T *= 10;
          else if (d <= 0.02) T *= 5;
          else if (d <= 0.2) T *= 2;
          else T = 10;
          T = std::min(T, std::max(10, std::min(1000, n / 3)));
          if (d > 1e-3) {  // the core changed: bounds of the old core are not comparable
            best = -INF;
            ref = -INF;
            noimp = 0;
          }
        }
        const double gap = target_ub(L) - L;
        if (gap <= 0) break;
        double norm = 0.0;
        for (int i = 0; i < m; ++i)
          if (ract[i]) {
            if (sub[i] < 0 && u[i] <= 0) sub[i] = 0.0;
            norm += sub[i] * sub[i];
          }
        for (int t = 0; t < ns; ++t) {
          if (ssub[t] < 0 && v[t] <= 0) ssub[t] = 0.0;
          norm += ssub[t] * ssub[t];
        }
        if (norm <= 0) break;
        const double step = lambda * gap / norm;
        for (int i = 0; i < m; ++i)
          if (ract[i]) u[i] = std::max(0.0, u[i] + step * sub[i]);
        for (int t = 0; t < ns; ++t) v[t] = std::max(0.0, v[t] + step * ssub[t]);

        if (heur) {
          chosen.clear();
          if (greedy(chosen, INT_MAX)) evaluate(chosen);
          continue;
        }
        if (root && it % 50 == 0 && core.size() > 0) {  // keep the incumbent moving while the bound converges
          chosen.clear();
          if (greedy(chosen, INT_MAX)) evaluate(chosen);
        }
        if (DEBUG_SUB && id == 0 && it % 200 == 0)
          std::fprintf(stderr, "it=%d L=%.4f best=%.4f lam=%.3g norm=%.3g gap=%.3g core=%zu T=%d\n", it, L, best, lambda, norm, gap, core.size(), T);
        if (it % window == 0) {
          if (best > -INF && ref > -INF && best - ref <= 1e-3 * std::fabs(best)) break;
          ref = best;
        }
      }
      if (heur) break;
      if (best > -INF) {
        u = ubest;
        v = vbest;
      }
      Lf = price();
      if (root && Lf > lb_root) save_root(Lf);
      if (DEBUG_SUB && id == 0) std::fprintf(stderr, "round=%d it=%d core=%.4f full=%.4f elig=%d core_size=%zu\n", round, it, best, Lf, n_elig, core.size());
      // the core bound is only trustworthy once pricing confirms it
      if (core_full || round >= 15 || it >= max_iter || S.timeup() || best - Lf <= 2e-3 * std::fabs(best)) break;
      ubest = u;
      vbest = v;
      best = -INF;
      ref = -INF;
      noimp = 0;
      lambda = root ? 0.25 : 0.15;
    }
    S.sub_iters += iters;
    if (heur) {
      u = u0;
      v = v0;
      return best;
    }
    return Lf;
  }

  // ----------------------------------------------------------------- greedy
  // Covers the active rows with core columns, cheapest Lagrangian score first
  // (score gamma/mu if gamma > 0, gamma*mu otherwise). Scores only grow while
  // rows get covered, so a heap with lazy re-insertion is exact.
  bool greedy(std::vector<int>& chosen, int max_cols) {
    ++S.greedy_calls;
    const size_t nc = core.size();
    gam.resize(nc);
    mu.resize(nc);
    heap.clear();
    auto score = [&](size_t p) { return gam[p] > 0 ? gam[p] / mu[p] : gam[p] * mu[p]; };
    for (size_t p = 0; p < nc; ++p) {
      const int j = core[p];
      double g = P.cost[j];
      int k = 0;
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q)
        if (ract[P.col_row[q]]) {
          g -= u[P.col_row[q]];
          ++k;
        }
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) g += v[P.cs_row[q]] * P.cs_val[q];
      gam[p] = g;
      mu[p] = k;
      if (k) heap.emplace_back(score(p), static_cast<int>(p));
    }
    std::make_heap(heap.begin(), heap.end(), std::greater<>());
    int left = mact, added = 0;
    for (int i = 0; i < m; ++i) unc[i] = ract[i];
    capl = cap;
    while (left > 0 && added < max_cols) {
      if (heap.empty()) return false;
      std::pop_heap(heap.begin(), heap.end(), std::greater<>());
      const auto top = heap.back();
      heap.pop_back();
      const size_t p = static_cast<size_t>(top.second);
      if (mu[p] == 0) continue;
      const double now = score(p);
      if (now > top.first + 1e-12 * std::max(1.0, std::fabs(now))) {
        heap.emplace_back(now, top.second);
        std::push_heap(heap.begin(), heap.end(), std::greater<>());
        continue;
      }
      const int j = core[p];
      bool ok = true;
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q)
        if (P.cs_val[q] > capl[P.cs_row[q]] + 1e-9) ok = false;
      if (!ok) continue;  // capacity only shrinks: never eligible again
      chosen.push_back(j);
      ++added;
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) capl[P.cs_row[q]] -= P.cs_val[q];
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
        const int i = P.col_row[q];
        if (!unc[i]) continue;
        unc[i] = 0;
        --left;
        for (int s = cr_ptr[i]; s < cr_ptr[i + 1]; ++s) {
          gam[static_cast<size_t>(cr_col[s])] += u[i];
          --mu[static_cast<size_t>(cr_col[s])];
        }
      }
    }
    return true;
  }

  // ------------------------------------------------- solution bookkeeping
  void add_col(int j) {
    insol[j] = 1;
    cur_cost += P.cost[j];
    for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
      const int i = P.col_row[q];
      if (cnt[i] == 1) --uniq[sumid[i]];
      ++cnt[i];
      sumid[i] += j;
      if (cnt[i] == 1) ++uniq[j];
    }
    for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) usage[P.cs_row[q]] += P.cs_val[q];
  }
  void remove_col(int j) {
    insol[j] = 0;
    cur_cost -= P.cost[j];
    for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
      const int i = P.col_row[q];
      if (cnt[i] == 1) --uniq[j];
      --cnt[i];
      sumid[i] -= j;
      if (cnt[i] == 1) ++uniq[sumid[i]];
    }
    for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) usage[P.cs_row[q]] -= P.cs_val[q];
  }
  void load(const std::vector<int>& a, const std::vector<int>& b) {
    cur.clear();
    cur_cost = 0.0;
    for (const auto* part : {&a, &b})
      for (int j : *part)
        if (!insol[j]) {
          add_col(j);
          cur.push_back(j);
        }
  }
  void unload() {
    for (int j : cur)
      if (insol[j]) remove_col(j);
    cur.clear();
    cur_cost = 0.0;
    for (int t = 0; t < ns; ++t) usage[t] = 0.0;
  }
  void drop_redundant() {
    std::sort(cur.begin(), cur.end(), [&](int a, int b) { return P.cost[a] > P.cost[b] || (P.cost[a] == P.cost[b] && a < b); });
    for (int j : cur)
      if (uniq[j] == 0) remove_col(j);
    compact_cur();
  }
  void compact_cur() {
    std::sort(cur.begin(), cur.end());
    cur.erase(std::unique(cur.begin(), cur.end()), cur.end());
    size_t w = 0;
    for (int j : cur)
      if (insol[j]) cur[w++] = j;
    cur.resize(w);
  }
  bool side_ok() const {
    for (int t = 0; t < ns; ++t)
      if (usage[t] > P.side_rhs[t] + 1e-9) return false;
    return true;
  }

  // Local search on the loaded solution: add one column, then drop every column
  // that became redundant (most expensive first); keep the move if the cost drops.
  void improve() {
    bool improved = true;
    long long moves = 0;
    for (int pass = 0; improved && pass < 30 && !S.timeup(); ++pass) {
      improved = false;
      for (int k : ls_cand) {
        if (insol[k] || dead[k]) continue;
        touched.clear();
        for (int q = P.col_ptr[k]; q < P.col_ptr[k + 1]; ++q) {
          const int i = P.col_row[q];
          if (cnt[i] == 1) {
            const int j = static_cast<int>(sumid[i]);
            if (dec[j]++ == 0) touched.push_back(j);
          }
        }
        if (touched.empty()) continue;
        removable.clear();
        double sum = 0.0;
        for (int j : touched) {
          if (dec[j] == uniq[j]) {
            removable.push_back(j);
            sum += P.cost[j];
          }
          dec[j] = 0;
        }
        const double ck = P.cost[k];
        if (sum <= ck + 1e-9 * std::max(1.0, ck)) continue;
        std::sort(removable.begin(), removable.end(), [&](int a, int b) { return P.cost[a] > P.cost[b]; });
        const double before = cur_cost;
        add_col(k);
        size_t nrem = 0;
        for (int j : removable)
          if (uniq[j] == 0) {
            remove_col(j);
            removable[nrem++] = j;
          }
        if (cur_cost < before - 1e-9 * std::max(1.0, before) && side_ok()) {
          cur.push_back(k);
          improved = true;
          ++moves;
        } else {
          for (size_t a = 0; a < nrem; ++a) add_col(removable[a]);
          remove_col(k);
          cur_cost = before;
        }
      }
    }
    S.ls_moves += moves;
    compact_cur();
  }

  // fixed + chosen -> remove redundant columns -> local search -> shared incumbent
  void evaluate(const std::vector<int>& chosen) {
    load(fixed, chosen);
    drop_redundant();
    const double ubn = S.ub.load(std::memory_order_relaxed);
    if (ubn == INF || cur_cost < est_ub) est_ub = std::min(est_ub, cur_cost);
    const double refc = explorer && pass_cost < INF ? std::max(ubn, pass_cost) : ubn;
    if (S.opt.local_search && have_root && (refc == INF || cur_cost <= refc * 1.01 + 1e-9)) improve();
    keep_current();
    unload();
  }

  // ---------------------------------------------------------- column fixing
  void fix_col(int j) {
    infix[j] = 1;
    fixed.push_back(j);
    fixed_cost += P.cost[j];
    for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
      const int i = P.col_row[q];
      if (ract[i]) {
        ract[i] = 0;
        u[i] = 0.0;
        --mact;
      }
    }
    for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) cap[P.cs_row[q]] -= P.cs_val[q];
  }
  bool fits(int j) const {
    for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q)
      if (P.cs_val[q] > cap[P.cs_row[q]] + 1e-9) return false;
    return true;
  }
  bool covers_active(int j) const {
    for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q)
      if (ract[P.col_row[q]]) return true;
    return false;
  }
  void unfix_all() {
    for (int j : fixed) infix[j] = 0;
    fixed.clear();
    fixed_cost = 0.0;
    std::fill(ract.begin(), ract.end(), 1);
    mact = m;
    cap = P.side_rhs;
  }

  // Columns with clearly negative reduced cost, then the first greedy picks.
  bool column_fix() {
    double usum = 0.0;
    for (int i = 0; i < m; ++i)
      if (ract[i]) usum += u[i];
    const double tol = 1e-3 * (mact > 0 ? usum / mact : 1.0);
    std::vector<int> js;
    for (int j : core)
      if (rc[j] < -tol) js.push_back(j);
    std::sort(js.begin(), js.end(), [&](int a, int b) { return rc[a] < rc[b]; });
    for (int j : js)
      if (covers_active(j) && fits(j)) fix_col(j);
    if (mact == 0) return true;
    price();
    std::vector<int> chosen;
    const bool ok = greedy(chosen, std::max(1, mact / 200));
    for (int j : chosen) fix_col(j);
    return ok;
  }

  // One run of the three phases from the current fixing down to an empty residual problem.
  void three_phase() {
    bool first = true;
    while (!S.timeup()) {
      if (mact == 0) {
        evaluate({});
        break;
      }
      const bool root = fixed.empty();
      const double L0 = price();
      if (root && !have_root) save_root(L0);
      if (first && S.ub.load() == INF) {
        std::vector<int> chosen;
        if (greedy(chosen, INT_MAX)) evaluate(chosen);
      }
      first = false;
      const double Lres = subgradient(root, root ? 300 : 100, root ? 100000 : 3000, false);
      if (root) apply_rc_fix();
      if (!explorer && fixed_cost + Lres > S.thr()) break;  // this residual problem cannot improve the incumbent
      subgradient(root, 0, root ? 250 : 80, true);
      if (S.timeup()) break;
      if (!column_fix()) break;
    }
  }

  // ------------------------------------------------- reduced-cost fixing
  bool apply_rc_fix() {
    if (!have_root || !fixed.empty()) return false;
    const double thr = S.thr();
    bool changed = false;
    if (thr < INF)
      for (int j = 0; j < n; ++j)
        if (!dead[j] && ((rc_root[j] > 0 && lb_root + rc_root[j] > thr) ||
                         (lp_adopted && S.lp_rc[j] > 0 && S.lp_bound + S.lp_rc[j] > thr))) {
          dead[j] = 1;
          changed = true;
        }
    if (!changed && !ls_cand.empty()) return false;
    ls_cand.clear();
    for (int j = 0; j < n; ++j)
      if (!dead[j] && rc_root[j] < INF) ls_cand.push_back(j);
    std::sort(ls_cand.begin(), ls_cand.end(), [&](int a, int b) { return rc_root[a] < rc_root[b] || (rc_root[a] == rc_root[b] && a < b); });
    if (ls_cand.size() > 60000) ls_cand.resize(60000);
    return changed;
  }

  // ------------------------------------------------------------ refinement
  // Fix the columns of the incumbent that contribute least to the gap until a
  // fraction pi of the rows is covered, then restart from the root multipliers.
  void refine_fix(double pi, double noise) {
    unfix_all();
    if (!lp_adopted && !NO_LP_ADOPT && S.lp_ready.load(std::memory_order_acquire)) {
      // The root LP only tightens reduced-cost fixing here. Its duals are a vertex of the dual
      // polyhedron (many zeros) and make worse greedy solutions than the subgradient multipliers.
      lp_adopted = true;
      ls_cand.clear();
      apply_rc_fix();
    }
    u = u_root;
    v = v_root;
    for (int i = 0; i < m; ++i) u[i] *= 0.9 + 0.2 * rnd();  // different multipliers, different greedy solutions
    std::vector<int> best;
    if (explorer && !own.empty()) {
      best = own;
      pi = 0.15;
      noise = 1.0;
    } else {
      std::lock_guard<std::mutex> g(S.mu);
      best = S.best;
    }
    if (best.empty()) return;
    std::vector<int> c(m, 0);
    for (int j : best)
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) ++c[P.col_row[q]];
    std::vector<std::pair<double, int>> order;
    order.reserve(best.size());
    for (int j : best) {
      if (dead[j] || rc_root[j] == INF) continue;
      double d = std::max(rc_root[j], 0.0);
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
        const int i = P.col_row[q];
        d += u_root[i] * (c[i] - 1) / c[i];
      }
      order.emplace_back(d * (1.0 + noise * rnd()) + 1e-12 * rnd(), j);
    }
    std::sort(order.begin(), order.end());
    const int target = static_cast<int>(pi * m);
    for (const auto& e : order) {
      if (m - mact >= target) break;
      if (fits(e.second) && covers_active(e.second)) fix_col(e.second);
    }
  }

  // ----------------------------------------------------------- enumeration
  // Depth-first search that branches on an uncovered row (Balas & Ho 1980;
  // Beasley 1987) over the residual problem left by the current fixing. The
  // bound of a node is the Lagrangian value of what is still uncovered, at the
  // multipliers (uu, vv) the search was started with:
  //   cost(chosen) + sum_{i uncovered} u_i - sum_t v_t capleft_t
  //                + sum_{j free} min(0, c_j + v a_j - sum_{i in j, uncovered} u_i).
  // With nothing fixed, a complete search proves that no solution better than the
  // incumbent exists. With a part of the incumbent fixed it is a large-neighbourhood
  // step that solves the freed part exactly.
  struct Enum {
    Worker& w;
    const Problem& P;
    const std::vector<double>* uu = nullptr;
    const std::vector<double>* vv = nullptr;
    std::vector<int> lr_ptr, lr_col, cnt, chosen, stack, rows;
    std::vector<char> state;  // 0 free, 1 chosen, 2 excluded, 3 not in the search
    std::vector<double> capl, rcur;
    double usum = 0.0, vterm = 0.0, neg = 0.0, cost = 0.0;
    unsigned long long nodes = 0, limit = 0;
    bool aborted = false;

    explicit Enum(Worker& w_) : w(w_), P(w_.P) {}

    // Residual problem of the worker's current fixing; L is its Lagrangian value at (u, v, rc).
    void build(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& rc, double L) {
      const int n = P.n, m = P.m;
      uu = &u;
      vv = &v;
      const double thr = w.cur_thr(), base = w.fixed_cost;
      lr_ptr.assign(m + 1, 0);
      lr_col.clear();
      state.assign(n, 3);
      rcur.assign(n, 0.0);
      cnt.assign(m, 1);
      rows.clear();
      usum = vterm = neg = 0.0;
      cost = base;
      for (int j = 0; j < n; ++j)
        if (!w.dead[j] && !w.infix[j] && rc[j] < INF && base + L + std::max(rc[j], 0.0) <= thr) {
          state[j] = 0;
          rcur[j] = rc[j];
          if (rcur[j] < 0) neg += rcur[j];
        }
      for (int i = 0; i < m; ++i) {
        if (w.ract[i]) {
          cnt[i] = 0;
          rows.push_back(i);
          usum += u[i];
          for (int q = P.row_ptr[i]; q < P.row_ptr[i + 1]; ++q)
            if (state[P.row_col[q]] == 0) lr_col.push_back(P.row_col[q]);
        }
        lr_ptr[i + 1] = static_cast<int>(lr_col.size());
      }
      capl = w.cap;
      for (int t = 0; t < P.ns; ++t) vterm += v[t] * capl[t];
      chosen = w.fixed;
      stack.clear();
    }
    bool fits(int j) const {
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q)
        if (P.cs_val[q] > capl[P.cs_row[q]] + 1e-9) return false;
      return true;
    }
    double bound() const { return cost + usum - vterm + neg; }
    void choose(int j) {
      state[j] = 1;
      if (rcur[j] < 0) neg -= rcur[j];
      cost += P.cost[j];
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) {
        capl[P.cs_row[q]] -= P.cs_val[q];
        vterm -= (*vv)[P.cs_row[q]] * P.cs_val[q];
      }
      for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
        const int i = P.col_row[q];
        if (cnt[i]++) continue;
        const double ui = (*uu)[i];
        usum -= ui;
        if (ui == 0.0) continue;
        for (int s = lr_ptr[i]; s < lr_ptr[i + 1]; ++s) {
          const int k = lr_col[s];
          const double old = rcur[k];
          rcur[k] = old + ui;
          if (state[k] == 0 && old < 0) neg += std::min(0.0, rcur[k]) - old;
        }
      }
    }
    void unchoose(int j) {
      for (int q = P.col_ptr[j + 1] - 1; q >= P.col_ptr[j]; --q) {
        const int i = P.col_row[q];
        if (--cnt[i]) continue;
        const double ui = (*uu)[i];
        if (ui == 0.0) continue;
        for (int s = lr_ptr[i]; s < lr_ptr[i + 1]; ++s) rcur[lr_col[s]] -= ui;
      }
      for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) capl[P.cs_row[q]] += P.cs_val[q];
      state[j] = 0;
    }
    void dfs() {
      if (aborted) return;
      if ((++nodes & 255ULL) == 0 && (nodes > limit || w.S.timeup())) {
        aborted = true;
        return;
      }
      const double lb = bound();
      const double thr = w.cur_thr();
      if (lb > thr) return;
      int bi = -1, bc = INT_MAX;
      for (int i : rows) {
        if (cnt[i]) continue;
        int c = 0;
        for (int q = lr_ptr[i]; q < lr_ptr[i + 1] && c < bc; ++q) {
          const int j = lr_col[q];
          if (state[j] == 0 && lb + std::max(rcur[j], 0.0) <= thr && fits(j)) ++c;
        }
        if (c < bc) {
          bc = c;
          bi = i;
          if (c == 0) return;
        }
      }
      if (bi < 0) {  // every row covered
        if (cost < w.cur_ub()) w.finish_solution(chosen);
        return;
      }
      const size_t base = stack.size();
      for (int q = lr_ptr[bi]; q < lr_ptr[bi + 1]; ++q) {
        const int j = lr_col[q];
        if (state[j] == 0 && lb + std::max(rcur[j], 0.0) <= thr && fits(j)) stack.push_back(j);
      }
      const size_t end = stack.size();
      std::sort(stack.begin() + static_cast<long>(base), stack.end(),
                [&](int a, int b) { return rcur[a] < rcur[b] || (rcur[a] == rcur[b] && a < b); });
      const double usum0 = usum, vterm0 = vterm, neg0 = neg, cost0 = cost;
      double neg_here = neg0;  // grows as siblings get excluded
      size_t done = base;
      for (size_t a = base; a < end; ++a) {
        const int j = stack[a];
        if (cost0 + usum0 - vterm0 + neg_here + std::max(rcur[j], 0.0) <= w.cur_thr()) {
          neg = neg_here;
          chosen.push_back(j);
          choose(j);
          dfs();
          unchoose(j);
          chosen.pop_back();
          usum = usum0;
          vterm = vterm0;
          cost = cost0;
          if (aborted) break;
        }
        state[j] = 2;  // x_j = 0 for the remaining siblings
        if (rcur[j] < 0) neg_here -= rcur[j];
        done = a + 1;
        if (cost0 + usum0 - vterm0 + neg_here > w.cur_thr()) break;
      }
      for (size_t a = base; a < done; ++a) state[stack[a]] = 0;
      neg = neg0;
      stack.resize(base);
    }
    bool run(unsigned long long node_limit) {  // true if the search finished
      const unsigned long long before = nodes;
      limit = nodes + node_limit;
      aborted = false;
      dfs();
      w.S.enum_nodes += static_cast<long long>(nodes - before);
      return !aborted;
    }
  };

  Enum en{*this};
  double phi = 0.3;  // share of the rows freed in a neighbourhood step
  std::vector<int> inc, pre, queue;
  std::vector<char> in_inc, freed, seen;

  // Explorer workers (odd ids) run the neighbourhood search on a solution of their
  // own instead of the shared incumbent, and restart it from a fresh Lagrangian
  // solution when it stalls. `local` is on while such a search runs: it then
  // accepts anything better than the worker's own solution.
  const bool explorer;
  bool local = false;
  std::vector<int> own, pass_best;
  double own_cost = INF, pass_cost = INF;
  int own_stall = 0, lns_fail = 0;
  unsigned long long lns_nodes = 20000;

  double thr_of(double ubv) const {
    if (ubv == INF) return INF;
    return P.integral ? ubv - 1.0 + 1e-6 : ubv - 1e-9 * std::max(1.0, std::fabs(ubv));
  }
  double cur_ub() const { return local ? own_cost : S.ub.load(std::memory_order_relaxed); }
  double cur_thr() const { return local ? thr_of(own_cost) : S.thr(); }

  void finish_solution(const std::vector<int>& cols) {
    load(cols, {});
    drop_redundant();
    if (S.opt.local_search) improve();
    keep_current();
    unload();
  }
  // the loaded solution -> shared incumbent, own solution, best of this pass
  void keep_current() {
    if (cur_cost < S.ub.load(std::memory_order_relaxed)) {
      std::vector<int> sol(cur);
      S.offer(sol, cur_cost);
    }
    if (explorer) {
      if (cur_cost < own_cost && (local || own.empty())) {
        own = cur;
        own_cost = cur_cost;
      }
      if (cur_cost < pass_cost) {
        pass_best = cur;
        pass_cost = cur_cost;
      }
    }
  }

  // Complete search from the root multipliers; proves optimality if it finishes.
  void enumerate(unsigned long long node_limit) {
    unfix_all();
    apply_rc_fix();
    en.build(u_root, v_root, rc_root, lb_root);
    if (en.run(node_limit)) S.set_optimal();
  }

  // One neighbourhood step: keep a part of the incumbent, free the columns around a
  // random row (or the ones with the largest gap contribution), re-optimise the
  // multipliers of the freed part and search it exhaustively.
  // Returns true if the incumbent improved.
  bool lns_step() {
    unfix_all();
    apply_rc_fix();
    local = explorer && !own.empty() && own_cost > S.ub.load();
    if (local) {
      inc = own;
    } else {
      std::lock_guard<std::mutex> g(S.mu);
      inc = S.best;
    }
    if (inc.empty()) {
      local = false;
      return false;
    }
    if (in_inc.empty()) {
      in_inc.assign(n, 0);
      freed.assign(n, 0);
      seen.assign(m, 0);
    }
    for (int j : inc) in_inc[j] = 1;
    const int target = std::max(2, static_cast<int>(phi * m));
    int got = 0;
    if (phi >= 1.0) {
      for (int j : inc) freed[j] = 1;
    } else if (rnd() < 0.75) {
      queue.clear();
      const int r0 = static_cast<int>(rng() % static_cast<unsigned long long>(m));
      queue.push_back(r0);
      seen[r0] = 1;
      got = 1;
      for (size_t h = 0; h < queue.size() && got < target; ++h) {
        const int i = queue[h];
        for (int q = P.row_ptr[i]; q < P.row_ptr[i + 1]; ++q) {
          const int j = P.row_col[q];
          if (!in_inc[j] || freed[j]) continue;
          freed[j] = 1;
          for (int s = P.col_ptr[j]; s < P.col_ptr[j + 1]; ++s) {
            const int i2 = P.col_row[s];
            if (!seen[i2]) {
              seen[i2] = 1;
              queue.push_back(i2);
              ++got;
            }
          }
        }
        const int len = P.row_ptr[i + 1] - P.row_ptr[i];
        for (int hop = 0; hop < 2 && len > 0; ++hop) {  // step to neighbouring rows through two random columns
          const int k = P.row_col[static_cast<size_t>(P.row_ptr[i]) + static_cast<size_t>(rng() % static_cast<unsigned long long>(len))];
          for (int s = P.col_ptr[k]; s < P.col_ptr[k + 1]; ++s) {
            const int i2 = P.col_row[s];
            if (!seen[i2]) {
              seen[i2] = 1;
              queue.push_back(i2);
            }
          }
        }
      }
      for (int i : queue) seen[i] = 0;
    } else {
      std::vector<int> c(m, 0);
      for (int j : inc)
        for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) ++c[P.col_row[q]];
      std::vector<std::pair<double, int>> order;
      for (int j : inc) {
        double d = std::max(rc_root[j] < INF ? rc_root[j] : 0.0, 0.0);
        for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) d += u_root[P.col_row[q]] * (c[P.col_row[q]] - 1) / c[P.col_row[q]];
        order.emplace_back(-(d + 1e-9) * (0.5 + rnd()), j);
      }
      std::sort(order.begin(), order.end());
      for (const auto& e : order) {
        if (got >= target) break;
        freed[e.second] = 1;
        got += P.col_ptr[e.second + 1] - P.col_ptr[e.second];
      }
    }
    pre.clear();
    for (int j : inc) {
      if (!freed[j] && !dead[j]) pre.push_back(j);
      in_inc[j] = 0;
      freed[j] = 0;
    }
    u = u_root;
    v = v_root;
    for (int j : pre)
      if (fits(j)) fix_col(j);
    const bool whole = fixed.empty();
    const double ub0 = cur_ub(), gub0 = S.ub.load();
    bool complete = true;
    if (mact > 0) {
      price();
      const double Lres = subgradient(false, 100, 400, false);
      if (S.timeup()) {
        local = false;
        return false;
      }
      if (fixed_cost + Lres <= cur_thr()) {
        en.build(u, v, rc, Lres);
        complete = en.run(lns_nodes);
      }
    }
    const bool improved = cur_ub() < ub0;
    const bool was_local = local;
    local = false;
    if (complete && whole && !was_local && !S.timeup()) {
      S.set_optimal();
      return S.ub.load() < gub0;
    }
    if (improved) {
      own_stall = 0;
      lns_fail = 0;
    } else {
      ++own_stall;
      if (complete) phi = std::min(1.0, phi * 1.15);
      else phi = std::max(2.0 / std::max(2, m), phi * 0.85);
      if (++lns_fail >= 40) {  // stuck: allow larger neighbourhoods to be searched exactly
        lns_fail = 0;
        lns_nodes = std::min<unsigned long long>(lns_nodes * 2, 2000000ULL);
      }
    }
    return S.ub.load() < gub0;
  }
};

}  // namespace

// Cover rows first, then the side rows.
LpProblem to_lp(const Problem& P) {
  LpProblem L;
  L.n = P.n;
  L.m = P.m + P.ns;
  L.cost = P.cost;
  L.lb.assign(P.n, 0.0);
  L.ub.assign(P.n, 1.0);
  L.rlo.assign(L.m, -INF);
  L.rup.assign(L.m, INF);
  for (int i = 0; i < P.m; ++i) L.rlo[i] = 1.0;
  for (int t = 0; t < P.ns; ++t) L.rup[P.m + t] = P.side_rhs[t];
  L.cptr.assign(1, 0);
  for (int j = 0; j < P.n; ++j) {
    for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) {
      L.cidx.push_back(P.col_row[q]);
      L.cval.push_back(1.0);
    }
    for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) {
      L.cidx.push_back(P.m + P.cs_row[q]);
      L.cval.push_back(P.cs_val[q]);
    }
    L.cptr.push_back(static_cast<int>(L.cidx.size()));
  }
  return L;
}

// One thread shared by a heuristic worker and the LP-based branch and bound: root LP first
// (published to all workers), then the tree search hands the thread to the worker at node
// boundaries so that each side gets its share of the time.
//
// With a thread of its own (with_worker false) the tree is watched instead: when the root LP is too
// expensive or the lower bound rises too slowly to reach the incumbent in the time that is left, the
// tree is given up and the thread becomes the heuristic worker `reserve` for the rest of the run.
void run_hybrid(Shared& S, int id, bool with_worker, int reserve = -1) {
  const Problem& P = S.P;
  Worker wk(S, with_worker || reserve < 0 ? id : reserve);
  bool alive = false;
  if (with_worker) {
    wk.begin();
    alive = wk.pass();  // a first solution before the root LP
  }
  const LpProblem L = to_lp(P);
  // tree_share: part of this thread given to the tree. It grows while the worker's passes bring
  // no better solution and shrinks again when they do.
  double tree_share = 0.5, share_min = 0.5, share_max = 0.9;
  double tree_time = 0.0, work_time = 0.0, last = S.elapsed();
  BnbHooks H;
  H.thr = [&S]() { return S.thr(); };
  H.offer = [&S](const std::vector<int>& sol, double cost) { S.offer(sol, cost); };
  H.offer_lb = [&S](double l) { S.offer_lb(l); };
  H.stop = [&S]() { return S.timeup(); };
  H.root = [&](const std::vector<double>& y, const std::vector<double>& d, double z) {
    S.lp_u.resize(P.m);
    S.lp_v.resize(P.ns);
    for (int i = 0; i < P.m; ++i) S.lp_u[i] = std::max(y[i], 0.0);
    for (int t = 0; t < P.ns; ++t) S.lp_v[t] = std::max(-y[P.m + t], 0.0);
    S.lp_rc = d;
    S.lp_bound = z;
    S.lp_ready.store(true, std::memory_order_release);
    // a wide root gap makes a finished tree unlikely: leave most of the thread to the heuristic
    const double ub = S.ub.load();
    if (ub < INF && (ub - z) / std::max(std::fabs(ub), 1e-12) > 0.05) tree_share = share_min = 0.15, share_max = 0.3;
    last = S.elapsed();
  };
  H.yield = [&]() {
    double now = S.elapsed();
    tree_time += now - last;
    // the worker runs whole passes, so it is charged for what it used before it may run again
    while (alive && !S.timeup() && work_time < tree_time * (1.0 - tree_share) / tree_share) {
      const double ub0 = S.ub.load();
      alive = wk.pass();
      const double t = S.elapsed();
      work_time += t - now;
      now = t;
      if (S.ub.load() < ub0) tree_share = std::max(share_min, tree_share - 0.1);
      else tree_share = std::min(share_max, tree_share + 0.05);
    }
    last = S.elapsed();
  };
  BnbOptions o;
  o.seed = static_cast<unsigned>(S.opt.seed);
  o.integral = P.integral;
  o.time_limit = std::max(0.0, S.opt.time_limit - S.elapsed());
  o.ryan_foster = false;
  o.cuts = S.opt.cuts < 0 ? 1 : S.opt.cuts;
  o.rins = S.opt.local_search;
  H.incumbent = [&S](std::vector<int>& cols) {
    std::lock_guard<std::mutex> g(S.mu);
    cols = S.best;
  };
  // Watching the tree (only when the tree has a thread of its own and a worker waits to take it over).
  //   0. Before anything: with fractional costs (no rounding of bounds, so a proof needs a real tree) and
  //      a root LP estimated at more than 2% of the time limit (1.5e-8 s x rows x columns, fitted on the
  //      benchmark set), neither the LP nor the tree is started.
  //   1. Root LP: given up when it takes longer than 4% of the time limit (at least 0.3 s).
  //   2. Size: a node costs about (root LP time) x (share of columns left after reduced cost fixing).
  //      If the time left pays for fewer than 300 such nodes (1000 with fractional costs), the tree is
  //      not started.
  //   3. Progress: f = share of the root gap closed by the tree's bound (cut rounds included), read at
  //      times c, 2c, 4c, ... With the gain d over the last interval, even at that rate (the bound of a
  //      best-first search slows down, it does not speed up) the rest takes (1 - f) / d intervals. If
  //      that is more than tree_giveup times the time left at two readings in a row, the tree is given up.
  const bool watched = !with_worker && reserve >= 0 && S.opt.tree_giveup > 0.0;
  double t_tree = -1.0, t_check = 0.0, t_last = 0.0, f_prev = 0.0, t_log = 0.0;
  long n_last = 0;
  int strikes = 0;
  bool skip_tree = false;
  if (watched) o.root_time_limit = std::max(0.3, 0.04 * S.opt.time_limit);
  const double t_lp0 = S.elapsed();
  if (watched || DEBUG_TREE) {
    const auto root0 = H.root;
    H.root = [&, root0](const std::vector<double>& y, const std::vector<double>& d, double z) {
      root0(y, d, z);
      const double now = S.elapsed(), t_lp = now - t_lp0, thr = S.thr();
      if (thr == INF) return;
      long left_cols = 0;
      for (int j = 0; j < P.n; ++j) left_cols += z + std::max(d[j], 0.0) <= thr;
      const double node = std::max(t_lp * static_cast<double>(left_cols) / std::max(1, P.n), 1e-6);
      const double nodes = (S.opt.time_limit - now) / node;
      skip_tree = watched && nodes < (P.integral ? 300.0 : 1000.0);
      if (DEBUG_TREE)
        std::fprintf(stderr, "tree root t_lp=%.3f cols %ld/%d est. nodes in the time left %.0f -> %s\n", t_lp, left_cols, P.n,
                     nodes, skip_tree ? "no tree" : "tree");
    };
    H.keep = [&](double bound, long nodes) {
      if (skip_tree) return false;
      const double now = S.elapsed();
      if (t_tree < 0.0) {
        t_tree = now;
        t_check = std::max(0.25, 0.01 * S.opt.time_limit);
      }
      const double t = now - t_tree, thr = S.thr(), z = S.lp_bound;
      if (thr == INF) return true;
      const double f = thr - z > 1e-12 ? std::min(1.0, std::max(0.0, (bound - z) / (thr - z))) : 1.0;
      if (DEBUG_TREE && t >= t_log) {
        std::fprintf(stderr, "tree t=%.3f nodes=%ld z=%.6g bound=%.6g thr=%.6g f=%.4f\n", t, nodes, z, bound, thr, f);
        t_log = std::max(0.01, t * 1.25);
      }
      if (!watched || t < t_check) return true;
      const double left = S.opt.time_limit - now;
      if (nodes - n_last < 20) {  // too few nodes since the last reading to judge a rate
        t_check = t * 1.25;
        // ... unless the nodes are simply too slow: not even 20 of them in a quarter of the time limit
        return !(nodes < 20 && t > std::max(1.0, 0.25 * S.opt.time_limit));
      }
      const double d = std::max(f - f_prev, 0.0);
      const double rest = d > 1e-12 ? (1.0 - f) / d * (t - t_last) : INF;  // time for the rest at the last interval's rate
      if (rest > 10.0 * S.opt.tree_giveup * left) strikes += 2;
      else if (rest > S.opt.tree_giveup * left) ++strikes;
      else strikes = 0;
      if (DEBUG_TREE)
        std::fprintf(stderr, "tree check t=%.3f nodes=%ld f=%.4f gain=%.4f rest=%.1f s left=%.1f s strikes=%d\n", t, nodes, f, d, rest, left,
                     strikes);
      f_prev = f;
      t_last = t;
      n_last = nodes;
      t_check = t * 2.0;
      return strikes < 2;
    };
  }
  const double lp_estimate = 1.5e-8 * static_cast<double>(P.m + P.ns) * static_cast<double>(P.n);
  const bool no_lp = watched && !P.integral && lp_estimate > 0.02 * S.opt.time_limit;
  if (DEBUG_TREE && no_lp) std::fprintf(stderr, "tree: root LP estimated at %.2f s, not started\n", lp_estimate);
  if (!S.timeup() && !no_lp) {
    S.bnb = branch_and_bound(L, H, o);
    S.bnb_ran = true;
    if (S.bnb.status == BnbResult::EXHAUSTED) {
      if (S.ub.load() < INF) {
        S.set_optimal();
      } else {
        S.infeasible = true;
        S.stop = true;
      }
    } else if (S.bnb.status == BnbResult::LP_INFEASIBLE) {
      S.infeasible = true;
      S.stop = true;
    }
  }
  if (!with_worker && reserve >= 0 && !S.timeup()) {  // the tree was given up: this thread turns into a worker
    if (DEBUG_TREE) std::fprintf(stderr, "tree thread becomes worker %d at %.3f s\n", reserve, S.elapsed());
    S.tree_gave_up = S.elapsed();
    wk.begin();
    alive = true;
  }
  while (alive && !S.timeup()) alive = wk.pass();  // the tree gave up early: the worker takes the rest
}

SolveResult solve_sc(const Problem& orig, const SolveOptions& opt, const std::vector<int>* warm) {
  const auto t0 = Clock::now();
  SolveResult res;
  PresolveResult pre = presolve(orig, opt.presolve);
  const Problem& P = pre.red;
  res.red_cols = P.n;
  res.red_rows = P.m;
  res.fixed_cols = static_cast<int>(pre.fixed_one.size());
  res.presolve_time = std::chrono::duration<double>(Clock::now() - t0).count();
  res.bound = NaN;

  auto orig_cost = [&](const std::vector<int>& cols) {
    double c = orig.offset;
    for (int j : cols) c += orig.cost[j];
    return c;
  };
  if (pre.infeasible) {
    res.status = "infeasible";
    res.runtime = std::chrono::duration<double>(Clock::now() - t0).count();
    return res;
  }

  Shared S(P, opt, t0);

  // warm start -> reduced space
  if (warm && !warm->empty() && P.m > 0) {
    std::vector<char> in(P.n, 0);
    for (int j0 : *warm) {
      int j = j0, guard = 0;
      while (j >= 0 && pre.to_red[j] < 0 && pre.rep[j] != j && guard++ < orig.n) j = pre.rep[j];
      if (j >= 0 && pre.to_red[j] >= 0) in[pre.to_red[j]] = 1;
    }
    std::vector<int> c(P.m, 0);
    for (int j = 0; j < P.n; ++j)
      if (in[j])
        for (int q = P.col_ptr[j]; q < P.col_ptr[j + 1]; ++q) ++c[P.col_row[q]];
    for (int i = 0; i < P.m; ++i) {
      if (c[i]) continue;
      int bj = -1;
      for (int q = P.row_ptr[i]; q < P.row_ptr[i + 1]; ++q)
        if (bj < 0 || P.cost[P.row_col[q]] < P.cost[bj]) bj = P.row_col[q];
      if (bj < 0) continue;
      in[bj] = 1;
      for (int q = P.col_ptr[bj]; q < P.col_ptr[bj + 1]; ++q) ++c[P.col_row[q]];
    }
    std::vector<double> use(P.ns, 0.0);
    std::vector<int> sol;
    double cost = 0.0;
    for (int j = 0; j < P.n; ++j)
      if (in[j]) {
        sol.push_back(j);
        cost += P.cost[j];
        for (int q = P.cs_ptr[j]; q < P.cs_ptr[j + 1]; ++q) use[P.cs_row[q]] += P.cs_val[q];
      }
    bool ok = true;
    for (int t = 0; t < P.ns; ++t)
      if (use[t] > P.side_rhs[t] + 1e-9) ok = false;
    for (int i = 0; i < P.m; ++i)
      if (!c[i]) ok = false;
    if (ok) {
      S.offer(sol, cost);
      res.warm_used = true;
    }
  }

  if (P.m == 0) {
    S.offer({}, 0.0);
    S.set_optimal();
  } else {
    const int nt = std::max(1, opt.threads);
    // With three or more threads one of them does nothing but the root LP and the tree search and the
    // others are heuristic workers. With fewer, the last thread shares its time between a worker and the tree.
    const bool own_thread = opt.bnb && nt >= 3;
    // Which worker gives up its thread was measured on the 2000-column DMS instances: keeping both
    // "explorer" workers (odd ids) loses the least solution quality (docs/p3_report.md).
    const int dropped = !own_thread ? -1 : (nt >= 4 ? 2 : nt - 1);
    std::vector<int> ids;
    for (int w = 0; w < nt; ++w)
      if (w != dropped) ids.push_back(w);
    const int hybrid = opt.bnb && !own_thread ? nt - 1 : -1;
    std::vector<std::thread> pool;
    for (size_t k = 1; k < ids.size(); ++k) {
      const int w = ids[k];
      pool.emplace_back([&S, w, hybrid]() {
        if (w == hybrid) {
          run_hybrid(S, w, true);
        } else {
          Worker wk(S, w);
          wk.run();
        }
      });
    }
    if (own_thread) pool.emplace_back([&S, nt, dropped]() { run_hybrid(S, nt - 1, false, dropped); });
    if (hybrid == 0) {
      run_hybrid(S, 0, true);
    } else {
      Worker wk(S, 0);
      wk.run();
    }
    for (auto& t : pool) t.join();
  }

  const double ub = S.ub.load(), lb = S.lb.load();
  std::vector<int> cols;
  if (ub < INF) {
    cols = pre.fixed_one;
    for (int j : S.best) cols.push_back(pre.col_orig[static_cast<size_t>(j)]);
    std::sort(cols.begin(), cols.end());
  }
  bool have = ub < INF;
  double obj = have ? orig_cost(cols) : INF;
  if (warm && !warm->empty()) {
    const double wc = orig_cost(*warm);
    if (!have || wc < obj - 1e-12 * std::max(1.0, std::fabs(wc))) {
      cols = *warm;
      std::sort(cols.begin(), cols.end());
      obj = wc;
      have = true;
    }
  }
  res.trace = S.trace;
  res.runtime = std::chrono::duration<double>(Clock::now() - t0).count();
  res.sub_iters = S.sub_iters;
  res.greedy_calls = S.greedy_calls;
  res.ls_moves = S.ls_moves;
  res.enum_nodes = S.enum_nodes;
  res.passes = S.passes;
  res.lp_root = NaN;
  if (S.bnb_ran) {
    static const char* names[] = {"exhausted", "stopped", "failed", "lp_infeasible"};
    res.bnb_status = names[S.bnb.status];
    if (S.lp_ready.load()) res.lp_root = S.lp_bound + P.offset;
    res.lp_root_time = S.bnb.root_time;
    res.bnb_nodes = S.bnb.nodes;
    res.bnb_lp_iters = S.bnb.lp_iters;
    res.bnb_strong = S.bnb.strong_lps;
    res.bnb_cols = S.bnb.lp_cols;
    res.tree_gave_up = S.tree_gave_up;
    res.lp_root_cut = S.bnb.root_cut + P.offset;
    res.cuts = S.bnb.cuts;
    res.cut_rounds = S.bnb.cut_rounds;
    res.cut_time = S.bnb.cut_time;
    res.node_fixed = S.bnb.node_fixed;
    res.rins_calls = S.bnb.rins_calls;
    res.rins_found = S.bnb.rins_found;
  }
  if (S.infeasible.load() && !have) {
    res.status = "infeasible";
    return res;
  }
  if (!have) {
    res.status = "no_solution";
    res.bound = S.shown_bound(lb, INF);
    return res;
  }
  res.cols = cols;
  res.obj = obj;
  if (S.proven.load()) {
    res.status = "optimal";
    res.bound = obj;
  } else {
    res.status = "time_limit";
    res.bound = lb == -INF ? NaN : std::min(S.shown_bound(lb, INF), obj);
  }
  return res;
}

}  // namespace scsp
