#include "scsp/bin_solver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <thread>

#include "scsp/bnb.hpp"

namespace scsp {

namespace {
const double INF = std::numeric_limits<double>::infinity();
const double NaN = std::numeric_limits<double>::quiet_NaN();
using Clock = std::chrono::steady_clock;

SolveResult solve_core(const LpProblem& L, const SolveOptions& opt, const std::vector<int>* warm, Clock::time_point t0) {
  auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  SolveResult res;
  res.red_cols = L.n;
  res.red_rows = L.m;
  res.bound = NaN;
  res.lp_root = NaN;

  bool integral = std::floor(L.offset) == L.offset;
  for (double c : L.cost)
    if (std::floor(c) != c) integral = false;
  bool nonneg = true;
  for (double v : L.cval)
    if (v < 0.0) nonneg = false;

  std::mutex mu;
  std::vector<int> best;
  double ub = INF, lb = -INF, last_lb_trace = -1.0;  // costs without the offset
  std::atomic<double> cur_ub{INF}, cur_lb{-INF};
  auto shown = [&](double l) {
    if (l == -INF) return NaN;
    double b = l + L.offset;
    if (integral) b = std::ceil(b - 1e-6);
    return ub == INF ? b : std::min(b, ub + L.offset);
  };
  auto thr = [&]() {
    std::lock_guard<std::mutex> g(mu);
    if (ub == INF) return INF;
    if (opt.mip_gap > 0.0) return ub - std::max(opt.mip_gap * std::fabs(ub + L.offset), 1e-9);
    return integral ? ub - 1.0 + 1e-6 : ub - 1e-9 * std::max(1.0, std::fabs(ub));
  };
  auto feasible = [&](const std::vector<int>& cols) {
    std::vector<double> act(L.m, 0.0);
    for (int j : cols) {
      if (j < 0 || j >= L.n) return false;
      for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) act[L.cidx[q]] += L.cval[q];
    }
    for (int i = 0; i < L.m; ++i)
      if (act[i] < L.rlo[i] - 1e-6 || act[i] > L.rup[i] + 1e-6) return false;
    return true;
  };
  auto offer = [&](const std::vector<int>& cols, double cost) {
    std::lock_guard<std::mutex> g(mu);
    if (cost >= ub - 1e-12 * std::max(1.0, std::fabs(cost))) return;
    best = cols;
    ub = cost;
    cur_ub.store(cost, std::memory_order_relaxed);
    res.trace.push_back({elapsed(), cost + L.offset, shown(lb), 0});
  };

  if (warm && !warm->empty() && feasible(*warm)) {
    double c = 0.0;
    for (int j : *warm) c += L.cost[j];
    std::vector<int> w = *warm;
    std::sort(w.begin(), w.end());
    w.erase(std::unique(w.begin(), w.end()), w.end());
    offer(w, c);
    res.warm_used = true;
  }

  // ---- neighbourhood search around the incumbent (sub-problems solved by the same branch and bound)
  // Four complementary neighbourhoods (P4/P6):
  //   1. RINS (Danna, Rothberg & Le Pape 2005): columns on which a node LP solution and the incumbent agree are fixed;
  //   2. Row-cluster / bridge-connected LNS (Shaw 1998; Ropke & Pisinger 2006): frees a connected cluster of incumbent
  //      columns sharing low-RC bridging columns so far more alternative columns fit the freed region;
  //   3. Asymmetric Local Branching (Fischetti & Lodi 2003): sum_{j in cur} x_j >= |cur| - delta over cur + low-RC pool;
  //   4. Biased-RC fixing LNS: keeps a subset of the incumbent biased toward low-RC columns.
  std::vector<int> by_rc, rptr, rcol;     // sorted by root reduced cost (written once in H.root, then root_ready)
  std::vector<double> droot;
  std::atomic<bool> root_ready{false}, done{false};
  std::vector<int> pt_cols;               // latest node LP solution of the tree (under mu)
  std::vector<double> pt_x;
  long pt_version = 0;
  std::atomic<long> lns_calls{0}, lns_improved{0};
  auto timeup = [&]() { return done.load(std::memory_order_relaxed) || elapsed() >= opt.time_limit; };

  struct LnsState {
    std::mt19937_64 rng;
    double frac = 0.5;     // share of the incumbent that stays fixed (fixing / cluster neighbourhood)
    long target = 0;       // free columns besides the incumbent's
    int lb_delta = 10;     // asymmetric local branching radius delta
    int mode_ctr = 0;
    long seen_point = 0;
    std::vector<char> inc;
  };
  auto lns_step = [&](LnsState& st, double budget, long node_limit) {
    std::vector<int> cur, pc;
    std::vector<double> px;
    double cost;
    long ver;
    {
      std::lock_guard<std::mutex> g(mu);
      cur = best;
      cost = ub;
      ver = pt_version;
      if (ver != st.seen_point) pc = pt_cols, px = pt_x;
    }
    if (cost == INF || cur.empty()) return false;
    st.inc.assign(L.n, 0);
    for (int j : cur) st.inc[j] = 1;
    std::vector<int> ones, freec;
    const bool rins = ver != st.seen_point && (st.rng() & 1);
    int mkind = -1;
    if (rins) {
      st.seen_point = ver;
      std::vector<char> in_point(L.n, 0);
      for (size_t k = 0; k < pc.size(); ++k) {
        const int j = pc[k];
        in_point[j] = 1;
        if (st.inc[j] && px[k] > 1.0 - 1e-6) ones.push_back(j);
        else if (st.inc[j] || px[k] > 1e-6) freec.push_back(j);
      }
      for (int j : cur)
        if (!in_point[j]) freec.push_back(j);
    } else {
      mkind = (st.mode_ctr++) % 5;
      if (mkind == 1 && cur.size() >= 8) {
        // Compact Asymmetric Local Branching (Fischetti & Lodi 2003): sum_{j in cur} x_j >= |cur| - delta.
        freec = cur;
        const long lb_pool = std::min<long>(L.n, std::max<long>(2L * L.m, 600L));
        long added = 0;
        for (int j : by_rc) {
          if (added >= lb_pool) break;
          if (st.inc[j]) continue;
          freec.push_back(j);
          ++added;
        }
      } else if (mkind == 0 || mkind == 2 || mkind == 3) {
        // Row-cluster / bridge-connected LNS (Shaw 1998; Ropke & Pisinger 2006).
        const int nc = static_cast<int>(cur.size());
        const int free_k = std::max(4, std::min(nc, nc - static_cast<int>(st.frac * static_cast<double>(nc))));
        std::vector<int> row_owner(L.m, -1);
        for (int u = 0; u < nc; ++u)
          for (int q = L.cptr[cur[u]]; q < L.cptr[cur[u] + 1]; ++q) row_owner[L.cidx[q]] = u;
        std::vector<char> freed(nc, 0);
        std::vector<int> score(nc, 0), frontier;
        // Tournament seed biased toward higher root reduced cost.
        int seed_u = static_cast<int>(st.rng() % static_cast<uint64_t>(nc));
        if ((st.rng() % 10) < 7) {
          for (int t = 0; t < 3; ++t) {
            const int cand_u = static_cast<int>(st.rng() % static_cast<uint64_t>(nc));
            if (droot[cur[cand_u]] > droot[cur[seed_u]]) seed_u = cand_u;
          }
        }
        auto free_u = [&](int u) {
          freed[u] = 1;
          const int j0 = cur[u];
          for (int q = L.cptr[j0]; q < L.cptr[j0 + 1]; ++q) {
            const int i = L.cidx[q];
            const int rlim = std::min(rptr[i + 1], rptr[i] + 24);
            for (int a = rptr[i]; a < rlim; ++a) {
              const int j = rcol[a];
              if (st.inc[j]) continue;
              for (int b = L.cptr[j]; b < L.cptr[j + 1]; ++b) {
                const int v = row_owner[L.cidx[b]];
                if (v >= 0 && !freed[v]) {
                  if (score[v]++ == 0) frontier.push_back(v);
                }
              }
            }
          }
        };
        free_u(seed_u);
        int nfreed = 1;
        while (nfreed < free_k) {
          int pick = -1;
          // Compact frontier and pick among top-3 highest-relatedness neighbours.
          size_t w = 0;
          for (size_t idx = 0; idx < frontier.size(); ++idx)
            if (!freed[frontier[idx]]) frontier[w++] = frontier[idx];
          frontier.resize(w);
          if (!frontier.empty()) {
            const size_t topk = std::min<size_t>(3, frontier.size());
            std::partial_sort(frontier.begin(), frontier.begin() + topk, frontier.end(), [&](int a, int b) {
              if (score[a] != score[b]) return score[a] > score[b];
              return droot[cur[a]] > droot[cur[b]];
            });
            pick = frontier[static_cast<size_t>(st.rng() % topk)];
          } else {
            for (int tries = 0; tries < nc; ++tries) {
              const int u = static_cast<int>(st.rng() % static_cast<uint64_t>(nc));
              if (!freed[u]) {
                pick = u;
                break;
              }
            }
            if (pick < 0) {
              for (int u = 0; u < nc; ++u)
                if (!freed[u]) {
                  pick = u;
                  break;
                }
            }
          }
          if (pick < 0) break;
          free_u(pick);
          ++nfreed;
        }
        for (int u = 0; u < nc; ++u) {
          if (freed[u]) freec.push_back(cur[u]);
          else ones.push_back(cur[u]);
        }
        std::vector<double> act(L.m, 0.0);
        for (int j : ones)
          for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) act[L.cidx[q]] += L.cval[q];
        long added = 0;
        for (int j : by_rc) {
          if (added >= st.target) break;
          if (st.inc[j]) continue;
          bool fits = true;
          if (nonneg)
            for (int q = L.cptr[j]; q < L.cptr[j + 1] && fits; ++q)
              if (act[L.cidx[q]] + L.cval[q] > L.rup[L.cidx[q]] + 1e-9) fits = false;
          if (!fits) continue;
          freec.push_back(j);
          ++added;
        }
      } else {
        // Randomized / RC-biased fixing LNS.
        std::shuffle(cur.begin(), cur.end(), st.rng);
        const size_t keep = static_cast<size_t>(st.frac * static_cast<double>(cur.size()));
        ones.assign(cur.begin(), cur.begin() + keep);
        freec.assign(cur.begin() + keep, cur.end());
        std::vector<double> act(L.m, 0.0);
        for (int j : ones)
          for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) act[L.cidx[q]] += L.cval[q];
        long added = 0;
        for (int j : by_rc) {
          if (added >= st.target) break;
          if (st.inc[j]) continue;
          bool fits = true;
          if (nonneg)
            for (int q = L.cptr[j]; q < L.cptr[j + 1] && fits; ++q)
              if (act[L.cidx[q]] + L.cval[q] > L.rup[L.cidx[q]] + 1e-9) fits = false;
          if (!fits) continue;
          freec.push_back(j);
          ++added;
        }
      }
    }
    std::sort(freec.begin(), freec.end());
    freec.erase(std::unique(freec.begin(), freec.end()), freec.end());
    SubProblem sub = make_sub(L, freec, ones, nonneg);
    if (!sub.ok || sub.P.n < 2) {
      if (!rins && mkind != 1) st.frac = std::max(0.1, st.frac - 0.05);
      return true;
    }
    const bool is_lb = (!rins && mkind == 1 && cur.size() >= 8);
    if (is_lb) {
      // Append the asymmetric Local Branching constraint: sum_{k: col_of[k] in cur} x_k >= |cur| - lb_delta.
      const int lb_row = sub.P.m++;
      const double rhs = std::max(0.0, static_cast<double>(cur.size()) - static_cast<double>(st.lb_delta));
      sub.P.rlo.push_back(rhs);
      sub.P.rup.push_back(INF);
      std::vector<int> nptr(1, 0), nidx;
      std::vector<double> nval;
      nidx.reserve(sub.P.cidx.size() + cur.size());
      nval.reserve(sub.P.cval.size() + cur.size());
      for (int k = 0; k < sub.P.n; ++k) {
        nidx.insert(nidx.end(), sub.P.cidx.begin() + sub.P.cptr[k], sub.P.cidx.begin() + sub.P.cptr[k + 1]);
        nval.insert(nval.end(), sub.P.cval.begin() + sub.P.cptr[k], sub.P.cval.begin() + sub.P.cptr[k + 1]);
        if (st.inc[sub.col_of[k]]) {
          nidx.push_back(lb_row);
          nval.push_back(1.0);
        }
        nptr.push_back(static_cast<int>(nidx.size()));
      }
      sub.P.cptr.swap(nptr);
      sub.P.cidx.swap(nidx);
      sub.P.cval.swap(nval);
    }
    const double t_end = elapsed() + budget;
    BnbHooks h;
    h.thr = [&]() { return thr() - sub.fixed_cost; };
    h.offer = [&](const std::vector<int>& sc, double c) {
      std::vector<int> full = ones;
      for (int k : sc) full.push_back(sub.col_of[k]);
      std::sort(full.begin(), full.end());
      offer(full, c + sub.fixed_cost);
    };
    h.offer_lb = [](double) {};
    h.stop = [&]() { return timeup() || elapsed() >= t_end; };
    BnbOptions so;
    so.seed = static_cast<unsigned>(st.rng());
    so.integral = integral;
    so.time_limit = std::max(0.0, std::min(budget, opt.time_limit - elapsed()));
    so.node_limit = is_lb ? std::min<long>(node_limit, 150) : node_limit;
    so.strong_cands = 4;
    so.strong_iters = 40;
    const BnbResult r = branch_and_bound(sub.P, h, so);
    ++lns_calls;
    bool improved;
    {
      std::lock_guard<std::mutex> g(mu);
      improved = ub < cost;
    }
    if (improved) ++lns_improved;
    if (is_lb) {
      if (r.status == BnbResult::EXHAUSTED && !improved) st.lb_delta = std::min(16, st.lb_delta + 2);
      else if (improved) st.lb_delta = std::max(4, st.lb_delta - 1);
      else st.lb_delta = std::max(4, st.lb_delta - 2);
    } else if (!rins) {
      // a neighbourhood searched to the end without a better solution was too small, one that hit its limit too large
      if (r.status == BnbResult::EXHAUSTED && !improved) {
        if (st.frac > 0.15) st.frac -= 0.05;
        else st.target = std::min<long>(L.n, st.target + st.target / 4);
      } else if (r.status != BnbResult::EXHAUSTED) {
        st.frac = std::min(0.92, st.frac + 0.05);
      }
    }
    return true;
  };
  auto new_state = [&](int id) {
    LnsState st;
    st.rng.seed(opt.seed * 7919ULL + 104729ULL * static_cast<unsigned>(id) + 17ULL);
    static const double init_fracs[4] = {0.55, 0.65, 0.75, 0.85};
    st.frac = init_fracs[id & 3];
    st.target = std::max<long>(300, 3L * L.m);
    st.lb_delta = 6 + 2 * (id % 3);
    st.mode_ctr = id;
    return st;
  };

  BnbTeam team;
  BnbHooks H;
  H.thr = thr;
  H.offer = offer;
  H.offer_lb = [&](double l) {
    std::lock_guard<std::mutex> g(mu);
    if (l <= lb) return;
    lb = l;
    cur_lb.store(l, std::memory_order_relaxed);
    const double t = elapsed();
    if (t - last_lb_trace >= 0.1) {
      last_lb_trace = t;
      res.trace.push_back({t, ub == INF ? NaN : ub + L.offset, shown(l), 1});
    }
  };
  H.stop = [&]() { return elapsed() >= opt.time_limit; };
  const int nt = std::max(1, opt.threads);
  if (nt > 1) H.team = &team;
  const bool lns = opt.local_search && L.n > 0 && L.m > 0;
  if (lns) {
    H.root = [&](const std::vector<double>&, const std::vector<double>& d, double) {
      droot = d;
      by_rc.resize(L.n);
      for (int j = 0; j < L.n; ++j) by_rc[j] = j;
      std::sort(by_rc.begin(), by_rc.end(), [&](int a, int b) { return d[a] < d[b] || (d[a] == d[b] && a < b); });
      // Row-wise column lists sorted by root reduced cost for fast bridge-connected cluster LNS.
      rptr.assign(L.m + 1, 0);
      for (int i : L.cidx) ++rptr[i + 1];
      for (int i = 0; i < L.m; ++i) rptr[i + 1] += rptr[i];
      rcol.resize(L.cidx.size());
      std::vector<int> pos(rptr.begin(), rptr.end() - 1);
      for (int j = 0; j < L.n; ++j)
        for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) rcol[pos[L.cidx[q]]++] = j;
      for (int i = 0; i < L.m; ++i)
        std::sort(rcol.begin() + rptr[i], rcol.begin() + rptr[i + 1],
                  [&](int a, int b) { return d[a] < d[b] || (d[a] == d[b] && a < b); });
      root_ready.store(true, std::memory_order_release);
    };
    H.lp_point = [&](const std::vector<int>& cols, const std::vector<double>& x) {
      std::lock_guard<std::mutex> g(mu);
      pt_cols = cols;
      pt_x = x;
      ++pt_version;
    };
  }
  // With one thread the tree lends it to the neighbourhood search for a sixth of the time.
  LnsState own = new_state(0);
  double lns_time = 0.0;
  if (lns && nt == 1) {
    H.yield = [&]() {
      const double t = elapsed();
      if (!root_ready.load(std::memory_order_acquire) || lns_time > 0.2 * (t - lns_time)) return;
      lns_step(own, std::max(0.05, 0.02 * opt.time_limit), 200);
      lns_time += elapsed() - t;
    };
  }
  // Gap-driven split between parallel B&B tree workers and LNS/RINS workers (P5):
  // - Before an incumbent exists, all threads help the tree find one.
  // - With a wide gap (>= 5%), keep 1 B&B + 3 LNS (strong primal improvement on large synthetic SPP).
  // - As the relative gap narrows (< 5%, < 3.5%, < 1.5%), move workers 1, 2, 3 into parallel B&B
  //   so provable instances scale up to 3-4 tree threads without starving LNS on hard wide-gap ones.
  auto want_tree = [&](int w) -> bool {
    if (!team.active()) return false;
    if (!lns) return true;
    const double u = cur_ub.load(std::memory_order_relaxed);
    if (u == INF) return true;
    const double l = cur_lb.load(std::memory_order_relaxed);
    const double denom = std::fabs(u + L.offset);
    const double gap = (l > -INF && denom > 1e-12) ? std::max(0.0, (u - l) / denom) : 1.0;
    const bool close_int = integral && l > -INF && (u - std::ceil(l - 1e-6) <= 1.0 + 1e-9);
    if (w == 1) return gap < 0.05 || close_int;
    if (w == 2) return gap < 0.035 || close_int;
    return gap < 0.015 || close_int;
  };
  std::vector<std::thread> pool;
  for (int w = 1; w < nt; ++w)
    pool.emplace_back([&, w]() {
      LnsState st = new_state(w);
      while (!timeup()) {
        if (want_tree(w)) {
          if (team.help([&]() { return !timeup() && want_tree(w); })) continue;
        }
        if (!lns || !root_ready.load(std::memory_order_acquire) || !lns_step(st, std::max(0.2, 0.03 * opt.time_limit), 500))
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });

  BnbOptions o;
  o.seed = static_cast<unsigned>(opt.seed);
  o.integral = integral;
  o.time_limit = opt.time_limit;
  o.cuts = opt.cuts < 0 ? 15 : opt.cuts;
  const BnbResult r = branch_and_bound(L, H, o);
  done = true;
  for (auto& t : pool) t.join();

  static const char* names[] = {"exhausted", "stopped", "failed", "lp_infeasible"};
  res.bnb_status = names[r.status];
  if (r.status != BnbResult::LP_INFEASIBLE && r.status != BnbResult::FAILED) res.lp_root = r.root_lp + L.offset;
  res.lp_root_time = r.root_time;
  res.bnb_nodes = r.nodes;
  res.bnb_lp_iters = r.lp_iters;
  res.bnb_strong = r.strong_lps;
  res.bnb_cols = r.lp_cols;
  res.lp_root_cut = r.root_cut + L.offset;
  res.cuts = r.cuts;
  res.cut_rounds = r.cut_rounds;
  res.cut_time = r.cut_time;
  res.node_fixed = r.node_fixed;
  res.passes = lns_calls;      // neighbourhood searches run
  res.ls_moves = lns_improved;  // ... that found a better solution
  res.runtime = elapsed();
  const bool have = ub < INF;
  if (r.status == BnbResult::LP_INFEASIBLE || (r.status == BnbResult::EXHAUSTED && !have)) {
    res.status = "infeasible";
    return res;
  }
  if (!have) {
    res.status = "no_solution";
    res.bound = shown(lb);
    return res;
  }
  res.cols = best;
  res.obj = ub + L.offset;
  if (r.status == BnbResult::EXHAUSTED) {
    res.status = "optimal";
    res.bound = opt.mip_gap > 0.0 ? std::min(shown(lb), res.obj) : res.obj;
  } else {
    res.status = "time_limit";
    res.bound = lb == -INF ? NaN : std::min(shown(lb), res.obj);
  }
  return res;
}

// Presolve for set partitioning rows (unit coefficients, = 1), in the spirit of Borndoerfer (1998):
//   - of two identical columns that cannot both be one, the dearer goes;
//   - partition row r1 inside partition row r2 (as column sets): the columns of r2 outside r1 are zero,
//     and r2 then repeats r1;
//   - a partition row with one column fixes it at one; a set row whose upper side is used up empties.
// Rows of other kinds stay as they are (their sides move when a column is fixed at one).
struct BinPresolve {
  LpProblem red;
  std::vector<int> col_orig;   // reduced column -> original column
  std::vector<int> to_red;     // original column -> reduced column, or -1
  std::vector<int> rep;        // removed duplicate -> the column kept instead (else -1)
  std::vector<int> fixed_one;
  bool infeasible = false, changed = false;
};

BinPresolve presolve_bin(const LpProblem& L) {
  BinPresolve Q;
  const int n = L.n, m = L.m;
  bool nonneg = true;
  for (double v : L.cval)
    if (v < 0.0) nonneg = false;
  std::vector<double> rlo = L.rlo, rup = L.rup;
  std::vector<char> ca(n, 1), ra(m, 1), unit(m, 1);
  Q.rep.assign(n, -1);
  // row-wise copy
  std::vector<int> rptr(m + 1, 0), rcol(L.cidx.size());
  for (int i : L.cidx) ++rptr[i + 1];
  for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
  {
    std::vector<int> pos(rptr.begin(), rptr.end() - 1);
    for (int j = 0; j < n; ++j)
      for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) {
        rcol[pos[L.cidx[q]]++] = j;
        if (L.cval[q] != 1.0) unit[L.cidx[q]] = 0;
      }
  }
  std::vector<int> rsize(m, 0), csize(n, 0);
  for (int i = 0; i < m; ++i) rsize[i] = rptr[i + 1] - rptr[i];
  for (int j = 0; j < n; ++j) csize[j] = L.cptr[j + 1] - L.cptr[j];
  auto part = [&](int i) { return ra[i] && unit[i] && rlo[i] == 1.0 && rup[i] == 1.0; };
  auto drop_col = [&](int j) {
    if (!ca[j]) return;
    ca[j] = 0;
    Q.changed = true;
    for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) --rsize[L.cidx[q]];
  };
  std::vector<int> work;
  auto fix_one = [&](int j) {
    if (!ca[j]) return;
    Q.fixed_one.push_back(j);
    ca[j] = 0;
    Q.changed = true;
    for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) {
      const int i = L.cidx[q];
      --rsize[i];
      if (rlo[i] > -INF) rlo[i] -= L.cval[q];
      if (rup[i] < INF) rup[i] -= L.cval[q];
      work.push_back(i);
    }
  };

  // identical columns
  if (nonneg) {
    std::vector<int> ord(n);
    for (int j = 0; j < n; ++j) ord[j] = j;
    auto same = [&](int a, int b) {
      if (csize[a] != csize[b]) return false;
      for (int q = 0; q < csize[a]; ++q)
        if (L.cidx[L.cptr[a] + q] != L.cidx[L.cptr[b] + q] || L.cval[L.cptr[a] + q] != L.cval[L.cptr[b] + q]) return false;
      return true;
    };
    auto less = [&](int a, int b) {
      if (csize[a] != csize[b]) return csize[a] < csize[b];
      for (int q = 0; q < csize[a]; ++q) {
        const int ia = L.cidx[L.cptr[a] + q], ib = L.cidx[L.cptr[b] + q];
        if (ia != ib) return ia < ib;
        const double va = L.cval[L.cptr[a] + q], vb = L.cval[L.cptr[b] + q];
        if (va != vb) return va < vb;
      }
      if (L.cost[a] != L.cost[b]) return L.cost[a] < L.cost[b];
      return a < b;
    };
    std::sort(ord.begin(), ord.end(), less);
    for (int a = 0; a < n;) {
      int b = a + 1;
      while (b < n && same(ord[a], ord[b])) ++b;
      if (b > a + 1 && csize[ord[a]] > 0) {
        bool exclusive = false;  // some row has no room for two of them
        const int j = ord[a];
        for (int q = L.cptr[j]; q < L.cptr[j + 1] && !exclusive; ++q) exclusive = 2.0 * L.cval[q] > rup[L.cidx[q]] + 1e-9;
        if (exclusive)
          for (int k = a + 1; k < b; ++k) {
            Q.rep[ord[k]] = j;
            drop_col(ord[k]);
          }
      }
      a = b;
    }
  }

  std::vector<int> stamp(n, -1);
  int tick = 0;
  bool again = nonneg;
  for (int pass = 0; pass < 20 && again && !Q.infeasible; ++pass) {
    again = false;
    // rows that are used up or have one column left
    for (int i = 0; i < m; ++i) work.push_back(i);
    while (!work.empty() && !Q.infeasible) {
      const int i = work.back();
      work.pop_back();
      if (!ra[i] || !unit[i]) continue;
      if (rup[i] < -1e-9 || (rsize[i] == 0 && rlo[i] > 1e-9)) {
        Q.infeasible = true;
        break;
      }
      if (rup[i] < 1.0 - 1e-9 && rsize[i] > 0) {  // no column of this row can be one any more
        for (int q = rptr[i]; q < rptr[i + 1]; ++q)
          if (ca[rcol[q]]) {
            const int j = rcol[q];
            drop_col(j);
            for (int a = L.cptr[j]; a < L.cptr[j + 1]; ++a) work.push_back(L.cidx[a]);
          }
        again = true;
      }
      if (rsize[i] == 0) {
        if (rlo[i] > 1e-9) Q.infeasible = true;
        else ra[i] = 0, Q.changed = true;
        continue;
      }
      if (rsize[i] == 1 && rlo[i] == 1.0) {  // the last column has to cover the row
        for (int q = rptr[i]; q < rptr[i + 1]; ++q)
          if (ca[rcol[q]]) {
            fix_one(rcol[q]);
            break;
          }
        work.push_back(i);
        again = true;
      }
    }
    if (Q.infeasible) break;
    // 1-step conflict probing on packing/partition rows: if x_j = 1 eliminates every alive column
    // of some partition row r2 that j does not cover, then x_j must be 0.
    if (nonneg && m <= 2000 && n <= 25000) {
      std::vector<int> stamp_row(m, -1), stamp_col(n, -1), hit_cnt(m, 0), touched_rows;
      long probe_work = 0;
      const long probe_limit = 3000000L;
      for (int j = 0; j < n && probe_work < probe_limit; ++j) {
        if (!ca[j] || csize[j] == 0) continue;
        bool has_pack = false;
        for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) {
          const int i = L.cidx[q];
          if (ra[i]) {
            stamp_row[i] = j;
            if (unit[i] && rup[i] == 1.0 && rptr[i + 1] - rptr[i] <= 200) has_pack = true;
          }
        }
        if (!has_pack) continue;
        touched_rows.clear();
        bool bad_j = false;
        for (int q = L.cptr[j]; q < L.cptr[j + 1] && !bad_j && probe_work < probe_limit; ++q) {
          const int i = L.cidx[q];
          if (!ra[i] || !unit[i] || rup[i] != 1.0 || rptr[i + 1] - rptr[i] > 200) continue;
          for (int a = rptr[i]; a < rptr[i + 1] && !bad_j; ++a) {
            const int k = rcol[a];
            if (!ca[k] || k == j || stamp_col[k] == j) continue;
            stamp_col[k] = j;
            for (int b = L.cptr[k]; b < L.cptr[k + 1]; ++b) {
              const int r2 = L.cidx[b];
              ++probe_work;
              if (!part(r2) || stamp_row[r2] == j) continue;
              if (hit_cnt[r2]++ == 0) touched_rows.push_back(r2);
              if (hit_cnt[r2] == rsize[r2]) {
                bad_j = true;
                break;
              }
            }
          }
        }
        for (int r2 : touched_rows) hit_cnt[r2] = 0;
        if (bad_j) {
          drop_col(j);
          again = true;
        }
      }
      if (again) continue;
    }
    // partition row inside partition row
    for (int r1 = 0; r1 < m; ++r1) {
      if (!part(r1) || rsize[r1] == 0) continue;
      int cmin = -1, best = 1 << 30;
      ++tick;
      for (int q = rptr[r1]; q < rptr[r1 + 1]; ++q) {
        const int j = rcol[q];
        if (!ca[j]) continue;
        stamp[j] = tick;
        int alive_rows = 0;
        for (int a = L.cptr[j]; a < L.cptr[j + 1]; ++a) alive_rows += ra[L.cidx[a]];
        if (alive_rows < best) best = alive_rows, cmin = j;
      }
      for (int a = L.cptr[cmin]; a < L.cptr[cmin + 1]; ++a) {
        const int r2 = L.cidx[a];
        if (r2 == r1 || !part(r2) || rsize[r2] < rsize[r1]) continue;
        int common = 0;
        for (int q = rptr[r2]; q < rptr[r2 + 1]; ++q) common += ca[rcol[q]] && stamp[rcol[q]] == tick;
        if (common != rsize[r1]) continue;
        for (int q = rptr[r2]; q < rptr[r2 + 1]; ++q)
          if (ca[rcol[q]] && stamp[rcol[q]] != tick) drop_col(rcol[q]);
        ra[r2] = 0;  // the same columns as r1 now
        Q.changed = true;
        again = true;
      }
    }
  }
  if (Q.infeasible || !Q.changed) return Q;
  for (int i = 0; i < m; ++i)
    if (ra[i] && rsize[i] == 0) {  // a row without columns is either satisfied or impossible
      if (rlo[i] > 1e-9 || rup[i] < -1e-9) {
        Q.infeasible = true;
        return Q;
      }
      ra[i] = 0;
    }

  Q.to_red.assign(n, -1);
  std::vector<int> row_new(m, -1);
  LpProblem& P = Q.red;
  for (int i = 0; i < m; ++i)
    if (ra[i]) {
      row_new[i] = P.m++;
      P.rlo.push_back(rlo[i]);
      P.rup.push_back(rup[i]);
    }
  P.cptr.assign(1, 0);
  P.offset = L.offset;
  for (int j : Q.fixed_one) P.offset += L.cost[j];
  for (int j = 0; j < n; ++j) {
    if (!ca[j]) continue;
    Q.to_red[j] = P.n++;
    Q.col_orig.push_back(j);
    P.cost.push_back(L.cost[j]);
    for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q)
      if (row_new[L.cidx[q]] >= 0) {
        P.cidx.push_back(row_new[L.cidx[q]]);
        P.cval.push_back(L.cval[q]);
      }
    P.cptr.push_back(static_cast<int>(P.cidx.size()));
  }
  P.lb.assign(P.n, 0.0);
  P.ub.assign(P.n, 1.0);
  // a removed row that is not satisfied by the fixed columns alone would have been reported above
  return Q;
}
}  // namespace

SolveResult solve_bin(const LpProblem& L, const SolveOptions& opt, const std::vector<int>* warm) {
  const auto t0 = Clock::now();
  if (!opt.presolve) return solve_core(L, opt, warm, t0);
  BinPresolve Q = presolve_bin(L);
  if (Q.infeasible) {
    SolveResult res;
    res.red_cols = L.n;
    res.red_rows = L.m;
    res.bound = NaN;
    res.lp_root = NaN;
    res.status = "infeasible";
    res.runtime = std::chrono::duration<double>(Clock::now() - t0).count();
    return res;
  }
  if (!Q.changed) return solve_core(L, opt, warm, t0);
  // the warm start in the reduced columns: a removed duplicate is replaced by the column kept for it
  std::vector<int> w;
  bool warm_ok = warm && !warm->empty();
  if (warm_ok) {
    std::vector<char> is_fixed(L.n, 0);
    for (int j : Q.fixed_one) is_fixed[j] = 1;
    for (int j0 : *warm) {
      if (j0 < 0 || j0 >= L.n) {
        warm_ok = false;
        break;
      }
      const int j = Q.rep[j0] >= 0 ? Q.rep[j0] : j0;
      if (is_fixed[j]) continue;
      if (Q.to_red[j] < 0) {
        warm_ok = false;  // uses a column the presolve proved to be zero: not a feasible solution
        break;
      }
      w.push_back(Q.to_red[j]);
    }
  }
  SolveResult res = solve_core(Q.red, opt, warm_ok ? &w : nullptr, t0);
  res.fixed_cols = static_cast<int>(Q.fixed_one.size());
  if (!res.cols.empty() || res.status == "optimal" || res.status == "time_limit") {
    std::vector<int> cols = Q.fixed_one;
    for (int k : res.cols) cols.push_back(Q.col_orig[k]);
    std::sort(cols.begin(), cols.end());
    res.cols = cols;
  }
  return res;
}

}  // namespace scsp
