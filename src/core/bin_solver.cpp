#include "scsp/bin_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>

#include "scsp/bnb.hpp"

namespace scsp {

namespace {
const double INF = std::numeric_limits<double>::infinity();
const double NaN = std::numeric_limits<double>::quiet_NaN();
using Clock = std::chrono::steady_clock;
}  // namespace

SolveResult solve_bin(const LpProblem& L, const SolveOptions& opt, const std::vector<int>* warm) {
  const auto t0 = Clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  SolveResult res;
  res.red_cols = L.n;
  res.red_rows = L.m;
  res.bound = NaN;
  res.lp_root = NaN;

  bool integral = std::floor(L.offset) == L.offset;
  for (double c : L.cost)
    if (std::floor(c) != c) integral = false;

  std::mutex mu;
  std::vector<int> best;
  double ub = INF, lb = -INF, last_lb_trace = -1.0;  // costs without the offset
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

  BnbHooks H;
  H.thr = thr;
  H.offer = offer;
  H.offer_lb = [&](double l) {
    std::lock_guard<std::mutex> g(mu);
    if (l <= lb) return;
    lb = l;
    const double t = elapsed();
    if (t - last_lb_trace >= 0.1) {
      last_lb_trace = t;
      res.trace.push_back({t, ub == INF ? NaN : ub + L.offset, shown(l), 1});
    }
  };
  H.stop = [&]() { return elapsed() >= opt.time_limit; };
  BnbOptions o;
  o.seed = static_cast<unsigned>(opt.seed);
  o.integral = integral;
  o.time_limit = opt.time_limit;
  const BnbResult r = branch_and_bound(L, H, o);

  static const char* names[] = {"exhausted", "stopped", "failed", "lp_infeasible"};
  res.bnb_status = names[r.status];
  if (r.status != BnbResult::LP_INFEASIBLE && r.status != BnbResult::FAILED) res.lp_root = r.root_lp + L.offset;
  res.lp_root_time = r.root_time;
  res.bnb_nodes = r.nodes;
  res.bnb_lp_iters = r.lp_iters;
  res.bnb_strong = r.strong_lps;
  res.bnb_cols = r.lp_cols;
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

}  // namespace scsp
