#include "scsp/bnb.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "scsp/cuts.hpp"

namespace scsp {

bool BnbTeam::help(const std::function<bool()>& stay) {
  std::function<void(const std::function<bool()>&)> fn;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!fn_) return false;
    fn = fn_;
    ++in_flight_;
  }
  struct Guard {
    BnbTeam* t;
    ~Guard() {
      std::lock_guard<std::mutex> lk(t->mu_);
      if (--t->in_flight_ == 0) t->done_cv_.notify_all();
    }
  } guard{this};
  fn(stay);
  return true;
}

void BnbTeam::activate(std::function<void(const std::function<bool()>&)> fn) {
  std::lock_guard<std::mutex> lk(mu_);
  fn_ = std::move(fn);
  is_active_.store(true, std::memory_order_release);
}

void BnbTeam::deactivate() {
  std::unique_lock<std::mutex> lk(mu_);
  is_active_.store(false, std::memory_order_release);
  fn_ = nullptr;
  done_cv_.wait(lk, [&]() { return in_flight_ == 0; });
}

namespace {

const double INF = std::numeric_limits<double>::infinity();
const double FRAC = 1e-6;

using Clock = std::chrono::steady_clock;

struct Node {
  double bound = -INF;   // LP value of the parent (or a strong branching estimate)
  int depth = 0;
  std::vector<int> fix;  // LP column * 2 + value
  // how this node was created, for the pseudocost update
  int bvar = -1, bdir = 0;
  double bfrac = 0.0, pz = 0.0;
  // optimal basis of the parent: a node taken from the queue starts from it instead of from wherever
  // the LP was left (shared by the nodes created from the same parent)
  std::shared_ptr<const std::vector<char>> basis;
};

struct NodeCmp {
  bool operator()(const Node* a, const Node* b) const {
    return a->bound > b->bound || (a->bound == b->bound && a->depth < b->depth);
  }
};

// One tree search over a fixed subset of the columns, supporting parallel workers via BnbTeam
// and/or internal threads (P5).
class Search {
 public:
  // SWITCH: a restricted search met its purpose (an incumbent exists); the exact search should take over
  enum { EXHAUSTED = 0, STOPPED = 1, INCOMPLETE = 2, NO_INCUMBENT = 3, SWITCH = 4 };

  Search(const LpProblem& P_, const BnbHooks& H_, const BnbOptions& o_, const std::vector<double>& droot_,
         double zroot_, std::vector<int> cols_, bool exact_, double time_left_, BnbResult& R_)
      : P(P_), H(H_), o(o_), droot(droot_), zroot(zroot_), cols(std::move(cols_)), exact(exact_),
        time_left(time_left_), R(R_), total_nodes(R_.nodes), rins_time_acc(R_.rins_time) {
    LpOptions lo;
    lo.seed = o.seed;
    lo.time_limit = time_left;
    lo.stop = &abort_lp;
    lp0.reset(new LpSolver(P, cols, lo));
    nk = static_cast<int>(cols.size());
    dk.resize(nk);
    for (int k = 0; k < nk; ++k) dk[k] = droot[cols[k]];
    zr0 = zroot;
    cached_gb.store(zr0, std::memory_order_relaxed);
    for (int d = 0; d < 2; ++d) {
      pcs[d].assign(nk, 0.0);
      pcn[d].assign(nk, 0);
    }
    nonneg = true;
    for (double v : P.cval)
      if (v < 0.0) nonneg = false;
    for (int i = 0; i < P.m; ++i)
      if (P.rlo[i] < 0.0 && P.rlo[i] > -INF) nonneg = false;
    pure_sp = P.m > 0;
    for (int i = 0; i < P.m && pure_sp; ++i) pure_sp = P.rlo[i] == 1.0 && P.rup[i] == 1.0;
    for (double v : P.cval)
      if (v != 1.0) pure_sp = false;
    partition = pure_sp && o.ryan_foster;
    if (partition && static_cast<double>(P.m) * P.m > 8e6) partition = false;
    active_bound.push_back(INF);  // slot 0: main thread
  }

  ~Search() {
    for (Node* nd : open) delete nd;
    open.clear();
  }

  int run() {
    if (H.keep && exact && !check_keep(zr0)) {
      R.gave_up = true;
      R.lp_iters += lp0->iterations();
      return STOPPED;
    }
    if (o.cuts && exact && !root_cuts()) {
      R.gave_up = true;
      R.lp_iters += lp0->iterations();
      return STOPPED;
    }
    // Prepare template LP for parallel helpers right after root cuts (clean [0,1] bounds + optimal root basis).
    const bool may_parallel = H.team != nullptr || o.threads > 1;
    if (may_parallel) {
      template_lp = lp0->clone(o.seed, time_left, &abort_lp);
    }
    push_locked(new Node());

    // Build main worker (slot 0) taking ownership of lp0.
    Worker main_w(*this, 0, std::move(lp0));

    // First incumbent: one LP dive (good solutions on easy problems) and, for partitioning problems,
    // the cover search. If the best-first tree has still found nothing after a while, the feasibility
    // pump and restarted dives get their turn; a restricted column set that yields nothing is given up.
    if (H.thr() == INF) {
      bool ok = main_w.dive_for_incumbent(1, 0.1);
      if (ok && H.thr() == INF && pure_sp) ok = main_w.cover_search(0.1);
      if (!ok) {
        finish_worker(main_w);
        R.nodes = total_nodes.load(std::memory_order_relaxed);
        return STOPPED;
      }
    }
    heur_at = H.thr() == INF ? total_nodes.load(std::memory_order_relaxed) + 3000 : -1;
    heur_time = now() + 0.1 * o.time_limit;

    // Activate cooperative helpers (external BnbTeam and/or internal threads).
    if (H.team) {
      H.team->activate([this](const std::function<bool()>& stay) { run_helper(stay); });
    }
    std::vector<std::thread> internal_pool;
    for (int t = 1; t < o.threads; ++t) {
      internal_pool.emplace_back([this]() { run_helper(nullptr); });
    }

    worker_loop(main_w, nullptr);

    // Signal all helpers to exit and wait until every helper has finished before destroying Search.
    abort_lp.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lk(tree_mu);
      terminated.store(true, std::memory_order_release);
      tree_cv.notify_all();
    }
    for (auto& th : internal_pool) th.join();
    if (H.team) H.team->deactivate();

    finish_worker(main_w);
    if (gave_up_flag.load(std::memory_order_relaxed)) R.gave_up = true;
    R.lp_iters += rins_lp_iters;
    R.nodes = total_nodes.load(std::memory_order_relaxed);
    return exit_code;
  }

 private:
  const LpProblem& P;
  const BnbHooks& H;
  const BnbOptions& o;
  const std::vector<double>& droot;
  const double zroot;
  std::vector<int> cols;
  const bool exact;
  const double time_left;
  BnbResult& R;

  std::unique_ptr<LpSolver> lp0;
  std::unique_ptr<LpSolver> template_lp;
  std::mutex clone_mu;
  int nk = 0;
  std::vector<double> dk;   // reduced costs of the loaded columns at the root (after the cut rounds, if any)
  double zr0 = 0.0;         // ... and the root LP value they belong to
  bool nonneg = true, partition = false, pure_sp = false;

  const Clock::time_point t_start = Clock::now();
  double now() const { return std::chrono::duration<double>(Clock::now() - t_start).count(); }

  std::atomic<bool> abort_lp{false};
  std::atomic<bool> terminated{false};
  std::atomic<bool> gave_up_flag{false};
  std::atomic<long> total_nodes{0};
  std::atomic<size_t> open_size{0};
  std::atomic<double> cached_gb{-INF};
  std::atomic<bool> rins_busy{false};
  std::atomic<long> rins_at{0};
  std::mutex rins_mu;
  double rins_time_acc = 0.0;
  long rins_lp_iters = 0;

  long heur_at = -1;
  double heur_time = 0.0;

  // Protected by tree_mu
  std::mutex tree_mu;
  std::condition_variable tree_cv;
  std::vector<Node*> open;  // heap, best bound first
  std::vector<double> active_bound;
  std::vector<int> free_slots;
  int active_workers = 0;
  bool incomplete = false;
  int exit_code = EXHAUSTED;
  double last_offered_lb = -INF;

  // Pseudocosts protected by pc_mu
  std::mutex pc_mu;
  std::vector<double> pcs[2];
  std::vector<int> pcn[2];
  double pc_avg_sum[2] = {0.0, 0.0};
  long pc_nonzero[2] = {0, 0};

  // Keep hook guarded by keep_mu
  std::mutex keep_mu;

  struct Worker {
    Search& S;
    int wid;
    std::unique_ptr<LpSolver> lp;
    std::vector<int> fv, curfix, want, lfixed;
    std::vector<char> gfix, lfix, rmark;
    std::vector<double> x, dnode, act, pairw;
    std::vector<int> order, chosen, touched_pairs;
    double last_thr = INF;
    long sb_iters = 0, local_strong = 0, local_fixed = 0;

    Worker(Search& S_, int wid_, std::unique_ptr<LpSolver> lp_)
        : S(S_), wid(wid_), lp(std::move(lp_)), fv(S_.nk, -1), want(S_.nk, 0), gfix(S_.nk, 0), lfix(S_.nk, 0),
          act(S_.P.m, 0.0) {
      if (S.partition) {
        rmark.assign(S.P.m, 0);
        pairw.assign(static_cast<size_t>(S.P.m) * S.P.m, 0.0);
      }
    }

    void global_fix(double thr) {
      if (!(thr < last_thr)) return;
      last_thr = thr;
      for (int k = 0; k < S.nk; ++k) {
        if (gfix[k]) continue;
        const double dj = S.dk[k];
        if (dj > 0.0 && S.round_up(S.zr0 + dj - 1e-9 * std::max(1.0, std::fabs(S.zr0))) > thr) {
          gfix[k] = 1;
          if (fv[k] == -1) lp->set_col_bounds(k, 0.0, 0.0);
        }
      }
    }

    void local_fix(double z, double thr) {
      lp->get_red_cost(dnode);
      const double eps = 1e-7 * std::max(1.0, std::fabs(z));
      for (int k = 0; k < S.nk; ++k) {
        if (fv[k] != -1 || gfix[k] || lfix[k]) continue;
        const double dk_val = dnode[k];
        if (x[k] < 1e-9 && dk_val > 0.0 && S.round_up(z + dk_val - eps) > thr) {
          lfix[k] = 1;
          lfixed.push_back(k);
          lp->set_col_bounds(k, 0.0, 0.0);
          ++local_fixed;
        } else if (x[k] > 1.0 - 1e-9 && dk_val < 0.0 && S.round_up(z - dk_val - eps) > thr) {
          lfix[k] = 2;
          lfixed.push_back(k);
          lp->set_col_bounds(k, 1.0, 1.0);
          ++local_fixed;
        }
      }
    }

    void undo_local() {
      for (int k : lfixed) {
        lfix[k] = 0;
        if (fv[k] == -1) lp->set_col_bounds(k, 0.0, gfix[k] ? 0.0 : 1.0);
      }
      lfixed.clear();
    }

    void switch_to(const std::vector<int>& fix) {
      for (int e : fix) want[e >> 1] = (e & 1) + 1;
      for (int e : curfix) {
        const int k = e >> 1;
        if (want[k] - 1 != fv[k]) {
          fv[k] = -1;
          if (lfix[k]) lp->set_col_bounds(k, lfix[k] - 1.0, lfix[k] - 1.0);
          else lp->set_col_bounds(k, 0.0, gfix[k] ? 0.0 : 1.0);
        }
      }
      for (int e : fix) {
        const int k = e >> 1, v = e & 1;
        if (fv[k] != v) {
          fv[k] = v;
          lp->set_col_bounds(k, v, v);
        }
        want[k] = 0;
      }
      curfix = fix;
    }

    void offer_integral() {
      std::vector<int> sol;
      double cost = 0.0;
      for (int k = 0; k < S.nk; ++k)
        if (x[k] > 0.5) {
          sol.push_back(S.cols[k]);
          cost += S.P.cost[S.cols[k]];
        }
      std::sort(sol.begin(), sol.end());
      S.H.offer(sol, cost);
    }

    std::shared_ptr<const std::vector<char>> node_basis() const {
      const size_t osz = S.open_size.load(std::memory_order_relaxed);
      if (!S.o.node_basis || static_cast<double>(osz + 1) * (S.P.m + S.nk) > 4e8) return nullptr;
      return std::make_shared<const std::vector<char>>(lp->get_basis());
    }

    void round_greedy() {
      order.clear();
      for (int k = 0; k < S.nk; ++k)
        if (x[k] > FRAC) order.push_back(k);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (x[a] != x[b]) return x[a] > x[b];
        if (S.P.cost[S.cols[a]] != S.P.cost[S.cols[b]]) return S.P.cost[S.cols[a]] < S.P.cost[S.cols[b]];
        return a < b;
      });
      std::fill(act.begin(), act.end(), 0.0);
      chosen.clear();
      for (int k : order) {
        const int j = S.cols[k];
        bool helps = false, fits = true;
        for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1] && fits; ++q) {
          const int i = S.P.cidx[q];
          if (act[i] < S.P.rlo[i] - 1e-9) helps = true;
          if (act[i] + S.P.cval[q] > S.P.rup[i] + 1e-9) fits = false;
        }
        if (!helps || !fits) continue;
        chosen.push_back(j);
        for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1]; ++q) act[S.P.cidx[q]] += S.P.cval[q];
      }
      for (int i = 0; i < S.P.m; ++i)
        if (act[i] < S.P.rlo[i] - 1e-9) return;
      std::sort(chosen.begin(), chosen.end(),
                [&](int a, int b) { return S.P.cost[a] > S.P.cost[b] || (S.P.cost[a] == S.P.cost[b] && a < b); });
      std::vector<int> sol;
      double cost = 0.0;
      for (int j : chosen) {
        bool redundant = true;
        for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1] && redundant; ++q)
          if (act[S.P.cidx[q]] - S.P.cval[q] < S.P.rlo[S.P.cidx[q]] - 1e-9) redundant = false;
        if (redundant) {
          for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1]; ++q) act[S.P.cidx[q]] -= S.P.cval[q];
        } else {
          sol.push_back(j);
          cost += S.P.cost[j];
        }
      }
      if (cost <= S.H.thr()) {
        std::sort(sol.begin(), sol.end());
        S.H.offer(sol, cost);
      }
    }

    void rins() {
      if (!S.H.incumbent) return;
      const double t_begin = S.now(), thr0 = S.H.thr();
      std::vector<int> inc, ones, freec;
      S.H.incumbent(inc);
      if (inc.empty()) return;
      std::vector<char> is_inc(S.P.n, 0);
      for (int j : inc) is_inc[j] = 1;
      for (int k = 0; k < S.nk; ++k) {
        const int j = S.cols[k];
        if (is_inc[j] && x[k] > 1.0 - 1e-6) ones.push_back(j);
        else if (is_inc[j] || x[k] > 1e-6) freec.push_back(j);
        is_inc[j] = 0;
      }
      for (int j : inc)
        if (is_inc[j]) freec.push_back(j);
      std::sort(freec.begin(), freec.end());
      const SubProblem sub = make_sub(S.P, freec, ones, S.nonneg);
      if (!sub.ok || sub.P.n < 3) return;
      const double budget = std::max(0.05, 0.02 * S.o.time_limit), t_end = t_begin + budget;
      BnbHooks h;
      h.thr = [&]() { return S.H.thr() - sub.fixed_cost; };
      h.offer = [&](const std::vector<int>& sc, double c) {
        std::vector<int> full = ones;
        for (int k : sc) full.push_back(sub.col_of[k]);
        std::sort(full.begin(), full.end());
        S.H.offer(full, c + sub.fixed_cost);
      };
      h.offer_lb = [](double) {};
      h.stop = [&]() { return S.terminated.load(std::memory_order_relaxed) || S.stop() || S.now() >= t_end; };
      BnbOptions so = S.o;
      so.rins = false;
      so.threads = 1;
      long rcalls;
      {
        std::lock_guard<std::mutex> lk(S.rins_mu);
        rcalls = S.R.rins_calls;
      }
      so.seed = S.o.seed + static_cast<unsigned>(rcalls) * 7919u + 1u;
      so.node_limit = 100;
      so.time_limit = budget;
      so.cut_rounds = std::min(S.o.cut_rounds, 10);
      const BnbResult r = branch_and_bound(sub.P, h, so);
      const bool improved = S.H.thr() < thr0;
      {
        std::lock_guard<std::mutex> lk(S.rins_mu);
        S.rins_lp_iters += r.lp_iters;
        ++S.R.rins_calls;
        if (improved) ++S.R.rins_found;
        S.rins_time_acc += S.now() - t_begin;
        S.R.rins_time = S.rins_time_acc;
      }
      const long nn = S.total_nodes.load(std::memory_order_relaxed);
      S.rins_at.store(std::max<long>(20, nn * (improved ? 4 : 8)), std::memory_order_relaxed);
    }

    // Exact cover search for pure partitioning problems, without LPs (called on worker 0 only).
    bool cover_search(double time_frac) {
      const int m = S.P.m;
      std::vector<int> rptr(m + 1, 0), rcol;
      for (int k = 0; k < S.nk; ++k)
        if (!gfix[k])
          for (int q = S.P.cptr[S.cols[k]]; q < S.P.cptr[S.cols[k] + 1]; ++q) ++rptr[S.P.cidx[q] + 1];
      for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
      rcol.resize(rptr[m]);
      {
        std::vector<int> pos(rptr.begin(), rptr.end() - 1);
        for (int k = 0; k < S.nk; ++k)
          if (!gfix[k])
            for (int q = S.P.cptr[S.cols[k]]; q < S.P.cptr[S.cols[k] + 1]; ++q) rcol[pos[S.P.cidx[q]]++] = k;
      }
      std::vector<double> key(S.nk);
      std::vector<int> ban(S.nk, 0), avail(m), chosen_k;
      std::vector<char> cov(m, 0);
      std::mt19937 rng(S.o.seed * 69069u + 4242u);
      std::uniform_real_distribution<double> uni(0.0, 1.0);
      double dmax = 1e-12;
      for (int k = 0; k < S.nk; ++k) dmax = std::max(dmax, std::fabs(S.droot[S.cols[k]]));
      const double t0 = S.now();
      long limit = 2000;
      bool found = false;
      struct Frame {
        int row, next;
        std::vector<int> cands;
      };
      for (int round = 0; !found; ++round, limit *= 2) {
        const double noise = round == 0 ? 0.0 : 0.3;
        for (int k = 0; k < S.nk; ++k) key[k] = S.droot[S.cols[k]] / dmax + noise * uni(rng);
        std::fill(ban.begin(), ban.end(), 0);
        std::fill(cov.begin(), cov.end(), 0);
        for (int i = 0; i < m; ++i) avail[i] = rptr[i + 1] - rptr[i];
        chosen_k.clear();
        std::vector<Frame> stack;
        long nodes = 0;
        int ncov = 0;
        auto choose = [&](int k) {
          const int j = S.cols[k];
          for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1]; ++q) {
            const int r = S.P.cidx[q];
            cov[r] = 1;
            ++ncov;
            for (int a = rptr[r]; a < rptr[r + 1]; ++a) {
              const int k2 = rcol[a];
              if (ban[k2]++ == 0)
                for (int b = S.P.cptr[S.cols[k2]]; b < S.P.cptr[S.cols[k2] + 1]; ++b) --avail[S.P.cidx[b]];
            }
          }
          chosen_k.push_back(k);
        };
        auto unchoose = [&]() {
          const int k = chosen_k.back();
          chosen_k.pop_back();
          const int j = S.cols[k];
          for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1]; ++q) {
            const int r = S.P.cidx[q];
            cov[r] = 0;
            --ncov;
            for (int a = rptr[r]; a < rptr[r + 1]; ++a) {
              const int k2 = rcol[a];
              if (--ban[k2] == 0)
                for (int b = S.P.cptr[S.cols[k2]]; b < S.P.cptr[S.cols[k2] + 1]; ++b) ++avail[S.P.cidx[b]];
            }
          }
        };
        auto open_row = [&]() {
          int best = -1;
          for (int i = 0; i < m; ++i)
            if (!cov[i] && (best < 0 || avail[i] < avail[best])) best = i;
          if (best < 0 || avail[best] == 0) return false;
          Frame f;
          f.row = best;
          f.next = 0;
          for (int a = rptr[best]; a < rptr[best + 1]; ++a)
            if (ban[rcol[a]] == 0) f.cands.push_back(rcol[a]);
          std::sort(f.cands.begin(), f.cands.end(),
                    [&](int a, int b) { return key[a] < key[b] || (key[a] == key[b] && a < b); });
          stack.push_back(std::move(f));
          return true;
        };
        bool exhausted = false;
        if (!open_row()) exhausted = true;
        while (!exhausted && !found) {
          if ((nodes & 255) == 0 && (S.stop() || S.now() - t0 > time_frac * S.o.time_limit)) return !S.stop();
          if (nodes > limit) break;
          Frame& f = stack.back();
          if (f.next > 0) unchoose();
          if (f.next >= static_cast<int>(f.cands.size())) {
            stack.pop_back();
            if (stack.empty()) exhausted = true;
            continue;
          }
          choose(f.cands[f.next++]);
          ++nodes;
          ++S.R.cover_nodes;
          if (ncov == m) {
            found = true;
            break;
          }
          if (!open_row()) continue;
        }
        if (found) {
          std::vector<int> sol;
          double cost = 0.0;
          for (int k : chosen_k) {
            sol.push_back(S.cols[k]);
            cost += S.P.cost[S.cols[k]];
          }
          std::sort(sol.begin(), sol.end());
          S.H.offer(sol, cost);
        }
        if (exhausted) break;
      }
      return true;
    }

    // Feasibility pump (called on worker 0 only).
    bool feasibility_pump(double time_frac) {
      switch_to({});
      if (lp->solve() != LpSolver::OPTIMAL) return !S.stop();
      ++S.total_nodes;
      lp->get_x(x);
      std::mt19937 rng(S.o.seed * 40503u + 977u);
      std::vector<char> xt(S.nk), prev1, prev2;
      for (int k = 0; k < S.nk; ++k) xt[k] = x[k] > 0.5;
      double cmax = 0.0;
      for (double c : S.P.cost) cmax = std::max(cmax, std::fabs(c));
      if (cmax == 0.0) cmax = 1.0;
      const double t0 = S.now();
      const double budget = time_frac * S.o.time_limit;
      std::vector<std::pair<double, int>> sc;
      bool ok = true, found = false;
      for (int it = 0; it < 300 && !found; ++it) {
        if (S.stop()) {
          ok = false;
          break;
        }
        if (S.now() - t0 > budget || S.H.thr() < INF) break;
        for (int k = 0; k < S.nk; ++k) lp->set_col_cost(k, xt[k] ? -cmax : cmax);
        if (lp->solve() != LpSolver::OPTIMAL) break;
        ++S.R.pump_iters;
        lp->get_x(x);
        bool integral_point = true, same = true;
        for (int k = 0; k < S.nk; ++k) {
          if (x[k] > FRAC && x[k] < 1.0 - FRAC) integral_point = false;
          if ((x[k] > 0.5) != (xt[k] != 0)) same = false;
        }
        if (integral_point) {
          found = true;
          break;
        }
        if (!same) {
          prev2 = prev1;
          prev1 = xt;
          for (int k = 0; k < S.nk; ++k) xt[k] = x[k] > 0.5;
          if (xt != prev2) continue;
        }
        sc.clear();
        for (int k = 0; k < S.nk; ++k) {
          const double dist = std::fabs(x[k] - (xt[k] ? 1.0 : 0.0));
          if (dist > 1e-9) sc.push_back({-dist, k});
        }
        std::sort(sc.begin(), sc.end());
        const size_t flips = std::min<size_t>(sc.size(), 10 + rng() % 20);
        for (size_t q = 0; q < flips; ++q) xt[sc[q].second] ^= 1;
        if (sc.empty() || it % 30 == 29)
          for (int k = 0; k < S.nk; ++k)
            if (x[k] > FRAC && rng() % 4 == 0) xt[k] ^= 1;
      }
      for (int k = 0; k < S.nk; ++k) lp->set_col_cost(k, S.P.cost[S.cols[k]]);
      if (found) {
        offer_integral();
        ++S.R.pump_found;
      }
      return ok;
    }

    // Restarted randomized diving while there is no incumbent (called on worker 0 only).
    bool dive_for_incumbent(int max_dives, double time_frac) {
      std::mt19937 rng(S.o.seed * 2654435761u + 12345u);
      std::vector<int> fix, cand;
      const double t0 = S.now();
      const double budget = time_frac * S.o.time_limit;
      for (int dive = 0; dive < max_dives && S.H.thr() == INF; ++dive) {
        fix.clear();
        int flips = 0;
        const int max_flips = 10 + 5 * std::min(dive, 40);
        for (;;) {
          if (S.stop()) return false;
          if (S.now() - t0 > budget) {
            switch_to({});
            return true;
          }
          switch_to(fix);
          const LpSolver::Status st = lp->solve();
          ++S.total_nodes;
          ++S.R.dive_nodes;
          if (st != LpSolver::OPTIMAL) {
            if (st != LpSolver::INFEASIBLE && S.stop()) return false;
            while (!fix.empty() && (fix.back() & 1) == 0) fix.pop_back();
            if (fix.empty()) {
              if (dive == 0 && flips == 0) {
                switch_to({});
                return true;
              }
              break;
            }
            if (++flips > max_flips) break;
            fix.back() -= 1;
            continue;
          }
          lp->get_x(x);
          cand.clear();
          for (int k = 0; k < S.nk; ++k)
            if (x[k] > FRAC && x[k] < 1.0 - FRAC) cand.push_back(k);
          if (cand.empty()) {
            offer_integral();
            break;
          }
          if (S.nonneg && fix.empty()) round_greedy();
          std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(3, cand.size()), cand.end(),
                            [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
          const unsigned r = dive == 0 ? 0 : rng() % 6;
          const size_t pick = std::min<size_t>(r < 3 ? 0 : (r < 5 ? 1 : 2), cand.size() - 1);
          fix.push_back(cand[pick] * 2 + 1);
        }
        ++S.R.dives;
      }
      switch_to({});
      return true;
    }

    bool branch_pair(Node& cur, double z, const std::vector<int>& cand) {
      const size_t m = static_cast<size_t>(S.P.m);
      touched_pairs.clear();
      for (int k : cand) {
        const int j = S.cols[k];
        for (int a = S.P.cptr[j]; a < S.P.cptr[j + 1]; ++a)
          for (int b = a + 1; b < S.P.cptr[j + 1]; ++b) {
            const size_t id = static_cast<size_t>(S.P.cidx[a]) * m + S.P.cidx[b];
            if (pairw[id] == 0.0) touched_pairs.push_back(static_cast<int>(id));
            pairw[id] += x[k];
          }
      }
      int best = -1;
      double bd = 0.5 - 1e-6;
      for (int id : touched_pairs) {
        const double d = std::fabs(pairw[id] - 0.5);
        if (d < bd || (d == bd && id < best)) bd = d, best = id;
      }
      for (int id : touched_pairs) pairw[id] = 0.0;
      if (best < 0) return false;
      const int r = best / S.P.m, s = best % S.P.m;
      rmark[r] = rmark[s] = 1;
      const double zb = std::max(z, cur.bound);
      Node* differ = new Node();
      differ->basis = node_basis();
      differ->bound = zb;
      differ->depth = cur.depth + 1;
      differ->fix = cur.fix;
      for (int k = 0; k < S.nk; ++k) {
        if (fv[k] != -1 || gfix[k]) continue;
        const int j = S.cols[k];
        int hit = 0;
        for (int q = S.P.cptr[j]; q < S.P.cptr[j + 1]; ++q) hit += rmark[S.P.cidx[q]];
        if (hit == 1) cur.fix.push_back(k * 2);
        else if (hit == 2) differ->fix.push_back(k * 2);
      }
      rmark[r] = rmark[s] = 0;
      S.push_sibling_and_raise(wid, differ, zb);
      cur.bound = zb;
      ++cur.depth;
      return true;
    }

    int branch_var(Node& cur, double z, double thr, std::vector<int>& cand) {
      std::sort(cand.begin(), cand.end(), [&](int a, int b) {
        const double fa = std::fabs(x[a] - 0.5), fb = std::fabs(x[b] - 0.5);
        return fa < fb || (fa == fb && a < b);
      });
      struct CandPc {
        double s0, s1;
        int c0, c1;
      };
      std::vector<CandPc> cpc(cand.size());
      double avg[2] = {1.0, 1.0};
      {
        std::lock_guard<std::mutex> lk(S.pc_mu);
        for (int d = 0; d < 2; ++d)
          if (S.pc_nonzero[d] > 0) avg[d] = S.pc_avg_sum[d] / static_cast<double>(S.pc_nonzero[d]);
        for (size_t i = 0; i < cand.size(); ++i) {
          const int k = cand[i];
          cpc[i] = {S.pcs[0][k], S.pcs[1][k], S.pcn[0][k], S.pcn[1][k]};
        }
      }
      const bool may_strong = S.o.strong_cands > 0 &&
                              (cur.depth <= 8 || sb_iters < S.o.strong_share * static_cast<double>(lp->iterations()));
      const double cut = thr == INF ? INF : thr + S.P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
      LpSolver::State node_state;
      bool saved = false;
      int best = -1, tried = 0, since_best = 0;
      double best_score = -1.0;
      const std::vector<double> x0 = x;
      struct PcObs {
        int k;
        double v0, v1;
      };
      std::vector<PcObs> obs;
      for (size_t ci = 0; ci < cand.size(); ++ci) {
        const int k = cand[ci];
        const double f = x0[k];
        double d0, d1;
        const bool unreliable = cpc[ci].c0 < S.o.reliability || cpc[ci].c1 < S.o.reliability;
        if (may_strong && unreliable && tried < S.o.strong_cands && since_best < S.o.strong_lookahead) {
          ++tried;
          ++since_best;
          if (!saved) {
            lp->save_state(node_state);
            saved = true;
          }
          double zc[2];
          for (int v = 0; v < 2; ++v) {
            const long it0 = lp->iterations();
            lp->set_col_bounds(k, v, v);
            const LpSolver::Status st = lp->solve(cut, S.o.strong_iters);
            ++local_strong;
            zc[v] = (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF) ? INF
                    : (st == LpSolver::OPTIMAL || st == LpSolver::LIMIT) ? std::max(lp->obj() - S.P.offset, z) : z;
            if (zc[v] < INF && S.round_up(zc[v]) > thr) zc[v] = INF;
            sb_iters += lp->iterations() - it0;
            lp->set_col_bounds(k, 0.0, 1.0);
            lp->restore_state(node_state);
          }
          if (zc[0] == INF && zc[1] == INF) {
            if (!obs.empty()) S.record_pcs(obs);
            return -2;
          }
          if (zc[0] == INF || zc[1] == INF) {
            if (!obs.empty()) S.record_pcs(obs);
            cur.fix.push_back(k * 2 + (zc[0] == INF ? 1 : 0));
            return -1;
          }
          d0 = zc[0] - z;
          d1 = zc[1] - z;
          obs.push_back({k, d0 / std::max(f, 1e-9), d1 / std::max(1.0 - f, 1e-9)});
        } else {
          d0 = (cpc[ci].c0 ? cpc[ci].s0 / cpc[ci].c0 : avg[0]) * f;
          d1 = (cpc[ci].c1 ? cpc[ci].s1 / cpc[ci].c1 : avg[1]) * (1.0 - f);
        }
        const double score = std::max(d0, 1e-6) * std::max(d1, 1e-6);
        if (score > best_score) best_score = score, best = k, since_best = 0;
      }
      if (!obs.empty()) S.record_pcs(obs);
      const double zb = std::max(z, cur.bound);
      Node* down = new Node();
      down->basis = node_basis();
      down->bound = zb;
      down->depth = cur.depth + 1;
      down->fix = cur.fix;
      down->fix.push_back(best * 2);
      down->bvar = best;
      down->bdir = 0;
      down->bfrac = x0[best];
      down->pz = z;
      S.push_sibling_and_raise(wid, down, zb);
      cur.fix.push_back(best * 2 + 1);
      cur.bound = zb;
      ++cur.depth;
      cur.bvar = best;
      cur.bdir = 1;
      cur.bfrac = 1.0 - x0[best];
      cur.pz = z;
      return best;
    }
  };

  bool stop() const {
    return H.stop() || (o.node_limit >= 0 && total_nodes.load(std::memory_order_relaxed) >= o.node_limit);
  }
  double round_up(double z) const { return o.integral ? std::ceil(z - 1e-6) : z; }
  double ub_of(double thr) const {
    if (thr == INF) return INF;
    return o.integral ? thr + 1.0 - 1e-6 : thr;
  }

  double rins_time_sec() {
    std::lock_guard<std::mutex> lk(rins_mu);
    return rins_time_acc;
  }

  bool check_keep(double gb) {
    std::lock_guard<std::mutex> lk(keep_mu);
    return H.keep(gb, total_nodes.load(std::memory_order_relaxed));
  }

  void add_pc_locked(int d, int k, double val) {
    if (pcn[d][k] > 0) pc_avg_sum[d] -= pcs[d][k] / static_cast<double>(pcn[d][k]);
    else ++pc_nonzero[d];
    pcs[d][k] += val;
    ++pcn[d][k];
    pc_avg_sum[d] += pcs[d][k] / static_cast<double>(pcn[d][k]);
  }

  void record_pc(int d, int k, double val) {
    std::lock_guard<std::mutex> lk(pc_mu);
    add_pc_locked(d, k, val);
  }

  template <class ObsVec>
  void record_pcs(const ObsVec& obs) {
    std::lock_guard<std::mutex> lk(pc_mu);
    for (const auto& u : obs) {
      add_pc_locked(0, u.k, u.v0);
      add_pc_locked(1, u.k, u.v1);
    }
  }

  void push_locked(Node* nd) {
    open.push_back(nd);
    std::push_heap(open.begin(), open.end(), NodeCmp());
    open_size.store(open.size(), std::memory_order_relaxed);
  }

  void drop_open_locked() {
    for (Node* nd : open) delete nd;
    open.clear();
    open_size.store(0, std::memory_order_relaxed);
  }

  double global_bound_locked() const {
    double gb = INF;
    if (!open.empty()) gb = std::min(gb, open.front()->bound);
    for (double b : active_bound)
      if (b < gb) gb = b;
    return gb;
  }

  void update_cached_gb_locked() {
    const double gb = global_bound_locked();
    if (gb < INF) cached_gb.store(gb > -INF ? gb : zr0, std::memory_order_relaxed);
  }

  void terminate_locked(int code) {
    if (!terminated.load(std::memory_order_relaxed)) {
      exit_code = code;
      terminated.store(true, std::memory_order_release);
      abort_lp.store(true, std::memory_order_release);
      drop_open_locked();
      tree_cv.notify_all();
    }
  }

  void terminate(int code) {
    std::lock_guard<std::mutex> lk(tree_mu);
    terminate_locked(code);
  }

  void set_incomplete() {
    std::lock_guard<std::mutex> lk(tree_mu);
    incomplete = true;
  }

  void push_node(Node* nd) {
    std::lock_guard<std::mutex> lk(tree_mu);
    if (terminated.load(std::memory_order_relaxed)) {
      delete nd;
      return;
    }
    push_locked(nd);
    update_cached_gb_locked();
    tree_cv.notify_one();
  }

  void push_sibling_and_raise(int wid, Node* sibling, double new_bound) {
    std::lock_guard<std::mutex> lk(tree_mu);
    if (terminated.load(std::memory_order_relaxed)) {
      delete sibling;
      return;
    }
    push_locked(sibling);
    if (new_bound > active_bound[wid]) active_bound[wid] = new_bound;
    update_cached_gb_locked();
    tree_cv.notify_one();
  }

  void finish_worker(Worker& W) {
    std::lock_guard<std::mutex> lk(tree_mu);
    R.lp_iters += W.lp->iterations();
    R.strong_lps += W.local_strong;
    R.node_fixed += W.local_fixed;
    if (W.wid > 0) {
      active_bound[W.wid] = INF;
      free_slots.push_back(W.wid);
    }
  }

  void run_helper(const std::function<bool()>& stay) {
    if (terminated.load(std::memory_order_acquire) || stop()) return;
    int wid;
    {
      std::lock_guard<std::mutex> lk(tree_mu);
      if (terminated.load(std::memory_order_relaxed)) return;
      if (!free_slots.empty()) {
        wid = free_slots.back();
        free_slots.pop_back();
        active_bound[wid] = INF;
      } else {
        wid = static_cast<int>(active_bound.size());
        active_bound.push_back(INF);
      }
    }
    std::unique_ptr<LpSolver> wlp;
    {
      std::lock_guard<std::mutex> lk(clone_mu);
      if (terminated.load(std::memory_order_relaxed) || !template_lp) {
        std::lock_guard<std::mutex> lk2(tree_mu);
        free_slots.push_back(wid);
        return;
      }
      wlp = template_lp->clone(o.seed + 7919u * static_cast<unsigned>(wid), std::max(0.0, time_left - now()), &abort_lp);
    }
    Worker W(*this, wid, std::move(wlp));
    worker_loop(W, stay);
    finish_worker(W);
  }

  void worker_loop(Worker& W, const std::function<bool()>& stay) {
    const bool is_main = W.wid == 0;
    std::vector<int> cand;
    for (;;) {
      std::unique_ptr<Node> top;
      double thr = INF;
      double lb_to_offer = -INF;
      {
        std::unique_lock<std::mutex> lk(tree_mu);
        for (;;) {
          if (terminated.load(std::memory_order_relaxed)) return;
          if (stop()) {
            terminate_locked(STOPPED);
            return;
          }
          if (!is_main && stay && !stay()) return;
          thr = H.thr();
          if (!exact && thr < INF) {
            terminate_locked(SWITCH);
            return;
          }
          if (!open.empty() && round_up(open.front()->bound) > thr) {
            drop_open_locked();
          }
          if (open.empty()) {
            if (active_workers == 0) {
              terminate_locked(incomplete ? INCOMPLETE : EXHAUSTED);
              return;
            }
            tree_cv.wait_for(lk, std::chrono::milliseconds(5));
            continue;
          }
          if (is_main && heur_at >= 0 && (total_nodes.load(std::memory_order_relaxed) > heur_at || now() > heur_time)) {
            heur_at = -1;
            if (thr == INF) {
              active_bound[0] = open.front()->bound;
              ++active_workers;
              lk.unlock();
              bool ok = W.feasibility_pump(0.1);
              if (ok && H.thr() == INF) ok = W.dive_for_incumbent(1 << 30, 0.2);
              lk.lock();
              --active_workers;
              active_bound[0] = INF;
              if (!ok) {
                terminate_locked(STOPPED);
                return;
              }
              if (!exact && H.thr() == INF) {
                terminate_locked(NO_INCUMBENT);
                return;
              }
              tree_cv.notify_all();
              continue;
            }
          }
          std::pop_heap(open.begin(), open.end(), NodeCmp());
          top.reset(open.back());
          open.pop_back();
          open_size.store(open.size(), std::memory_order_relaxed);
          active_bound[W.wid] = top->bound;
          ++active_workers;
          const double gb = global_bound_locked();
          if (gb < INF) cached_gb.store(gb > -INF ? gb : zr0, std::memory_order_relaxed);
          if (exact && !incomplete && gb > -INF && gb < INF) {
            const double offer_b = std::min(round_up(gb), ub_of(thr));
            if (offer_b > last_offered_lb) {
              last_offered_lb = offer_b;
              lb_to_offer = offer_b;
            }
          }
          break;
        }
      }
      if (lb_to_offer > -INF) H.offer_lb(lb_to_offer);
      W.global_fix(thr);
      Node cur = std::move(*top);
      bool from_queue = true;
      for (;;) {
        if (terminated.load(std::memory_order_relaxed)) break;
        if (stop()) {
          terminate(STOPPED);
          break;
        }
        if (!is_main && stay && !stay()) {
          W.undo_local();
          Node* ret = new Node(std::move(cur));
          if (!from_queue) ret->basis = W.node_basis();
          push_node(ret);
          break;
        }
        if (H.keep && exact) {
          const double gb = cached_gb.load(std::memory_order_relaxed);
          if (!check_keep(gb)) {
            gave_up_flag.store(true, std::memory_order_relaxed);
            terminate(STOPPED);
            break;
          }
        }
        if (is_main && H.yield) {
          H.yield();
          thr = H.thr();
          W.global_fix(thr);
        }
        W.switch_to(cur.fix);
        if (from_queue && cur.basis) W.lp->set_basis(*cur.basis);
        from_queue = false;
        cur.basis.reset();
        const LpSolver::Status st = W.lp->solve(thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr)));
        const long nodes_now = ++total_nodes;
        if (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF) break;
        if (st != LpSolver::OPTIMAL) {
          if (terminated.load(std::memory_order_relaxed) || stop()) {
            if (stop()) terminate(STOPPED);
            break;
          }
          set_incomplete();
          break;
        }
        const double z = W.lp->obj() - P.offset;
        if (cur.bvar >= 0) {
          record_pc(cur.bdir, cur.bvar, std::max(z - cur.pz, 0.0) / std::max(cur.bfrac, 1e-9));
          cur.bvar = -1;
        }
        thr = H.thr();
        if (!exact && thr < INF) {
          terminate(SWITCH);
          break;
        }
        if (round_up(z) > thr) break;
        W.global_fix(thr);
        W.lp->get_x(W.x);
        if (H.lp_point && (nodes_now & 15) == 1) H.lp_point(cols, W.x);
        if (o.node_fixing && exact && thr < INF) W.local_fix(z, thr);
        if (o.rins && exact && thr < INF && nodes_now >= rins_at.load(std::memory_order_relaxed) &&
            rins_time_sec() < 0.08 * now() + 0.01 * o.time_limit) {
          if (!rins_busy.exchange(true, std::memory_order_acq_rel)) {
            W.rins();
            rins_busy.store(false, std::memory_order_release);
            thr = H.thr();
            if (round_up(z) > thr) break;
          }
        }
        cand.clear();
        for (int k = 0; k < nk; ++k)
          if (W.x[k] > FRAC && W.x[k] < 1.0 - FRAC) cand.push_back(k);
        if (cand.empty()) {
          W.offer_integral();
          break;
        }
        if (nonneg && (cur.depth == 0 || nodes_now % 10 == 0)) {
          W.round_greedy();
          thr = H.thr();
          if (round_up(z) > thr) break;
        }
        if (partition && W.branch_pair(cur, z, cand)) continue;
        const int r = W.branch_var(cur, z, thr, cand);
        if (r == -2) break;  // both children are infeasible or cut off
      }
      W.undo_local();
      {
        std::lock_guard<std::mutex> lk(tree_mu);
        active_bound[W.wid] = INF;
        --active_workers;
        update_cached_gb_locked();
        if (open.empty() && active_workers == 0 && !terminated.load(std::memory_order_relaxed)) {
          terminate_locked(incomplete ? INCOMPLETE : EXHAUSTED);
        } else {
          tree_cv.notify_all();
        }
      }
    }
  }

  // Cut rounds at the root: separate, add the most effective cuts, solve again with the dual simplex,
  // drop the cuts that are not tight. Afterwards the root value and reduced costs used for fixing are
  // those of the LP with cuts.
  // False when the watcher (BnbHooks::keep) gives the search up during the rounds.
  bool root_cuts() {
    const double t_begin = now(), t_end = t_begin + o.cut_time * o.time_limit;
    bool kept = true;
    if (lp0->solve(INF) != LpSolver::OPTIMAL) return true;
    const SetRows rows(P, cols);
    std::mt19937 rng(o.seed * 2654435761u + 99u);
    CutPool pool;
    std::vector<double> x;
    double z = lp0->obj() - P.offset;
    int stall = 0;
    // cuts are dense rows: their nonzeros together stay below those of the loaded columns (or 50000)
    long nnz0 = 0, cut_nnz = 0;
    for (int k = 0; k < nk; ++k) nnz0 += P.cptr[cols[k] + 1] - P.cptr[cols[k]];
    const long budget = std::max(nnz0, 50000L), longest = std::max(200L, nk / 5L);
    std::vector<int> cut_len;  // nonzeros of the cut rows now in the LP
    std::vector<int> ord, rptr, ridx;
    std::vector<double> rval, rlo, rup;
    for (int round = 0; round < o.cut_rounds && !stop() && now() < t_end; ++round) {
      lp0->get_x(x);
      pool.clear();
      const int cap = std::max(20, P.m / 4);
      if (o.cuts & 4) separate_cliques(rows, x, 0.02, cap, pool);
      if (o.cuts & 1) separate_zero_half(rows, x, true, o.cut_size, 0.02, cap, rng, pool);
      if (o.cuts & 2) separate_zero_half(rows, x, false, o.cut_size, 0.02, cap, rng, pool);
      if (pool.size() == 0) break;
      ord.resize(pool.size());
      for (int c = 0; c < pool.size(); ++c) ord[c] = c;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return pool.efficacy[a] > pool.efficacy[b] || (pool.efficacy[a] == pool.efficacy[b] && a < b);
      });
      rptr.assign(1, 0);
      ridx.clear();
      rval.clear();
      rlo.clear();
      rup.clear();
      for (int c : ord) {
        const long len = pool.ptr[c + 1] - pool.ptr[c];
        if (static_cast<int>(rlo.size()) >= cap) break;
        if (len > longest || cut_nnz + len > budget) continue;
        cut_nnz += len;
        cut_len.push_back(static_cast<int>(len));
        ridx.insert(ridx.end(), pool.idx.begin() + pool.ptr[c], pool.idx.begin() + pool.ptr[c + 1]);
        rval.insert(rval.end(), pool.val.begin() + pool.ptr[c], pool.val.begin() + pool.ptr[c + 1]);
        rptr.push_back(static_cast<int>(ridx.size()));
        rlo.push_back(pool.lo[c]);
        rup.push_back(pool.up[c]);
      }
      if (rlo.empty()) break;
      lp0->add_rows(rptr, ridx, rval, rlo, rup);
      ++R.cut_rounds;
      if (lp0->solve(INF) != LpSolver::OPTIMAL) break;
      const double zn = lp0->obj() - P.offset;
      const std::vector<char> rs = lp0->row_status();
      std::vector<int> loose;
      for (int i = P.m; i < static_cast<int>(rs.size()); ++i)
        if (rs[i] == 0) loose.push_back(i);
      if (!loose.empty()) {
        for (size_t a = loose.size(); a-- > 0;) {
          cut_nnz -= cut_len[loose[a] - P.m];
          cut_len.erase(cut_len.begin() + (loose[a] - P.m));
        }
        lp0->remove_rows(loose);
        if (lp0->solve(INF) != LpSolver::OPTIMAL) break;
      }
      stall = zn - z < 1e-5 * std::max(1.0, std::fabs(z)) ? stall + 1 : 0;
      z = zn;
      if (stall >= 3 || round_up(z) > H.thr()) break;
      if (H.keep && !check_keep(z)) {
        kept = false;
        break;
      }
    }
    // Cuts that close little of the gap relative to the fill-in they add only slow every node LP down.
    const double thr_now = H.thr();
    const double min_close =
        (o.cuts == 1 && o.integral) ? std::max(0.05, 0.25 * static_cast<double>(cut_nnz) / std::max(1L, nnz0)) : 0.05;
    if (kept && lp0->rows() > P.m && thr_now < INF && z - zr0 < min_close * (thr_now - zr0)) lp0->remove_rows(P.m);
    R.cuts = lp0->rows() - P.m;
    R.cut_time += now() - t_begin;
    if (lp0->solve(INF) == LpSolver::OPTIMAL) {
      zr0 = lp0->obj() - P.offset;
      lp0->get_red_cost(dk);
      R.root_cut = zr0;
      cached_gb.store(zr0, std::memory_order_relaxed);
    }
    return kept;
  }
};

}  // namespace

SubProblem make_sub(const LpProblem& L, const std::vector<int>& freec, const std::vector<int>& ones, bool nonneg) {
  SubProblem S;
  std::vector<double> act(L.m, 0.0);
  for (int j : ones) {
    S.fixed_cost += L.cost[j];
    for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) act[L.cidx[q]] += L.cval[q];
  }
  std::vector<int> row_new(L.m, -1);
  LpProblem& P = S.P;
  P.cptr.assign(1, 0);
  for (int j : freec) {
    bool fits = true;
    if (nonneg)
      for (int q = L.cptr[j]; q < L.cptr[j + 1] && fits; ++q)
        if (act[L.cidx[q]] + L.cval[q] > L.rup[L.cidx[q]] + 1e-9) fits = false;
    if (!fits) continue;
    for (int q = L.cptr[j]; q < L.cptr[j + 1]; ++q) {
      const int i = L.cidx[q];
      if (row_new[i] < 0) {
        row_new[i] = P.m++;
        P.rlo.push_back(L.rlo[i] - act[i]);
        P.rup.push_back(L.rup[i] - act[i]);
      }
      P.cidx.push_back(row_new[i]);
      P.cval.push_back(L.cval[q]);
    }
    P.cptr.push_back(static_cast<int>(P.cidx.size()));
    P.cost.push_back(L.cost[j]);
    S.col_of.push_back(j);
  }
  P.n = static_cast<int>(S.col_of.size());
  P.lb.assign(P.n, 0.0);
  P.ub.assign(P.n, 1.0);
  for (int i = 0; i < L.m; ++i)
    if (row_new[i] < 0 && (act[i] < L.rlo[i] - 1e-9 || act[i] > L.rup[i] + 1e-9)) return S;
  S.ok = true;
  return S;
}

BnbResult branch_and_bound(const LpProblem& P, const BnbHooks& H, const BnbOptions& opt) {
  BnbResult R;
  const auto t0 = Clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  const int n = P.n, m = P.m;

  LpOptions lo;
  lo.seed = opt.seed;
  lo.time_limit = std::min(opt.time_limit, opt.root_time_limit);
  const LpResult root = solve_lp(P, lo);
  R.root_time = elapsed();
  if (root.status == "infeasible") {
    R.status = BnbResult::LP_INFEASIBLE;
    return R;
  }
  if (root.status != "optimal") {
    if (root.status == "time_limit" && elapsed() < opt.time_limit && !H.stop()) {
      R.gave_up = true;  // the root LP alone is too expensive for the time there is
      R.status = BnbResult::STOPPED;
      return R;
    }
    R.status = H.stop() || elapsed() >= opt.time_limit ? BnbResult::STOPPED : BnbResult::FAILED;
    return R;
  }
  const double zroot = root.obj - P.offset;
  R.root_lp = zroot;
  R.lp_iters = root.dual_iters + root.primal_iters;
  const double zr = opt.integral ? std::ceil(zroot - 1e-6) : zroot;
  if (H.root) H.root(root.row_dual, root.red_cost, zroot);
  H.offer_lb(zroot);

  // columns ordered by reduced cost for the restricted searches
  std::vector<int> by_rc(n);
  for (int j = 0; j < n; ++j) by_rc[j] = j;
  std::sort(by_rc.begin(), by_rc.end(), [&](int a, int b) {
    return root.red_cost[a] < root.red_cost[b] || (root.red_cost[a] == root.red_cost[b] && a < b);
  });
  long target = std::max<long>(4L * m, 2000);

  for (;;) {
    if (H.stop() || elapsed() >= opt.time_limit) break;
    const double thr = H.thr();
    if (zr > thr) {  // the incumbent already matches the root bound
      R.status = BnbResult::EXHAUSTED;
      R.exact = true;
      return R;
    }
    std::vector<int> cols;
    bool exact;
    if (thr < INF) {
      for (int j = 0; j < n; ++j) {
        const double dj = root.red_cost[j];
        const double lbj = zroot + std::max(dj, 0.0) - 1e-9 * std::max(1.0, std::fabs(zroot));
        if ((opt.integral ? std::ceil(lbj - 1e-6) : lbj) <= thr) cols.push_back(j);
      }
      exact = true;
    } else {
      const long take = std::min<long>(n, target);
      cols.assign(by_rc.begin(), by_rc.begin() + take);
      // every column of the LP optimum must be present
      while (static_cast<long>(cols.size()) < n && root.red_cost[by_rc[cols.size()]] <= 1e-9) cols.push_back(by_rc[cols.size()]);
      std::sort(cols.begin(), cols.end());
      exact = static_cast<long>(cols.size()) == n;
    }
    R.lp_cols = static_cast<int>(cols.size());
    R.exact = exact;
    R.root_cut = zroot;
    ++R.restarts;
    int s;
    {
      Search search(P, H, opt, root.red_cost, zroot, std::move(cols), exact, std::max(0.0, opt.time_limit - elapsed()), R);
      s = search.run();
    }
    if (s == Search::STOPPED) break;
    if (s == Search::NO_INCUMBENT) {
      target *= 4;
      continue;
    }
    if (s == Search::SWITCH) continue;
    if (s == Search::INCOMPLETE) {
      R.status = BnbResult::FAILED;
      return R;
    }
    if (exact) {
      R.status = BnbResult::EXHAUSTED;
      return R;
    }
    // the restricted tree is finished: with an incumbent the next round is exact, otherwise widen it
    if (H.thr() == INF) target *= 4;
  }
  R.status = BnbResult::STOPPED;
  return R;
}

}  // namespace scsp
