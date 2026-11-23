#include "scsp/bnb.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
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

// Selects non-parallel cuts from `pool` by decreasing efficacy, respecting length, per-round, and nnz caps.
bool select_cuts_into(const CutPool& pool, int round_cap, long len_lim, long nnz_rem, double par_tol,
                      std::vector<int>& rptr, std::vector<int>& ridx, std::vector<double>& rval,
                      std::vector<double>& rlo, std::vector<double>& rup, std::vector<int>& sel_len,
                      std::vector<char>& sel_int, long& added_nnz, std::vector<int>& ord,
                      std::vector<double>& cdense, std::vector<double>& sel_norm) {
  if (pool.size() == 0 || round_cap <= 0 || nnz_rem <= 0) return false;
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
  sel_len.clear();
  sel_int.clear();
  sel_norm.clear();
  added_nnz = 0;
  for (int c : ord) {
    const long len = pool.ptr[c + 1] - pool.ptr[c];
    if (static_cast<int>(rlo.size()) >= round_cap) break;
    if (len > len_lim || added_nnz + len > nnz_rem) continue;
    const int p0 = pool.ptr[c], p1 = pool.ptr[c + 1];
    double cn2 = 0.0;
    for (int q = p0; q < p1; ++q) {
      cdense[pool.idx[q]] = pool.val[q];
      cn2 += pool.val[q] * pool.val[q];
    }
    const double cnorm = std::sqrt(cn2);
    bool parallel = false;
    const int nsel = static_cast<int>(sel_norm.size());
    for (int s = 0; s < nsel && !parallel; ++s) {
      double dot = 0.0;
      for (int q = rptr[s]; q < rptr[s + 1]; ++q) dot += cdense[ridx[q]] * rval[q];
      if (std::fabs(dot) > par_tol * cnorm * sel_norm[s]) parallel = true;
    }
    for (int q = p0; q < p1; ++q) cdense[pool.idx[q]] = 0.0;
    if (parallel) continue;

    added_nnz += len;
    sel_len.push_back(static_cast<int>(len));
    sel_int.push_back(pool.is_int[c]);
    sel_norm.push_back(cnorm);
    ridx.insert(ridx.end(), pool.idx.begin() + p0, pool.idx.begin() + p1);
    rval.insert(rval.end(), pool.val.begin() + p0, pool.val.begin() + p1);
    rptr.push_back(static_cast<int>(ridx.size()));
    rlo.push_back(pool.lo[c]);
    rup.push_back(pool.up[c]);
  }
  return !rlo.empty();
}

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

    // Exact integer nonnegative domain propagation tables (Savelsbergh 1994), read-only across workers.
    const int m = P.m;
    std::vector<char> prop_row(m, 1);
    for (int k = 0; k < nk; ++k) {
      const int j = cols[k];
      for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
        const double v = P.cval[q];
        if (v < 0.0 || v > 1e9 || std::floor(v) != v) prop_row[P.cidx[q]] = 0;
      }
    }
    row_has_lo.assign(m, 0);
    row_has_up.assign(m, 0);
    rlo_int.assign(m, 0);
    rup_int.assign(m, 0);
    row_max_coef.assign(m, 0);
    init_max_act.assign(m, 0);
    for (int i = 0; i < m; ++i) {
      if (!prop_row[i]) continue;
      if (P.rlo[i] > -INF && P.rlo[i] < 1e15) {
        row_has_lo[i] = 1;
        rlo_int[i] = static_cast<std::int64_t>(std::ceil(P.rlo[i] - 1e-9));
      }
      if (P.rup[i] < INF && P.rup[i] > -1e15) {
        row_has_up[i] = 1;
        rup_int[i] = static_cast<std::int64_t>(std::floor(P.rup[i] + 1e-9));
      }
      if (!row_has_lo[i] && !row_has_up[i]) prop_row[i] = 0;
    }
    pr_ptr.assign(m + 1, 0);
    pk_ptr.assign(nk + 1, 0);
    for (int k = 0; k < nk; ++k) {
      const int j = cols[k];
      for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
        const int i = P.cidx[q];
        if (!prop_row[i] || P.cval[q] == 0.0) continue;
        const auto a = static_cast<std::int64_t>(P.cval[q]);
        ++pr_ptr[i + 1];
        ++pk_ptr[k + 1];
        init_max_act[i] += a;
        if (a > row_max_coef[i]) row_max_coef[i] = a;
      }
    }
    for (int i = 0; i < m; ++i) pr_ptr[i + 1] += pr_ptr[i];
    for (int k = 0; k < nk; ++k) pk_ptr[k + 1] += pk_ptr[k];
    pr_col.resize(pr_ptr[m]);
    pr_val.resize(pr_ptr[m]);
    pk_row.resize(pk_ptr[nk]);
    pk_val.resize(pk_ptr[nk]);
    std::vector<int> rpos(pr_ptr.begin(), pr_ptr.end() - 1);
    std::vector<int> kpos(pk_ptr.begin(), pk_ptr.end() - 1);
    for (int k = 0; k < nk; ++k) {
      const int j = cols[k];
      for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
        const int i = P.cidx[q];
        if (!prop_row[i] || P.cval[q] == 0.0) continue;
        const auto a = static_cast<std::int64_t>(P.cval[q]);
        const int rp = rpos[i]++;
        pr_col[rp] = k;
        pr_val[rp] = a;
        const int kp = kpos[k]++;
        pk_row[kp] = i;
        pk_val[kp] = a;
      }
    }
    has_prop = !pr_col.empty();
    base_rows = m;
    base_row_int.assign(m, 1);
    for (int k = 0; k < nk; ++k) {
      const int j = cols[k];
      nnz0 += P.cptr[j + 1] - P.cptr[j];
      for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q)
        if (std::floor(P.cval[q]) != P.cval[q]) base_row_int[P.cidx[q]] = 0;
    }
    if (o.cuts & 7) set_rows = std::make_unique<const SetRows>(P, cols);
    root_gfix.assign(nk, -1);
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
    const bool may_parallel = H.team != nullptr || o.threads > 1;

    // Concurrent first-incumbent sprint BEFORE root cuts (P6): when no incumbent exists yet,
    // solve the clean pre-cut LP once and launch diving + objective/conflict-aware FP + exact cover
    // in parallel so a feasible solution is found quickly (low Primal Integral) and LNS workers
    // can immediately start improving it even while root cuts run.
    if (H.thr() == INF) {
      if (lp0->solve(INF) == LpSolver::OPTIMAL) {
        const double sprint_frac = exact ? std::min(0.04, 0.6 / std::max(1.0, o.time_limit))
                                         : std::min(0.04, 0.7 / std::max(1.0, o.time_limit));
        if (may_parallel) {
          template_lp = lp0->clone(o.seed, time_left, &abort_lp);
          heur_role.store(0, std::memory_order_relaxed);
          sprint_done.store(false, std::memory_order_release);
          auto sprint_helper = [&](const std::function<bool()>& stay) {
            if (sprint_done.load(std::memory_order_acquire) || H.thr() < INF || stop()) return;
            std::unique_ptr<LpSolver> wlp;
            const int role = heur_role.fetch_add(1, std::memory_order_relaxed) + 1;
            {
              std::lock_guard<std::mutex> lk(clone_mu);
              if (sprint_done.load(std::memory_order_relaxed) || H.thr() < INF || !template_lp) return;
              wlp = template_lp->clone(o.seed + 7919u * static_cast<unsigned>(role),
                                       std::max(0.0, time_left - now()), &abort_lp);
            }
            Worker W(*this, role, std::move(wlp));
            auto keep_going = [&]() {
              return !sprint_done.load(std::memory_order_relaxed) && H.thr() == INF && (!stay || stay());
            };
            if (role == 1) W.feasibility_pump(sprint_frac * 1.25, keep_going);
            else if (role == 2 && pure_sp) W.cover_search(sprint_frac, keep_going);
            else W.dive_for_incumbent(8, sprint_frac, role, keep_going);
            std::lock_guard<std::mutex> lk(tree_mu);
            R.lp_iters += W.lp->iterations();
          };
          std::vector<std::thread> sprint_pool;
          if (H.team) H.team->activate(sprint_helper);
          for (int t = 1; t < o.threads; ++t) sprint_pool.emplace_back([&]() { sprint_helper(nullptr); });
          const double sprint_end = now() + sprint_frac * o.time_limit;
          {
            Worker W0(*this, 0, std::move(lp0));
            W0.dive_for_incumbent(8, sprint_frac, 0, nullptr);
            if (H.thr() == INF && !stop() && now() < sprint_end) {
              W0.dive_for_incumbent(12, std::max(0.0, (sprint_end - now()) / std::max(1e-6, o.time_limit)), 17, nullptr);
            }
            lp0 = std::move(W0.lp);
          }
          if (H.thr() < INF || stop() || now() >= sprint_end) {
            sprint_done.store(true, std::memory_order_release);
          }
          for (auto& th : sprint_pool) th.join();
          sprint_done.store(true, std::memory_order_release);
          if (H.team) H.team->deactivate();
          template_lp.reset();
        } else {
          Worker W0(*this, 0, std::move(lp0));
          W0.dive_for_incumbent(1, sprint_frac, 0, nullptr);
          if (H.thr() == INF && pure_sp && !stop()) W0.cover_search(sprint_frac, nullptr);
          if (H.thr() == INF && !stop()) W0.feasibility_pump(sprint_frac, nullptr);
          lp0 = std::move(W0.lp);
        }
        if (stop()) {
          R.lp_iters += lp0->iterations();
          R.dives += a_dives.load(std::memory_order_relaxed);
          R.dive_nodes += a_dive_nodes.load(std::memory_order_relaxed);
          R.pump_iters += a_pump_iters.load(std::memory_order_relaxed);
          R.pump_found += a_pump_found.load(std::memory_order_relaxed);
          R.cover_nodes += a_cover_nodes.load(std::memory_order_relaxed);
          R.nodes = total_nodes.load(std::memory_order_relaxed);
          return STOPPED;
        }
        if (!exact && H.thr() < INF) {
          R.lp_iters += lp0->iterations();
          R.dives += a_dives.load(std::memory_order_relaxed);
          R.dive_nodes += a_dive_nodes.load(std::memory_order_relaxed);
          R.pump_iters += a_pump_iters.load(std::memory_order_relaxed);
          R.pump_found += a_pump_found.load(std::memory_order_relaxed);
          R.cover_nodes += a_cover_nodes.load(std::memory_order_relaxed);
          R.nodes = total_nodes.load(std::memory_order_relaxed);
          return SWITCH;
        }
      }
    }

    if (o.cuts && exact && !root_cuts()) {
      R.gave_up = true;
      R.lp_iters += lp0->iterations();
      return STOPPED;
    }
    if (exact && has_prop && nk > 0 && !stop()) {
      if (!root_probing()) {
        R.lp_iters += lp0 ? lp0->iterations() : 0;
        return EXHAUSTED;
      }
    }
    if (exact && nk > 8000 && !stop()) {
      sift_target = std::min<int>(nk, std::max<int>(4 * P.m, 4000));
      if (sift_target < nk - 500 && lp0->solve(INF) == LpSolver::OPTIMAL) {
        std::vector<double> x0;
        lp0->get_x(x0);
        std::vector<char> iw(nk, 0);
        int cnt = 0;
        for (int k = 0; k < nk; ++k) {
          if (lp0->is_col_basic(k) || x0[k] > 1e-9 || root_gfix[k] == 1) {
            iw[k] = 1;
            ++cnt;
          }
        }
        std::vector<int> ord(nk);
        for (int k = 0; k < nk; ++k) ord[k] = k;
        std::sort(ord.begin(), ord.end(),
                  [&](int a, int b) { return dk[a] < dk[b] || (dk[a] == dk[b] && a < b); });
        for (int k : ord) {
          if (cnt >= sift_target) break;
          if (!iw[k] && root_gfix[k] != 0) {
            iw[k] = 1;
            ++cnt;
          }
        }
        lp0->set_working_cols(iw);
        sifting_tree = true;
      }
    }
    // Prepare template LP for parallel helpers right after root cuts, probing, and working set setup.
    if (may_parallel) {
      template_lp = lp0->clone(o.seed, time_left, &abort_lp);
    }
    push_locked(new Node());

    // Build main worker (slot 0) taking ownership of lp0.
    Worker main_w(*this, 0, std::move(lp0));

    // If still no incumbent after the initial sprint: on a restricted column set (!exact), give B&B
    // only a short budget before widening the column set (NO_INCUMBENT); on exact, run fallback FP/dives later.
    heur_at = H.thr() == INF ? total_nodes.load(std::memory_order_relaxed) + (exact ? 3000 : 300) : -1;
    heur_time = now() + (exact ? 0.1 * o.time_limit : std::min(0.4, 0.025 * o.time_limit));

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
    R.dives += a_dives.load(std::memory_order_relaxed);
    R.dive_nodes += a_dive_nodes.load(std::memory_order_relaxed);
    R.pump_iters += a_pump_iters.load(std::memory_order_relaxed);
    R.pump_found += a_pump_found.load(std::memory_order_relaxed);
    R.cover_nodes += a_cover_nodes.load(std::memory_order_relaxed);
    R.node_cuts += a_node_cuts.load(std::memory_order_relaxed);
    R.node_cut_rounds += a_node_cut_rounds.load(std::memory_order_relaxed);
    R.learned_clauses += static_cast<long>(conf_clauses.size());
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
  std::atomic<bool> sprint_done{false};
  std::atomic<int> heur_role{0};
  std::atomic<long> total_nodes{0};
  std::atomic<long> a_dives{0}, a_dive_nodes{0}, a_pump_iters{0}, a_pump_found{0}, a_cover_nodes{0};
  std::atomic<long> a_node_cuts{0}, a_node_cut_rounds{0}, a_node_cut_ineff{0}, a_node_cut_eff{0};
  std::atomic<bool> node_cuts_off{false};
  int base_rows = 0, sift_target = 0;
  bool sifting_tree = false;
  long nnz0 = 0;
  std::vector<char> base_row_int;
  std::unique_ptr<const SetRows> set_rows;
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

  // Shared ConflictPool & root probing global fixings (Savelsbergh 1994; Atamturk et al. 2000; Achterberg 2007).
  struct ConfClause {
    std::array<int, 6> lits{};
    int len = 0;  // 1..6
  };
  std::vector<std::int8_t> root_gfix;
  std::mutex conf_mu;
  std::atomic<long> conf_version{0};
  std::vector<ConfClause> conf_clauses;
  std::set<std::uint64_t> seen_bin;
  std::set<std::array<int, 6>> seen_long;
  long bin_clause_cnt = 0, long_clause_cnt = 0;

  void add_conflict_clause(std::vector<int> lits) {
    std::sort(lits.begin(), lits.end());
    lits.erase(std::unique(lits.begin(), lits.end()), lits.end());
    if (lits.empty() || lits.size() > 6) return;
    for (size_t i = 0; i + 1 < lits.size(); ++i)
      if ((lits[i] ^ 1) == lits[i + 1]) return;  // tautology
    std::lock_guard<std::mutex> lk(conf_mu);
    if (lits.size() == 1) {
      ConfClause cc;
      cc.lits[0] = lits[0];
      cc.len = 1;
      conf_clauses.push_back(cc);
      conf_version.store(static_cast<long>(conf_clauses.size()), std::memory_order_release);
      return;
    }
    if (lits.size() == 2) {
      if (bin_clause_cnt >= 4096) return;
      const std::uint64_t key =
          (static_cast<std::uint64_t>(lits[0]) << 32) | static_cast<std::uint32_t>(lits[1]);
      if (!seen_bin.insert(key).second) return;
      ++bin_clause_cnt;
      ConfClause cc;
      cc.lits[0] = lits[0];
      cc.lits[1] = lits[1];
      cc.len = 2;
      conf_clauses.push_back(cc);
      conf_version.store(static_cast<long>(conf_clauses.size()), std::memory_order_release);
      return;
    }
    if (long_clause_cnt >= 2048) return;
    std::array<int, 6> key;
    key.fill(-1);
    for (size_t i = 0; i < lits.size(); ++i) key[i] = lits[i];
    if (!seen_long.insert(key).second) return;
    ++long_clause_cnt;
    ConfClause cc;
    cc.lits = key;
    cc.len = static_cast<int>(lits.size());
    conf_clauses.push_back(cc);
    conf_version.store(static_cast<long>(conf_clauses.size()), std::memory_order_release);
  }

  // Read-only exact integer nonnegative propagation tables (shared across workers).
  bool has_prop = false;
  std::vector<char> row_has_lo, row_has_up;
  std::vector<std::int64_t> rlo_int, rup_int, row_max_coef, init_max_act;
  std::vector<int> pr_ptr, pr_col, pk_ptr, pk_row;
  std::vector<std::int64_t> pr_val, pk_val;

  struct Worker {
    enum Reason : std::uint8_t {
      R_NONE = 0,
      R_DECISION = 1,
      R_GLOBAL = 2,
      R_LOCAL = 3,
      R_ROW_UP = 4,
      R_ROW_LO = 5,
      R_BIN = 6,
      R_CLAUSE = 7
    };

    Search& S;
    int wid;
    std::unique_ptr<LpSolver> lp;
    std::vector<int> fv, curfix, want, lfixed, pfixed, dirty_rows, prop_q, rd, bfs_mark, bfs_q, learned;
    std::vector<std::int8_t> gfix, eff;
    std::vector<char> lfix, pfix, rmark, row_in_q;
    std::vector<std::uint8_t> rk;
    std::vector<std::int64_t> min_act, max_act;
    std::vector<double> x, dnode, act, pairw;
    std::vector<int> order, chosen, touched_pairs;
    double last_thr = INF;
    long sb_iters = 0, local_strong = 0, local_fixed = 0;
    long local_conf_ver = 0;
    int bfs_tick = 0;
    bool has_clauses = false, had_thr_gfix = false, had_rc_lfix = false;
    std::vector<std::vector<int>> bin_imp, watch_false;
    std::vector<ConfClause> long_clauses;
    std::mt19937 cut_rng;
    std::vector<char> row_int;
    std::vector<int> local_cut_idle, local_cut_len, pending_drop_cuts, cut_ord;
    std::vector<double> cut_cdense, cut_sel_norm;
    long local_cut_nnz = 0, aged_at_node = 0;
    double local_cut_time = 0.0;
    int last_cut_depth = -1;
    std::vector<int> sift_neg;

    Worker(Search& S_, int wid_, std::unique_ptr<LpSolver> lp_)
        : S(S_), wid(wid_), lp(std::move(lp_)), fv(S_.nk, -1), want(S_.nk, 0), rd(S_.nk, 0), bfs_mark(S_.nk, 0),
          gfix(S_.root_gfix), eff(S_.nk, -1), lfix(S_.nk, 0), pfix(S_.nk, 0), row_in_q(S_.P.m, 0),
          rk(S_.nk, R_NONE), min_act(S_.P.m, 0), max_act(S_.init_max_act), act(S_.P.m, 0.0),
          cut_rng(S_.o.seed * 1315423911u + static_cast<unsigned>(wid_) * 2654435761u + 17u),
          row_int(S_.base_row_int), cut_cdense(S_.nk, 0.0) {
      if (S.partition) {
        rmark.assign(S.P.m, 0);
        pairw.assign(static_cast<size_t>(S.P.m) * S.P.m, 0.0);
      }
      sync_conflicts();
      for (int k = 0; k < S.nk; ++k) {
        if (gfix[k] != -1) set_eff(k, gfix[k], R_GLOBAL, 0);
      }
      if (S.has_prop) {
        for (int i = 0; i < S.P.m; ++i) {
          if (S.row_has_lo[i] && max_act[i] - S.rlo_int[i] < S.row_max_coef[i]) {
            row_in_q[i] = 1;
            dirty_rows.push_back(i);
          }
        }
      }
    }

    void sync_conflicts() {
      const long ver = S.conf_version.load(std::memory_order_acquire);
      if (ver == local_conf_ver) return;
      std::vector<ConfClause> batch;
      {
        std::lock_guard<std::mutex> lk(S.conf_mu);
        if (local_conf_ver < static_cast<long>(S.conf_clauses.size())) {
          batch.assign(S.conf_clauses.begin() + local_conf_ver, S.conf_clauses.end());
          local_conf_ver = static_cast<long>(S.conf_clauses.size());
        }
      }
      for (const auto& cc : batch) {
        if (cc.len == 1) {
          const int k = cc.lits[0] >> 1;
          const std::int8_t v = static_cast<std::int8_t>(cc.lits[0] & 1);
          if (gfix[k] == -1) {
            gfix[k] = v;
            if (eff[k] == -1) set_eff(k, v, R_GLOBAL, 0);
          }
        } else if (cc.len == 2) {
          if (bin_imp.empty()) bin_imp.resize(static_cast<size_t>(2 * S.nk));
          const int a = cc.lits[0], b = cc.lits[1];
          bin_imp[a ^ 1].push_back(b);
          bin_imp[b ^ 1].push_back(a);
          has_clauses = true;
        } else if (cc.len >= 3) {
          if (watch_false.empty()) watch_false.resize(static_cast<size_t>(2 * S.nk));
          const int cid = static_cast<int>(long_clauses.size());
          long_clauses.push_back(cc);
          for (int q = 0; q < cc.len; ++q) watch_false[cc.lits[q] ^ 1].push_back(cid);
          has_clauses = true;
        }
      }
    }

    void clear_dirty() {
      for (int i : dirty_rows) row_in_q[i] = 0;
      dirty_rows.clear();
      prop_q.clear();
    }

    void set_eff(int k, std::int8_t nv, std::uint8_t rkind = R_NONE, int rdata = 0) {
      const std::int8_t ov = eff[k];
      if (ov == nv) return;
      eff[k] = nv;
      if (nv == -1) {
        rk[k] = R_NONE;
        rd[k] = 0;
      } else {
        rk[k] = rkind;
        rd[k] = rdata;
        if (has_clauses) prop_q.push_back((k << 1) | nv);
      }
      lp->set_col_bounds(k, nv == 1 ? 1.0 : 0.0, nv == 0 ? 0.0 : 1.0);
      if (!S.has_prop) return;
      const int beg = S.pk_ptr[k], end = S.pk_ptr[k + 1];
      for (int q = beg; q < end; ++q) {
        const int i = S.pk_row[q];
        const std::int64_t a = S.pk_val[q];
        if (ov == 1) min_act[i] -= a;
        else if (ov == 0) max_act[i] += a;
        if (nv == 1) min_act[i] += a;
        else if (nv == 0) max_act[i] -= a;
        if (row_in_q[i]) continue;
        const bool trig =
            (S.row_has_up[i] && (min_act[i] > S.rup_int[i] ||
                                 (S.rup_int[i] - min_act[i] < S.row_max_coef[i] && max_act[i] > min_act[i]))) ||
            (S.row_has_lo[i] && (max_act[i] < S.rlo_int[i] ||
                                 (min_act[i] < S.rlo_int[i] && max_act[i] - S.rlo_int[i] < S.row_max_coef[i] &&
                                  max_act[i] > min_act[i])));
        if (trig) {
          row_in_q[i] = 1;
          dirty_rows.push_back(i);
        }
      }
    }

    std::int8_t desired_eff(int k, bool& conflict, std::uint8_t* rkind_out = nullptr) const {
      std::int8_t v = static_cast<std::int8_t>(fv[k]);
      std::uint8_t rkind = v != -1 ? R_DECISION : R_NONE;
      if (gfix[k] != -1) {
        if (v != -1 && v != gfix[k]) {
          conflict = true;
          if (rkind_out) *rkind_out = R_GLOBAL;
          return gfix[k];
        }
        v = gfix[k];
        rkind = R_GLOBAL;
      }
      if (lfix[k]) {
        const std::int8_t lv = static_cast<std::int8_t>(lfix[k] - 1);
        if (v != -1 && v != lv) {
          conflict = true;
          if (rkind_out) *rkind_out = rkind;
          return v;
        }
        if (v == -1) rkind = R_LOCAL;
        v = lv;
      }
      if (pfix[k]) {
        const std::int8_t pv = static_cast<std::int8_t>(pfix[k] - 1);
        if (v != -1 && v != pv) {
          conflict = true;
          if (rkind_out) *rkind_out = rkind;
          return v;
        }
        if (v == -1) rkind = R_DECISION;
        v = pv;
      }
      if (rkind_out) *rkind_out = rkind;
      return v;
    }

    bool global_fix(double thr) {
      if (!(thr < last_thr)) return false;
      last_thr = thr;
      bool conflict = false;
      for (int k = 0; k < S.nk; ++k) {
        if (gfix[k] != -1) continue;
        const double dj = S.dk[k];
        if (dj > 0.0 && S.round_up(S.zr0 + dj - 1e-9 * std::max(1.0, std::fabs(S.zr0))) > thr) {
          gfix[k] = 0;
          had_thr_gfix = true;
          if (eff[k] == 1) conflict = true;
          else set_eff(k, 0, R_LOCAL, 0);
        } else if (dj < 0.0 && S.round_up(S.zr0 - dj - 1e-9 * std::max(1.0, std::fabs(S.zr0))) > thr) {
          gfix[k] = 1;
          had_thr_gfix = true;
          if (eff[k] == 0) conflict = true;
          else set_eff(k, 1, R_LOCAL, 0);
        }
      }
      return conflict;
    }

    bool local_fix(double z, double thr) {
      lp->get_red_cost(dnode);
      const double eps = 1e-7 * std::max(1.0, std::fabs(z));
      bool any = false;
      for (int k = 0; k < S.nk; ++k) {
        if (eff[k] != -1) continue;
        const double dk_val = dnode[k];
        if (x[k] < 1e-9 && dk_val > 0.0 && S.round_up(z + dk_val - eps) > thr) {
          lfix[k] = 1;
          lfixed.push_back(k);
          set_eff(k, 0, R_LOCAL, 0);
          ++local_fixed;
          had_rc_lfix = true;
          any = true;
        } else if (x[k] > 1.0 - 1e-9 && dk_val < 0.0 && S.round_up(z - dk_val - eps) > thr) {
          lfix[k] = 2;
          lfixed.push_back(k);
          set_eff(k, 1, R_LOCAL, 0);
          ++local_fixed;
          had_rc_lfix = true;
          any = true;
        }
      }
      return any;
    }

    void undo_local() {
      for (int k : lfixed) {
        lfix[k] = 0;
        bool c = false;
        std::uint8_t rkind = R_NONE;
        const std::int8_t nv = desired_eff(k, c, &rkind);
        set_eff(k, nv, rkind, 0);
      }
      lfixed.clear();
      had_rc_lfix = false;
    }

    void fix_prop(int k, int v, bool to_probe, std::uint8_t rkind, int rdata) {
      if (to_probe) {
        pfix[k] = static_cast<char>(v + 1);
        pfixed.push_back(k);
      } else {
        lfix[k] = static_cast<char>(v + 1);
        lfixed.push_back(k);
        ++local_fixed;
      }
      set_eff(k, static_cast<std::int8_t>(v), rkind, rdata);
    }

    void trace_conflict(const int* seeds, int nseeds) {
      ++bfs_tick;
      bfs_q.clear();
      learned.clear();
      for (int s = 0; s < nseeds; ++s) {
        const int c = seeds[s];
        if (c >= 0 && c < S.nk && bfs_mark[c] != bfs_tick) {
          bfs_mark[c] = bfs_tick;
          bfs_q.push_back(c);
        }
      }
      for (size_t h = 0; h < bfs_q.size(); ++h) {
        if (bfs_q.size() > 64) return;
        const int c = bfs_q[h];
        const std::uint8_t kind = rk[c];
        if (kind == R_LOCAL || kind == R_NONE) return;
        if (kind == R_GLOBAL) continue;
        if (kind == R_DECISION) {
          if (eff[c] != 0 && eff[c] != 1) return;
          learned.push_back((c << 1) | (1 - eff[c]));
          if (learned.size() > 6) return;
          continue;
        }
        const int d = rd[c];
        if (kind == R_BIN) {
          const int p = d >> 1;
          if (bfs_mark[p] != bfs_tick) {
            bfs_mark[p] = bfs_tick;
            bfs_q.push_back(p);
          }
        } else if (kind == R_CLAUSE) {
          const auto& cl = long_clauses[static_cast<size_t>(d)];
          for (int q = 0; q < cl.len; ++q) {
            const int p = cl.lits[q] >> 1;
            if (p != c && bfs_mark[p] != bfs_tick) {
              bfs_mark[p] = bfs_tick;
              bfs_q.push_back(p);
            }
          }
        } else if (kind == R_ROW_UP || kind == R_ROW_LO) {
          const std::int8_t bad_val = (kind == R_ROW_UP) ? 1 : 0;
          const int beg = S.pr_ptr[d], end = S.pr_ptr[d + 1];
          for (int q = beg; q < end; ++q) {
            const int p = S.pr_col[q];
            if (p != c && eff[p] == bad_val && bfs_mark[p] != bfs_tick) {
              bfs_mark[p] = bfs_tick;
              bfs_q.push_back(p);
            }
          }
        }
      }
      if (!learned.empty() && learned.size() <= 6) S.add_conflict_clause(learned);
    }

    void learn_from_row(int i, bool upper_fail) {
      std::vector<int> seeds;
      const std::int8_t bad_val = upper_fail ? 1 : 0;
      for (int q = S.pr_ptr[i]; q < S.pr_ptr[i + 1]; ++q) {
        const int c = S.pr_col[q];
        if (eff[c] == bad_val) {
          seeds.push_back(c);
          if (seeds.size() > 32) return;
        }
      }
      if (!seeds.empty()) trace_conflict(seeds.data(), static_cast<int>(seeds.size()));
    }

    bool propagate(bool to_probe, bool learn = true) {
      if ((!S.has_prop || dirty_rows.empty()) && prop_q.empty()) return true;
      size_t qh = 0, pqh = 0;
      while (qh < dirty_rows.size() || pqh < prop_q.size()) {
        while (pqh < prop_q.size()) {
          const int lit = prop_q[pqh++];
          if (static_cast<size_t>(lit) < bin_imp.size()) {
            const auto& imps = bin_imp[static_cast<size_t>(lit)];
            for (int imp : imps) {
              const int bk = imp >> 1, bv = imp & 1;
              if (eff[bk] == 1 - bv) {
                if (learn) {
                  const int seeds[2] = {lit >> 1, bk};
                  trace_conflict(seeds, 2);
                }
                clear_dirty();
                return false;
              }
              if (eff[bk] == -1) fix_prop(bk, bv, to_probe, R_BIN, lit);
            }
          }
          if (static_cast<size_t>(lit) < watch_false.size()) {
            const auto& ws = watch_false[static_cast<size_t>(lit)];
            for (int cid : ws) {
              const auto& cl = long_clauses[static_cast<size_t>(cid)];
              bool sat = false;
              int undef_cnt = 0, last_undef = -1;
              for (int q = 0; q < cl.len; ++q) {
                const int clit = cl.lits[q];
                const std::int8_t ev = eff[clit >> 1];
                if (ev == (clit & 1)) {
                  sat = true;
                  break;
                }
                if (ev == -1) {
                  ++undef_cnt;
                  last_undef = clit;
                }
              }
              if (sat) continue;
              if (undef_cnt == 0) {
                if (learn) {
                  int seeds[6];
                  for (int q = 0; q < cl.len; ++q) seeds[q] = cl.lits[q] >> 1;
                  trace_conflict(seeds, cl.len);
                }
                clear_dirty();
                return false;
              }
              if (undef_cnt == 1) fix_prop(last_undef >> 1, last_undef & 1, to_probe, R_CLAUSE, cid);
            }
          }
        }
        if (qh < dirty_rows.size()) {
          const int i = dirty_rows[qh++];
          row_in_q[i] = 0;
          if (S.row_has_up[i]) {
            if (min_act[i] > S.rup_int[i]) {
              if (learn) learn_from_row(i, true);
              clear_dirty();
              return false;
            }
            const std::int64_t rem_up = S.rup_int[i] - min_act[i];
            if (rem_up < S.row_max_coef[i] && max_act[i] > min_act[i]) {
              const int beg = S.pr_ptr[i], end = S.pr_ptr[i + 1];
              for (int q = beg; q < end; ++q) {
                const int k = S.pr_col[q];
                if (eff[k] == -1 && S.pr_val[q] > rem_up) fix_prop(k, 0, to_probe, R_ROW_UP, i);
              }
            }
          }
          if (S.row_has_lo[i]) {
            if (max_act[i] < S.rlo_int[i]) {
              if (learn) learn_from_row(i, false);
              clear_dirty();
              return false;
            }
            const std::int64_t slack_lo = max_act[i] - S.rlo_int[i];
            if (min_act[i] < S.rlo_int[i] && slack_lo < S.row_max_coef[i] && max_act[i] > min_act[i]) {
              const int beg = S.pr_ptr[i], end = S.pr_ptr[i + 1];
              for (int q = beg; q < end; ++q) {
                const int k = S.pr_col[q];
                if (eff[k] == -1 && S.pr_val[q] > slack_lo) fix_prop(k, 1, to_probe, R_ROW_LO, i);
              }
            }
          }
        }
      }
      dirty_rows.clear();
      prop_q.clear();
      return true;
    }

    bool start_probe(int k, int v, bool learn = true) {
      if (eff[k] != -1) return eff[k] == v;
      pfix[k] = static_cast<char>(v + 1);
      pfixed.push_back(k);
      set_eff(k, static_cast<std::int8_t>(v), R_DECISION, 0);
      return propagate(true, learn);
    }

    void undo_probe() {
      for (int k : pfixed) {
        pfix[k] = 0;
        bool c = false;
        std::uint8_t rkind = R_NONE;
        const std::int8_t nv = desired_eff(k, c, &rkind);
        set_eff(k, nv, rkind, 0);
      }
      pfixed.clear();
      clear_dirty();
    }

    bool switch_to(const std::vector<int>& fix) {
      for (int e : fix) want[e >> 1] = (e & 1) + 1;
      for (int e : curfix) {
        const int k = e >> 1;
        if (want[k] - 1 != fv[k]) {
          fv[k] = -1;
          bool c = false;
          std::uint8_t rkind = R_NONE;
          const std::int8_t nv = desired_eff(k, c, &rkind);
          set_eff(k, nv, rkind, 0);
        }
      }
      bool ok = true;
      for (int e : fix) {
        const int k = e >> 1, v = e & 1;
        want[k] = 0;
        if (fv[k] != v) fv[k] = v;
        bool c = false;
        std::uint8_t rkind = R_NONE;
        const std::int8_t nv = desired_eff(k, c, &rkind);
        if (c) ok = false;
        else set_eff(k, nv, rkind, 0);
      }
      curfix = fix;
      return ok;
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
      if (!S.o.node_basis || lp->rows() != S.base_rows || static_cast<double>(osz + 1) * (S.P.m + S.nk) > 4e8)
        return nullptr;
      return std::make_shared<const std::vector<char>>(lp->get_basis());
    }

    bool ensure_priced(LpSolver::Status& st, double cut_val, double thr, double& z) {
      if (!lp->has_inactive_cols()) {
        if (st == LpSolver::OPTIMAL) z = lp->obj() - S.P.offset;
        return true;
      }
      for (int round = 0; round < 24; ++round) {
        if (st == LpSolver::INFEASIBLE) {
          lp->activate_all_unfixed();
          st = lp->solve(cut_val);
          if (st == LpSolver::OPTIMAL) z = lp->obj() - S.P.offset;
          return true;
        }
        if (st != LpSolver::OPTIMAL && st != LpSolver::CUTOFF) return false;
        sift_neg.clear();
        if (!lp->price_inactive(sift_neg, 1e-7)) {
          st = lp->solve(cut_val);
          continue;
        }
        if (sift_neg.empty()) {
          if (st == LpSolver::CUTOFF) {
            const double zn = lp->obj() - S.P.offset;
            if (S.round_up(zn) <= thr) {
              st = lp->solve(cut_val);
              continue;
            }
          } else {
            z = lp->obj() - S.P.offset;
          }
          return true;
        }
        const int cap = std::min<int>(static_cast<int>(sift_neg.size()), std::max(2 * S.P.m, 500));
        sift_neg.resize(static_cast<size_t>(cap));
        lp->activate_cols(sift_neg);
        st = lp->solve(cut_val);
      }
      lp->activate_all_unfixed();
      st = lp->solve(cut_val);
      if (st == LpSolver::OPTIMAL) z = lp->obj() - S.P.offset;
      return true;
    }

    void maybe_age_cols(long nodes_now) {
      if (!S.sifting_tree || nodes_now - aged_at_node < 400) return;
      if (lp->active_cols_count() <= 2 * S.sift_target) return;
      aged_at_node = nodes_now;
      lp->get_red_cost(dnode);
      std::vector<char> keep(S.nk, 0);
      int kept = 0;
      std::vector<int> cands;
      for (int k = 0; k < S.nk; ++k) {
        if (lp->is_col_basic(k) || x[k] > 1e-9 || eff[k] == 1) {
          keep[k] = 1;
          ++kept;
        } else if (eff[k] == -1) {
          cands.push_back(k);
        }
      }
      if (kept < S.sift_target && !cands.empty()) {
        const int need = std::min<int>(S.sift_target - kept, static_cast<int>(cands.size()));
        std::nth_element(cands.begin(), cands.begin() + need, cands.end(),
                         [&](int a, int b) { return dnode[a] < dnode[b] || (dnode[a] == dnode[b] && a < b); });
        for (int i = 0; i < need; ++i) keep[cands[i]] = 1;
      }
      lp->set_working_cols(keep);
    }

    void clear_local_cuts() {
      if (lp->rows() > S.base_rows) lp->remove_rows(S.base_rows);
      row_int = S.base_row_int;
      local_cut_idle.clear();
      local_cut_len.clear();
      pending_drop_cuts.clear();
      local_cut_nnz = 0;
      last_cut_depth = -1;
    }

    void drop_pending_cuts() {
      if (pending_drop_cuts.empty()) return;
      lp->remove_rows(pending_drop_cuts);
      for (size_t a = pending_drop_cuts.size(); a-- > 0;) {
        const int r = pending_drop_cuts[a];
        const int idx = r - S.base_rows;
        local_cut_nnz -= local_cut_len[idx];
        local_cut_len.erase(local_cut_len.begin() + idx);
        local_cut_idle.erase(local_cut_idle.begin() + idx);
        row_int.erase(row_int.begin() + r);
      }
      pending_drop_cuts.clear();
    }

    void mark_idle_cuts() {
      pending_drop_cuts.clear();
      const int nloc = lp->rows() - S.base_rows;
      if (nloc <= 0) return;
      const std::vector<char> rs = lp->row_status();
      for (int idx = 0; idx < nloc; ++idx) {
        if (rs[S.base_rows + idx] == 0) {
          if (++local_cut_idle[idx] >= 2) pending_drop_cuts.push_back(S.base_rows + idx);
        } else {
          local_cut_idle[idx] = 0;
        }
      }
    }

    // Shallow-depth local GMI + zero-half/clique cut round with automatic rollback when ineffective.
    // Returns -1 if the cuts proved infeasibility/cutoff, 1 if z improved, 0 otherwise.
    int try_node_cuts(int depth, double& z, double thr, long nodes_now, bool was_from_queue) {
      if (!S.exact || S.o.cuts == 0 || lp->has_inactive_cols() || depth < 1 || depth > 4 ||
          (!was_from_queue && depth > 2) || depth <= last_cut_depth ||
          S.node_cuts_off.load(std::memory_order_relaxed))
        return 0;
      if (S.R.cuts == 0 && (S.o.cuts & 6) == 0 && S.P.m >= 400) return 0;
      if (lp->rows() - S.base_rows >= 8) return 0;
      const long nnz_cap = std::min<long>(std::max<long>(S.nnz0 / 8, 800L), 2500L);
      if (local_cut_nnz >= nnz_cap) return 0;
      if (local_cut_time >= 0.05 * S.now() + 0.01) return 0;
      if (S.nnz0 > 120000 && (!was_from_queue || depth > 2 || S.a_node_cut_rounds.load(std::memory_order_relaxed) >= 12))
        return 0;
      if (depth >= 3 && nodes_now >= 32 && (nodes_now & 7) != 0) return 0;
      if (depth <= 2 && nodes_now >= 32 && !was_from_queue && (nodes_now & 3) != 0) return 0;
      last_cut_depth = depth;
      const double t0_cut = S.now();
      CutPool pool;
      const long avg_row = std::max<long>(1L, S.nnz0 / std::max(1, S.P.m));
      const long max_len = std::min<long>(S.nk, std::max<long>(80L, 6L * avg_row));
      if (S.o.cuts & 8) separate_gmi(*lp, row_int, x, 0.02, 5, static_cast<int>(max_len), pool);
      if ((S.o.cuts & 7) && S.set_rows) {
        if (S.o.cuts & 4) separate_cliques(*S.set_rows, x, 0.03, 5, pool);
        if (S.o.cuts & 1) separate_zero_half(*S.set_rows, x, true, std::min(S.o.cut_size, 11), 0.03, 5, cut_rng, pool);
        if (S.o.cuts & 2) separate_zero_half(*S.set_rows, x, false, std::min(S.o.cut_size, 11), 0.03, 5, cut_rng, pool);
      }
      std::vector<int> rptr, ridx, sel_len;
      std::vector<double> rval, rlo, rup;
      std::vector<char> sel_int;
      long added_nnz = 0;
      if (!select_cuts_into(pool, 6, max_len, nnz_cap - local_cut_nnz, 0.90, rptr, ridx, rval, rlo, rup, sel_len,
                            sel_int, added_nnz, cut_ord, cut_cdense, cut_sel_norm)) {
        local_cut_time += S.now() - t0_cut;
        return 0;
      }
      LpSolver::State pre_cut_state;
      lp->save_state(pre_cut_state);
      const int rows_before = lp->rows();
      lp->add_rows(rptr, ridx, rval, rlo, rup);
      S.a_node_cut_rounds.fetch_add(1, std::memory_order_relaxed);
      const double cut_val = thr == INF ? INF : thr + S.P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
      const LpSolver::Status st = lp->solve(cut_val);
      if (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF) {
        local_cut_time += S.now() - t0_cut;
        S.a_node_cut_eff.fetch_add(1, std::memory_order_relaxed);
        return -1;
      }
      if (st != LpSolver::OPTIMAL) {
        lp->remove_rows(rows_before);
        lp->restore_state(pre_cut_state);
        local_cut_time += S.now() - t0_cut;
        return 0;
      }
      const double zn = lp->obj() - S.P.offset;
      if (S.round_up(zn) > thr) {
        local_cut_time += S.now() - t0_cut;
        S.a_node_cut_eff.fetch_add(1, std::memory_order_relaxed);
        return -1;
      }
      const bool useful = (thr < INF)
                              ? (zn - z >= 0.02 * std::max(1e-9, thr - z) ||
                                 (S.o.integral && S.round_up(zn) > S.round_up(z)))
                              : (zn - z >= 1e-3 * std::max(1.0, std::fabs(z)));
      if (!useful) {
        lp->remove_rows(rows_before);
        lp->restore_state(pre_cut_state);
        const long ineff = S.a_node_cut_ineff.fetch_add(1, std::memory_order_relaxed) + 1;
        const long eff_cnt = S.a_node_cut_eff.load(std::memory_order_relaxed);
        if (ineff >= 6 && eff_cnt * 2 < ineff) S.node_cuts_off.store(true, std::memory_order_relaxed);
        local_cut_time += S.now() - t0_cut;
        return 0;
      }
      local_cut_nnz += added_nnz;
      local_cut_len.insert(local_cut_len.end(), sel_len.begin(), sel_len.end());
      local_cut_idle.insert(local_cut_idle.end(), sel_len.size(), 0);
      row_int.insert(row_int.end(), sel_int.begin(), sel_int.end());
      S.a_node_cuts.fetch_add(static_cast<long>(sel_len.size()), std::memory_order_relaxed);
      S.a_node_cut_eff.fetch_add(1, std::memory_order_relaxed);
      z = zn;
      lp->get_x(x);
      local_cut_time += S.now() - t0_cut;
      return 1;
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

    // Exact cover search for pure partitioning problems, without LPs (safe to run concurrently).
    bool cover_search(double time_frac, const std::function<bool()>& stay = nullptr) {
      const int m = S.P.m;
      std::vector<int> rptr(m + 1, 0), rcol;
      for (int k = 0; k < S.nk; ++k)
        if (gfix[k] != 0)
          for (int q = S.P.cptr[S.cols[k]]; q < S.P.cptr[S.cols[k] + 1]; ++q) ++rptr[S.P.cidx[q] + 1];
      for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
      rcol.resize(rptr[m]);
      {
        std::vector<int> pos(rptr.begin(), rptr.end() - 1);
        for (int k = 0; k < S.nk; ++k)
          if (gfix[k] != 0)
            for (int q = S.P.cptr[S.cols[k]]; q < S.P.cptr[S.cols[k] + 1]; ++q) rcol[pos[S.P.cidx[q]]++] = k;
      }
      std::vector<double> key(S.nk);
      std::vector<int> ban(S.nk, 0), avail(m), chosen_k;
      std::vector<char> cov(m, 0);
      std::mt19937 rng(S.o.seed * 69069u + 4242u + static_cast<unsigned>(wid) * 1009u);
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
      for (int round = 0; !found && S.H.thr() == INF; ++round, limit *= 2) {
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
          if ((nodes & 255) == 0 && (S.stop() || S.terminated.load(std::memory_order_relaxed) ||
                                     S.H.thr() < INF || (stay && !stay()) || S.now() - t0 > time_frac * S.o.time_limit))
            return !S.stop();
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
          S.a_cover_nodes.fetch_add(1, std::memory_order_relaxed);
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

    // Objective & conflict-aware feasibility pump (Fischetti, Glover & Lodi 2005; Achterberg & Berthold 2007).
    bool feasibility_pump(double time_frac, const std::function<bool()>& stay = nullptr) {
      switch_to({});
      if (lp->solve() != LpSolver::OPTIMAL) return !S.stop();
      ++S.total_nodes;
      lp->get_x(x);
      std::mt19937 rng(S.o.seed * 40503u + 977u + static_cast<unsigned>(wid) * 3571u);
      bool has_pack = false;
      std::vector<char> is_pack(S.P.m, 0);
      for (int i = 0; i < S.P.m; ++i) {
        if (S.row_has_up[i] && S.rup_int[i] == 1 && S.row_max_coef[i] == 1) {
          is_pack[i] = 1;
          has_pack = true;
        }
      }
      std::vector<int> rord;
      std::vector<char> rused(S.P.m, 0);
      auto round_xt = [&](std::vector<char>& xt) {
        if (!has_pack) {
          for (int k = 0; k < S.nk; ++k) xt[k] = x[k] > 0.5;
          return;
        }
        std::fill(xt.begin(), xt.end(), 0);
        std::fill(rused.begin(), rused.end(), 0);
        rord.clear();
        for (int k = 0; k < S.nk; ++k)
          if (x[k] > 1e-6 && gfix[k] != 0) rord.push_back(k);
        std::sort(rord.begin(), rord.end(), [&](int a, int b) {
          if (x[a] != x[b]) return x[a] > x[b];
          return S.P.cost[S.cols[a]] < S.P.cost[S.cols[b]];
        });
        for (int k : rord) {
          bool ok_k = true;
          const int beg = S.pk_ptr[k], end = S.pk_ptr[k + 1];
          for (int q = beg; q < end; ++q) {
            const int i = S.pk_row[q];
            if (is_pack[i] && rused[i]) {
              ok_k = false;
              break;
            }
          }
          if (!ok_k) continue;
          xt[k] = 1;
          for (int q = beg; q < end; ++q) {
            const int i = S.pk_row[q];
            if (is_pack[i]) rused[i] = 1;
          }
        }
      };

      std::vector<char> xt(S.nk), prev1, prev2;
      round_xt(xt);
      double cmax = 0.0;
      for (double c : S.P.cost) cmax = std::max(cmax, std::fabs(c));
      if (cmax == 0.0) cmax = 1.0;
      const double t0 = S.now();
      const double budget = time_frac * S.o.time_limit;
      std::vector<std::pair<double, int>> sc;
      bool ok = true, found = false;
      double alpha = 0.5;
      for (int it = 0; it < 300 && !found; ++it) {
        if (S.stop()) {
          ok = false;
          break;
        }
        if (S.terminated.load(std::memory_order_relaxed) || (stay && !stay()) || S.now() - t0 > budget ||
            S.H.thr() < INF)
          break;
        const double l1_w = (1.0 - alpha) * cmax;
        for (int k = 0; k < S.nk; ++k) lp->set_col_cost(k, (xt[k] ? -l1_w : l1_w) + alpha * S.P.cost[S.cols[k]]);
        alpha *= 0.85;
        if (alpha < 1e-3) alpha = 0.0;
        if (lp->solve() != LpSolver::OPTIMAL) break;
        S.a_pump_iters.fetch_add(1, std::memory_order_relaxed);
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
        std::vector<char> next_xt(S.nk);
        round_xt(next_xt);
        if (next_xt != xt && next_xt != prev1 && next_xt != prev2) {
          prev2 = prev1;
          prev1 = xt;
          xt.swap(next_xt);
          continue;
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
        S.a_pump_found.fetch_add(1, std::memory_order_relaxed);
      }
      return ok;
    }

    // Propagation-accelerated restarted randomized diving while there is no incumbent.
    bool dive_for_incumbent(int max_dives, double time_frac, int seed_offset = 0,
                            const std::function<bool()>& stay = nullptr) {
      std::mt19937 rng(S.o.seed * 2654435761u + 12345u + static_cast<unsigned>(seed_offset) * 1000003u);
      std::vector<int> fix, cand;
      const double t0 = S.now();
      const double budget = time_frac * S.o.time_limit;
      for (int dive = 0; dive < max_dives && S.H.thr() == INF; ++dive) {
        fix.clear();
        int flips = 0;
        const int max_flips = 10 + 5 * std::min(dive, 40);
        for (;;) {
          if (S.stop()) {
            undo_probe();
            switch_to({});
            return false;
          }
          if (S.terminated.load(std::memory_order_relaxed) || S.H.thr() < INF || (stay && !stay()) ||
              S.now() - t0 > budget) {
            undo_probe();
            switch_to({});
            return true;
          }
          undo_probe();
          const bool sw_ok = switch_to(fix) && propagate(true);
          const LpSolver::Status st = sw_ok ? lp->solve() : LpSolver::INFEASIBLE;
          ++S.total_nodes;
          S.a_dive_nodes.fetch_add(1, std::memory_order_relaxed);
          if (st != LpSolver::OPTIMAL) {
            if (st != LpSolver::INFEASIBLE && S.stop()) {
              undo_probe();
              switch_to({});
              return false;
            }
            while (!fix.empty() && (fix.back() & 1) == 0) fix.pop_back();
            if (fix.empty()) {
              if (dive == 0 && flips == 0) {
                undo_probe();
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
            if (x[k] > FRAC && x[k] < 1.0 - FRAC && eff[k] == -1) cand.push_back(k);
          if (cand.empty()) {
            offer_integral();
            break;
          }
          if (S.nonneg && fix.empty()) round_greedy();
          std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(3, cand.size()), cand.end(),
                            [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
          const unsigned r = (dive == 0 && seed_offset == 0) ? 0 : rng() % 6;
          const size_t pick = std::min<size_t>(r < 3 ? 0 : (r < 5 ? 1 : 2), cand.size() - 1);
          fix.push_back(cand[pick] * 2 + 1);
        }
        S.a_dives.fetch_add(1, std::memory_order_relaxed);
      }
      undo_probe();
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
        if (fv[k] != -1 || gfix[k] != -1) continue;
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

    int branch_var(Node& cur, double z, double thr, std::vector<int>& cand, bool lp_priced) {
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
      const bool large_model = S.nk > 8000 || (S.P.m >= 500 && S.nk > 4000);
      const int sb_cands = large_model ? (cur.depth <= 2 ? 6 : 4) : S.o.strong_cands;
      const long sb_iter_lim = (S.nk > 20000 || S.P.m >= 1000)
                                   ? std::min(S.o.strong_iters, 50L)
                                   : (S.nk > 8000 ? std::min(S.o.strong_iters, 75L) : S.o.strong_iters);
      const bool may_strong = sb_cands > 0 &&
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
      const bool has_inact = lp->has_inactive_cols();
      for (size_t ci = 0; ci < cand.size(); ++ci) {
        const int k = cand[ci];
        const double f = x0[k];
        double d0, d1;
        const bool unreliable = cpc[ci].c0 < S.o.reliability || cpc[ci].c1 < S.o.reliability;
        if (may_strong && unreliable && tried < sb_cands && since_best < S.o.strong_lookahead) {
          ++tried;
          ++since_best;
          if (!saved) {
            lp->save_state(node_state);
            saved = true;
          }
          double zc[2];
          for (int v = 0; v < 2; ++v) {
            const long it0 = lp->iterations();
            const bool prop_ok = start_probe(k, v);
            if (!prop_ok) {
              zc[v] = INF;
              undo_probe();
              lp->restore_state(node_state);
              continue;
            }
            const LpSolver::Status st = lp->solve(cut, sb_iter_lim);
            ++local_strong;
            if (!has_inact && st == LpSolver::INFEASIBLE && S.exact && !had_thr_gfix && !had_rc_lfix &&
                cur.fix.size() < 6) {
              std::vector<int> cl;
              cl.reserve(cur.fix.size() + 1);
              for (int e : cur.fix) cl.push_back(e ^ 1);
              cl.push_back(((k << 1) | v) ^ 1);
              S.add_conflict_clause(std::move(cl));
            }
            zc[v] = (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF) ? INF
                    : (st == LpSolver::OPTIMAL || st == LpSolver::LIMIT) ? std::max(lp->obj() - S.P.offset, z) : z;
            if (zc[v] < INF && S.round_up(zc[v]) > thr) zc[v] = INF;
            sb_iters += lp->iterations() - it0;
            undo_probe();
            lp->restore_state(node_state);
          }
          if (!has_inact) {
            if (zc[0] == INF && zc[1] == INF) {
              if (!obs.empty()) S.record_pcs(obs);
              return -2;
            }
            if (zc[0] == INF || zc[1] == INF) {
              if (!obs.empty()) S.record_pcs(obs);
              cur.fix.push_back(k * 2 + (zc[0] == INF ? 1 : 0));
              return -1;
            }
          } else {
            const double cap_z = thr < INF ? std::max(z + 1.0, thr) : z + 100.0;
            if (zc[0] == INF) zc[0] = cap_z;
            if (zc[1] == INF) zc[1] = cap_z;
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
      const double zb = lp_priced ? std::max(z, cur.bound) : cur.bound;
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
              if (!exact) {
                // Restricted column set already ran the concurrent sprint + short tree search without finding
                // a solution: widen the column set immediately instead of stalling on a small pool.
                terminate_locked(NO_INCUMBENT);
                return;
              }
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
          W.clear_local_cuts();
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
        const bool was_from_queue = from_queue;
        W.sync_conflicts();
        if (!W.switch_to(cur.fix) || !W.propagate(false)) break;
        W.drop_pending_cuts();
        if (from_queue && cur.basis) W.lp->set_basis(*cur.basis);
        from_queue = false;
        cur.basis.reset();
        const double cut_val = thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
        LpSolver::Status st = W.lp->solve(cut_val);
        const long nodes_now = ++total_nodes;
        bool lp_priced = !W.lp->has_inactive_cols();
        double z = (st == LpSolver::OPTIMAL) ? (W.lp->obj() - P.offset) : -INF;
        if (!lp_priced && (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF || was_from_queue || cur.depth <= 2 ||
                           (st == LpSolver::OPTIMAL && round_up(z) > thr))) {
          lp_priced = W.ensure_priced(st, cut_val, thr, z);
        }
        if (st == LpSolver::INFEASIBLE) {
          if (exact && !W.had_thr_gfix && !W.had_rc_lfix && !cur.fix.empty() && cur.fix.size() <= 6) {
            std::vector<int> cl;
            cl.reserve(cur.fix.size());
            for (int e : cur.fix) cl.push_back(e ^ 1);
            add_conflict_clause(std::move(cl));
          }
          break;
        }
        if (st == LpSolver::CUTOFF) break;
        if (st != LpSolver::OPTIMAL) {
          if (terminated.load(std::memory_order_relaxed) || stop()) {
            if (stop()) terminate(STOPPED);
            break;
          }
          set_incomplete();
          break;
        }
        if (cur.bvar >= 0 && lp_priced) {
          record_pc(cur.bdir, cur.bvar, std::max(z - cur.pz, 0.0) / std::max(cur.bfrac, 1e-9));
          cur.bvar = -1;
        } else if (cur.bvar >= 0) {
          cur.bvar = -1;
        }
        thr = H.thr();
        if (!exact && thr < INF) {
          terminate(SWITCH);
          break;
        }
        if (round_up(z) > thr) {
          if (!lp_priced) {
            const double cv = thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
            lp_priced = W.ensure_priced(st, cv, thr, z);
            if (st != LpSolver::OPTIMAL || round_up(z) > thr) break;
          } else {
            break;
          }
        }
        if (W.global_fix(thr)) break;
        W.lp->get_x(W.x);
        if (H.lp_point && (nodes_now & 15) == 1) H.lp_point(cols, W.x);
        bool tightened = false;
        if (lp_priced && o.node_fixing && exact && thr < INF && W.local_fix(z, thr)) tightened = true;
        if (tightened || !W.dirty_rows.empty()) {
          if (!W.propagate(false)) break;
        }
        if (o.rins && exact && thr < INF && nodes_now >= rins_at.load(std::memory_order_relaxed) &&
            rins_time_sec() < 0.08 * now() + 0.01 * o.time_limit) {
          if (!rins_busy.exchange(true, std::memory_order_acq_rel)) {
            W.rins();
            rins_busy.store(false, std::memory_order_release);
            thr = H.thr();
            if (round_up(z) > thr) {
              if (!lp_priced) {
                const double cv = thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
                lp_priced = W.ensure_priced(st, cv, thr, z);
                if (st != LpSolver::OPTIMAL || round_up(z) > thr) break;
                W.lp->get_x(W.x);
              } else {
                break;
              }
            }
          }
        }
        auto collect_cand = [&]() {
          cand.clear();
          bool nr = false;
          for (int k = 0; k < nk; ++k) {
            if (W.x[k] > FRAC && W.x[k] < 1.0 - FRAC) {
              if (W.eff[k] != -1) nr = true;
              else cand.push_back(k);
            } else if (W.eff[k] != -1 && std::fabs(W.x[k] - static_cast<double>(W.eff[k])) > 1e-6) {
              nr = true;
            }
          }
          return nr;
        };
        if (collect_cand()) continue;
        if (cand.empty()) {
          W.offer_integral();
          if (!lp_priced) {
            thr = H.thr();
            const double cv = thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
            lp_priced = W.ensure_priced(st, cv, thr, z);
            if (st != LpSolver::OPTIMAL || round_up(z) > thr) break;
            W.lp->get_x(W.x);
            if (collect_cand()) continue;
            if (cand.empty()) break;
          } else {
            break;
          }
        }
        if (exact && o.cuts != 0 && !W.lp->has_inactive_cols() && cur.depth >= 1 && cur.depth <= 4 &&
            (cur.depth <= 2 || was_from_queue)) {
          const int nc_res = W.try_node_cuts(cur.depth, z, thr, nodes_now, was_from_queue);
          if (nc_res < 0) break;
          if (nc_res > 0) {
            thr = H.thr();
            if (round_up(z) > thr) break;
            if (W.global_fix(thr)) break;
            if (o.node_fixing && exact && thr < INF && W.local_fix(z, thr)) {
              if (!W.propagate(false)) break;
            }
            if (collect_cand()) continue;
            if (cand.empty()) {
              W.offer_integral();
              break;
            }
          }
        }
        if (W.lp->rows() > base_rows) W.mark_idle_cuts();
        if (nonneg && (cur.depth == 0 || nodes_now % 10 == 0)) {
          W.round_greedy();
          thr = H.thr();
          if (round_up(z) > thr) {
            if (!lp_priced) {
              const double cv = thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
              lp_priced = W.ensure_priced(st, cv, thr, z);
              if (st != LpSolver::OPTIMAL || round_up(z) > thr) break;
              W.lp->get_x(W.x);
              if (collect_cand()) continue;
              if (cand.empty()) {
                W.offer_integral();
                break;
              }
            } else {
              break;
            }
          }
        }
        if (was_from_queue && lp_priced) W.maybe_age_cols(nodes_now);
        if (partition && W.branch_pair(cur, z, cand)) continue;
        const int r = W.branch_var(cur, z, thr, cand, lp_priced);
        if (r == -2) break;  // both children are infeasible or cut off
      }
      W.undo_local();
      W.clear_local_cuts();
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
    std::mt19937 rng(o.seed * 2654435761u + 99u);
    CutPool pool;
    std::vector<double> x;
    double z = lp0->obj() - P.offset;
    int stall = 0;
    // cuts are dense rows: their nonzeros together stay below those of the loaded columns (or 50000)
    long cut_nnz = 0;
    const long budget = std::max(nnz0, 50000L), longest = std::max(200L, nk / 5L);
    std::vector<int> cut_len;  // nonzeros of the cut rows now in the LP
    std::vector<char> row_int = base_row_int;
    std::vector<int> ord, rptr, ridx, sel_len;
    std::vector<double> rval, rlo, rup, cdense(nk, 0.0), sel_norm;
    std::vector<char> sel_int;
    auto select_and_add = [&](int round_cap, long len_lim, double par_tol, bool check_keep_hook) -> bool {
      long added_nnz = 0;
      if (!select_cuts_into(pool, round_cap, len_lim, budget - cut_nnz, par_tol, rptr, ridx, rval, rlo, rup, sel_len,
                            sel_int, added_nnz, ord, cdense, sel_norm))
        return false;
      cut_nnz += added_nnz;
      cut_len.insert(cut_len.end(), sel_len.begin(), sel_len.end());
      row_int.insert(row_int.end(), sel_int.begin(), sel_int.end());
      lp0->add_rows(rptr, ridx, rval, rlo, rup);
      ++R.cut_rounds;
      if (lp0->solve(INF) != LpSolver::OPTIMAL) return false;
      const double zn = lp0->obj() - P.offset;
      const std::vector<char> rs = lp0->row_status();
      std::vector<int> loose;
      for (int i = P.m; i < static_cast<int>(rs.size()); ++i)
        if (rs[i] == 0) loose.push_back(i);
      if (!loose.empty()) {
        for (size_t a = loose.size(); a-- > 0;) {
          cut_nnz -= cut_len[loose[a] - P.m];
          cut_len.erase(cut_len.begin() + (loose[a] - P.m));
          row_int.erase(row_int.begin() + loose[a]);
        }
        lp0->remove_rows(loose);
        if (lp0->solve(INF) != LpSolver::OPTIMAL) return false;
      }
      stall = zn - z < 1e-5 * std::max(1.0, std::fabs(z)) ? stall + 1 : 0;
      z = zn;
      if (check_keep_hook && H.keep && !check_keep(z)) {
        kept = false;
        return false;
      }
      return true;
    };

    // Phase 1: combinatorial cuts (cliques + zero-half).
    if ((o.cuts & 7) && set_rows) {
      for (int round = 0; round < o.cut_rounds && !stop() && now() < t_end; ++round) {
        lp0->get_x(x);
        pool.clear();
        const int cap = std::max(20, P.m / 4);
        if (o.cuts & 4) separate_cliques(*set_rows, x, 0.02, cap, pool);
        if (o.cuts & 1) separate_zero_half(*set_rows, x, true, o.cut_size, 0.02, cap, rng, pool);
        if (o.cuts & 2) separate_zero_half(*set_rows, x, false, o.cut_size, 0.02, cap, rng, pool);
        if (!select_and_add(cap, longest, 0.98, true)) break;
        if (stall >= 3 || round_up(z) > H.thr()) break;
      }
    }

    // Phase 2: Gomory mixed-integer (GMI) cuts after combinatorial cuts stall, with self-verifying rollback
    // if the added fill-in does not close enough remaining gap.
    if (kept && (o.cuts & 8) && !stop() && now() < t_end && round_up(z) <= H.thr()) {
      const int rows_before = lp0->rows();
      const double z_before = z;
      const long nnz_before = cut_nnz;
      const long gmi_longest = std::min<long>(nk, 1200L);
      const int gmi_cap = std::min(std::max(15, P.m / 6), 25);
      stall = 0;
      for (int gr = 0; gr < 4 && !stop() && now() < t_end; ++gr) {
        lp0->get_x(x);
        pool.clear();
        if (separate_gmi(*lp0, row_int, x, 0.005, gmi_cap, static_cast<int>(gmi_longest), pool) == 0) break;
        if (!select_and_add(gmi_cap, gmi_longest, 0.95, true)) break;
        if (stall >= 2 || round_up(z) > H.thr()) break;
      }
      if (kept && lp0->rows() > rows_before) {
        const double thr_g = H.thr();
        const long gmi_nnz = std::max(0L, cut_nnz - nnz_before);
        const double closed = z - z_before;
        bool drop_gmi = false;
        if (thr_g < INF) {
          const double rem_gap = std::max(1e-9, thr_g - z_before);
          const double req = std::max(0.02, 0.10 * static_cast<double>(gmi_nnz) / std::max(10000L, nnz0));
          if (closed < req * rem_gap) drop_gmi = true;
          if (o.integral && round_up(z) == round_up(z_before) && closed < 0.10 * rem_gap) drop_gmi = true;
        } else {
          if (closed < 1e-3 * std::max(1.0, std::fabs(z_before))) drop_gmi = true;
        }
        if (drop_gmi) {
          lp0->remove_rows(rows_before);
          cut_len.resize(static_cast<size_t>(rows_before - P.m));
          row_int.resize(static_cast<size_t>(rows_before));
          cut_nnz = nnz_before;
          if (lp0->solve(INF) == LpSolver::OPTIMAL) z = lp0->obj() - P.offset;
        }
      }
    }

    // Cuts that close little of the gap relative to the fill-in they add only slow every node LP down.
    const double thr_now = H.thr();
    const double min_close =
        ((o.cuts & 6) == 0 && o.integral) ? std::max(0.05, 0.25 * static_cast<double>(cut_nnz) / std::max(1L, nnz0))
                                          : 0.05;
    if (kept && lp0->rows() > P.m && thr_now < INF && z - zr0 < min_close * (thr_now - zr0)) {
      lp0->remove_rows(P.m);
      row_int.resize(static_cast<size_t>(P.m));
    }
    base_rows = lp0->rows();
    base_row_int = std::move(row_int);
    R.cuts = base_rows - P.m;
    R.cut_time += now() - t_begin;
    if (lp0->solve(INF) == LpSolver::OPTIMAL) {
      zr0 = lp0->obj() - P.offset;
      lp0->get_red_cost(dk);
      R.root_cut = zr0;
      cached_gb.store(zr0, std::memory_order_relaxed);
    }
    return kept;
  }

  // Root domain probing (Savelsbergh 1994) & binary conflict graph extraction (Atamturk et al. 2000).
  // Purely propagation-based (zero LPs per probe). Returns false only if root propagation proves infeasibility.
  bool root_probing() {
    if (lp0->solve(INF) != LpSolver::OPTIMAL) return true;
    std::vector<double> x0;
    lp0->get_x(x0);
    std::vector<int> cands(nk);
    for (int k = 0; k < nk; ++k) cands[k] = k;
    auto is_frac = [&](int k) { return x0[k] > 1e-3 && x0[k] < 1.0 - 1e-3; };
    std::sort(cands.begin(), cands.end(), [&](int a, int b) {
      const bool fa = is_frac(a), fb = is_frac(b);
      if (fa != fb) return fa;
      if (fa) {
        const double da = std::fabs(x0[a] - 0.5), db = std::fabs(x0[b] - 0.5);
        return da < db || (da == db && a < b);
      }
      const double ra = std::fabs(dk[a]), rb = std::fabs(dk[b]);
      return ra < rb || (ra == rb && a < b);
    });
    const int max_probes = nk <= 8000 ? std::min(nk, 180) : 120;
    cands.resize(static_cast<size_t>(max_probes));

    Worker W(*this, -1, std::move(lp0));
    auto promote_lfixed = [&]() {
      for (int k : W.lfixed) {
        const std::int8_t val = W.eff[k];
        W.lfix[k] = 0;
        if (val == 0 || val == 1) {
          if (root_gfix[k] == -1) {
            root_gfix[k] = val;
            W.gfix[k] = val;
            W.rk[k] = Worker::R_GLOBAL;
            ++R.root_probed;
          }
        }
      }
      W.lfixed.clear();
    };
    if (!W.propagate(false, false)) {
      lp0 = std::move(W.lp);
      return false;
    }
    promote_lfixed();

    const long probed_before = R.root_probed;
    const double t_end = now() + std::min(0.20, 0.025 * o.time_limit);
    std::vector<std::int8_t> imp1_val(nk, -1);
    struct Imp {
      int b;
      std::int8_t v;
      bool nontrivial;
    };
    std::vector<Imp> imp1, imp0;

    for (int k : cands) {
      if (now() >= t_end || stop()) break;
      if (W.eff[k] != -1) continue;
      const bool ok1 = W.start_probe(k, 1, false);
      imp1.clear();
      if (ok1) {
        for (int b : W.pfixed)
          if (b != k && (W.eff[b] == 0 || W.eff[b] == 1))
            imp1.push_back({b, W.eff[b], W.eff[b] == 1 || W.rk[b] != Worker::R_ROW_UP});
      }
      W.undo_probe();
      if (!ok1) {
        root_gfix[k] = 0;
        W.gfix[k] = 0;
        W.set_eff(k, 0, Worker::R_GLOBAL, 0);
        ++R.root_probed;
        if (!W.propagate(false, false)) {
          lp0 = std::move(W.lp);
          return false;
        }
        promote_lfixed();
        continue;
      }

      const bool ok0 = W.start_probe(k, 0, false);
      imp0.clear();
      if (ok0) {
        for (int b : W.pfixed)
          if (b != k && (W.eff[b] == 0 || W.eff[b] == 1)) imp0.push_back({b, W.eff[b], true});
      }
      W.undo_probe();
      if (!ok0) {
        root_gfix[k] = 1;
        W.gfix[k] = 1;
        W.set_eff(k, 1, Worker::R_GLOBAL, 0);
        ++R.root_probed;
        if (!W.propagate(false, false)) {
          lp0 = std::move(W.lp);
          return false;
        }
        promote_lfixed();
        continue;
      }

      for (const auto& pr : imp1) imp1_val[pr.b] = pr.v;
      std::vector<std::pair<int, std::int8_t>> common;
      for (const auto& pr : imp0) {
        if (imp1_val[pr.b] == pr.v) common.push_back({pr.b, pr.v});
      }
      for (const auto& pr : imp1) imp1_val[pr.b] = -1;
      for (const auto& pr : common) {
        const int b = pr.first;
        const std::int8_t val = pr.second;
        if (root_gfix[b] == 1 - val) {
          lp0 = std::move(W.lp);
          return false;
        }
        if (root_gfix[b] == -1) {
          root_gfix[b] = val;
          W.gfix[b] = val;
          W.set_eff(b, val, Worker::R_GLOBAL, 0);
          ++R.root_probed;
        }
      }
      if (!common.empty()) {
        if (!W.propagate(false, false)) {
          lp0 = std::move(W.lp);
          return false;
        }
        promote_lfixed();
      }
      if (bin_clause_cnt < 2048) {
        int added_k = 0;
        for (const auto& pr : imp1) {
          if (bin_clause_cnt >= 2048 || added_k >= 16) break;
          if (!pr.nontrivial || root_gfix[pr.b] != -1) continue;
          add_conflict_clause({(k << 1) | 0, (pr.b << 1) | pr.v});
          ++added_k;
        }
        for (const auto& pr : imp0) {
          if (bin_clause_cnt >= 2048 || added_k >= 32) break;
          if (root_gfix[pr.b] != -1) continue;
          add_conflict_clause({(k << 1) | 1, (pr.b << 1) | pr.v});
          ++added_k;
        }
      }
    }
    if (R.root_probed > probed_before) {
      const LpSolver::Status st = W.lp->solve(INF);
      if (st == LpSolver::INFEASIBLE) {
        lp0 = std::move(W.lp);
        return false;
      }
      if (st == LpSolver::OPTIMAL) {
        zr0 = W.lp->obj() - P.offset;
        W.lp->get_red_cost(dk);
        R.root_cut = zr0;
        cached_gb.store(zr0, std::memory_order_relaxed);
        H.offer_lb(round_up(zr0));
      }
    }
    lp0 = std::move(W.lp);
    return true;
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
