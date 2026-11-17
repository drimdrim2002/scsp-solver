#include "scsp/bnb.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>

namespace scsp {

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
};

struct NodeCmp {
  bool operator()(const Node* a, const Node* b) const {
    return a->bound > b->bound || (a->bound == b->bound && a->depth < b->depth);
  }
};

// One tree search over a fixed subset of the columns.
class Search {
 public:
  enum { EXHAUSTED = 0, STOPPED = 1, INCOMPLETE = 2, NO_INCUMBENT = 3 };

  Search(const LpProblem& P_, const BnbHooks& H_, const BnbOptions& o_, const std::vector<double>& droot_,
         double zroot_, std::vector<int> cols_, bool exact_, double time_left, BnbResult& R_)
      : P(P_), H(H_), o(o_), droot(droot_), zroot(zroot_), cols(std::move(cols_)), exact(exact_), R(R_) {
    LpOptions lo;
    lo.seed = o.seed;
    lo.time_limit = time_left;
    lp.reset(new LpSolver(P, cols, lo));
    nk = static_cast<int>(cols.size());
    fv.assign(nk, -1);
    gfix.assign(nk, 0);
    want.assign(nk, 0);
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
    act.assign(P.m, 0.0);
    if (partition) {
      rmark.assign(P.m, 0);
      if (static_cast<double>(P.m) * P.m <= 8e6) pairw.assign(static_cast<size_t>(P.m) * P.m, 0.0);
      else partition = false;
    }
  }

  ~Search() { R.lp_iters += lp->iterations(); }

  int run() {
    auto drop_all = [&]() {
      for (Node* nd : open) delete nd;
      open.clear();
    };
    push(new Node());
    bool incomplete = false;
    std::vector<int> cand;
    // First incumbent: one LP dive (good solutions on easy problems) and, for partitioning problems,
    // the cover search. If the best-first tree has still found nothing after a while, the feasibility
    // pump and restarted dives get their turn; a restricted column set that yields nothing is given up.
    if (H.thr() == INF) {
      bool ok = dive_for_incumbent(1, 0.1);
      if (ok && H.thr() == INF && pure_sp) ok = cover_search(0.1);
      if (!ok) {
        drop_all();
        return STOPPED;
      }
    }
    long heur_at = H.thr() == INF ? R.nodes + 3000 : -1;
    const double heur_time = now() + 0.1 * o.time_limit;
    while (!open.empty()) {
      if (stop()) {
        drop_all();
        return STOPPED;
      }
      if (H.yield) H.yield();
      if (heur_at >= 0 && (R.nodes > heur_at || now() > heur_time)) {
        heur_at = -1;
        if (H.thr() == INF) {
          bool ok = feasibility_pump(0.1);
          if (ok && H.thr() == INF) ok = dive_for_incumbent(1 << 30, 0.2);
          if (!ok) {
            drop_all();
            return STOPPED;
          }
          if (!exact && H.thr() == INF) {
            drop_all();
            return NO_INCUMBENT;
          }
        }
      }
      double thr = H.thr();
      global_fix(thr);
      std::pop_heap(open.begin(), open.end(), NodeCmp());
      std::unique_ptr<Node> top(open.back());
      open.pop_back();
      if (exact && top->bound > -INF && !incomplete) H.offer_lb(std::min(round_up(top->bound), ub_of(thr)));
      if (round_up(top->bound) > thr) {  // best-first: every other open node is at least as bad
        drop_all();
        break;
      }
      Node cur = std::move(*top);
      for (;;) {  // plunge
        if (stop()) {
          drop_all();
          return STOPPED;
        }
        if (H.yield) {
          H.yield();
          thr = H.thr();
          global_fix(thr);
        }
        switch_to(cur.fix);
        const LpSolver::Status st = lp->solve(thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr)));
        ++R.nodes;
        if (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF) break;
        if (st != LpSolver::OPTIMAL) {
          if (stop()) {
            drop_all();
            return STOPPED;
          }
          incomplete = true;
          break;
        }
        const double z = lp->obj() - P.offset;
        if (cur.bvar >= 0) {  // pseudocost of the branching that led here
          pcs[cur.bdir][cur.bvar] += std::max(z - cur.pz, 0.0) / std::max(cur.bfrac, 1e-9);
          ++pcn[cur.bdir][cur.bvar];
          cur.bvar = -1;
        }
        thr = H.thr();
        if (round_up(z) > thr) break;
        lp->get_x(x);
        cand.clear();
        for (int k = 0; k < nk; ++k)
          if (x[k] > FRAC && x[k] < 1.0 - FRAC) cand.push_back(k);
        if (cand.empty()) {
          offer_integral();
          break;
        }
        if (nonneg && (cur.depth == 0 || R.nodes % 10 == 0)) {
          round_greedy();
          thr = H.thr();
          if (round_up(z) > thr) break;
        }
        if (partition && branch_pair(cur, z, cand)) continue;
        const int r = branch_var(cur, z, thr, cand);
        if (r == -2) break;  // both children are infeasible or cut off
      }
    }
    return incomplete ? INCOMPLETE : EXHAUSTED;
  }

 private:
  const LpProblem& P;
  const BnbHooks& H;
  const BnbOptions& o;
  const std::vector<double>& droot;
  const double zroot;
  std::vector<int> cols;
  const bool exact;
  BnbResult& R;
  std::unique_ptr<LpSolver> lp;
  int nk = 0;
  std::vector<int> fv, curfix, want;
  std::vector<char> gfix, rmark;
  std::vector<double> x, act, pcs[2], pairw;
  std::vector<int> pcn[2];
  std::vector<int> order, chosen, touched_pairs;
  double last_thr = INF;
  long sb_iters = 0;
  bool nonneg = true, partition = false, pure_sp = false;

  std::vector<Node*> open;  // heap, best bound first

  void push(Node* nd) {
    open.push_back(nd);
    std::push_heap(open.begin(), open.end(), NodeCmp());
  }

  const Clock::time_point t_start = Clock::now();
  double now() const { return std::chrono::duration<double>(Clock::now() - t_start).count(); }

  bool stop() const { return H.stop() || (o.node_limit >= 0 && R.nodes >= o.node_limit); }
  double round_up(double z) const { return o.integral ? std::ceil(z - 1e-6) : z; }
  double ub_of(double thr) const {  // the incumbent value behind thr
    if (thr == INF) return INF;
    return o.integral ? thr + 1.0 - 1e-6 : thr;
  }

  // Columns whose root reduced cost rules them out of every solution that still counts.
  void global_fix(double thr) {
    if (!(thr < last_thr)) return;
    last_thr = thr;
    for (int k = 0; k < nk; ++k) {
      if (gfix[k]) continue;
      const double dj = droot[cols[k]];
      if (dj > 0.0 && round_up(zroot + dj - 1e-9 * std::max(1.0, std::fabs(zroot))) > thr) {
        gfix[k] = 1;
        if (fv[k] == -1) lp->set_col_bounds(k, 0.0, 0.0);
      }
    }
  }

  void switch_to(const std::vector<int>& fix) {
    for (int e : fix) want[e >> 1] = (e & 1) + 1;
    for (int e : curfix) {
      const int k = e >> 1;
      if (want[k] - 1 != fv[k]) {
        fv[k] = -1;
        lp->set_col_bounds(k, 0.0, gfix[k] ? 0.0 : 1.0);
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
    for (int k = 0; k < nk; ++k)
      if (x[k] > 0.5) {
        sol.push_back(cols[k]);
        cost += P.cost[cols[k]];
      }
    std::sort(sol.begin(), sol.end());
    H.offer(sol, cost);
  }

  // Exact cover search for pure partitioning problems, without LPs: take the uncovered row with the
  // fewest usable columns, try its columns (large root LP value first, then small reduced cost),
  // and propagate: a chosen column covers its rows and bans every column that meets them. The
  // search restarts with a longer node limit and a reshuffled column order when it runs dry.
  // Returns false on a stop request.
  bool cover_search(double time_frac) {
    const int m = P.m;
    std::vector<int> rptr(m + 1, 0), rcol;
    for (int k = 0; k < nk; ++k)
      if (!gfix[k])
        for (int q = P.cptr[cols[k]]; q < P.cptr[cols[k] + 1]; ++q) ++rptr[P.cidx[q] + 1];
    for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
    rcol.resize(rptr[m]);
    {
      std::vector<int> pos(rptr.begin(), rptr.end() - 1);
      for (int k = 0; k < nk; ++k)
        if (!gfix[k])
          for (int q = P.cptr[cols[k]]; q < P.cptr[cols[k] + 1]; ++q) rcol[pos[P.cidx[q]]++] = k;
    }
    std::vector<double> key(nk);
    std::vector<int> ban(nk, 0), avail(m), chosen_k;
    std::vector<char> cov(m, 0);
    std::mt19937 rng(o.seed * 69069u + 4242u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double dmax = 1e-12;
    for (int k = 0; k < nk; ++k) dmax = std::max(dmax, std::fabs(droot[cols[k]]));
    const double t0 = now();
    long limit = 2000;
    bool found = false;
    struct Frame {
      int row, next;
      std::vector<int> cands;
    };
    for (int round = 0; !found; ++round, limit *= 2) {
      // order key: smaller is tried first
      const double noise = round == 0 ? 0.0 : 0.3;
      for (int k = 0; k < nk; ++k) key[k] = droot[cols[k]] / dmax + noise * uni(rng);
      std::fill(ban.begin(), ban.end(), 0);
      std::fill(cov.begin(), cov.end(), 0);
      for (int i = 0; i < m; ++i) avail[i] = rptr[i + 1] - rptr[i];
      chosen_k.clear();
      std::vector<Frame> stack;
      long nodes = 0;
      int ncov = 0;
      auto choose = [&](int k) {
        const int j = cols[k];
        for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
          const int r = P.cidx[q];
          cov[r] = 1;
          ++ncov;
          for (int a = rptr[r]; a < rptr[r + 1]; ++a) {
            const int k2 = rcol[a];
            if (ban[k2]++ == 0)
              for (int b = P.cptr[cols[k2]]; b < P.cptr[cols[k2] + 1]; ++b) --avail[P.cidx[b]];
          }
        }
        chosen_k.push_back(k);
      };
      auto unchoose = [&]() {
        const int k = chosen_k.back();
        chosen_k.pop_back();
        const int j = cols[k];
        for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
          const int r = P.cidx[q];
          cov[r] = 0;
          --ncov;
          for (int a = rptr[r]; a < rptr[r + 1]; ++a) {
            const int k2 = rcol[a];
            if (--ban[k2] == 0)
              for (int b = P.cptr[cols[k2]]; b < P.cptr[cols[k2] + 1]; ++b) ++avail[P.cidx[b]];
          }
        }
      };
      auto open_row = [&]() {  // false if some uncovered row has no usable column left
        int best = -1;
        for (int i = 0; i < m; ++i)
          if (!cov[i] && (best < 0 || avail[i] < avail[best])) best = i;
        if (best < 0 || avail[best] == 0) return false;
        Frame f;
        f.row = best;
        f.next = 0;
        for (int a = rptr[best]; a < rptr[best + 1]; ++a)
          if (ban[rcol[a]] == 0) f.cands.push_back(rcol[a]);
        std::sort(f.cands.begin(), f.cands.end(), [&](int a, int b) { return key[a] < key[b] || (key[a] == key[b] && a < b); });
        stack.push_back(std::move(f));
        return true;
      };
      bool exhausted = false;
      if (!open_row()) exhausted = true;
      while (!exhausted && !found) {
        if ((nodes & 255) == 0 && (stop() || now() - t0 > time_frac * o.time_limit)) return !stop();
        if (nodes > limit) break;
        Frame& f = stack.back();
        if (f.next > 0) unchoose();  // undo the column tried last at this level
        if (f.next >= static_cast<int>(f.cands.size())) {
          stack.pop_back();
          if (stack.empty()) exhausted = true;
          continue;
        }
        choose(f.cands[f.next++]);
        ++nodes;
        ++R.cover_nodes;
        if (ncov == m) {
          found = true;
          break;
        }
        if (!open_row()) continue;  // dead end: the loop undoes this column and tries the next
      }
      if (found) {
        std::vector<int> sol;
        double cost = 0.0;
        for (int k : chosen_k) {
          sol.push_back(cols[k]);
          cost += P.cost[cols[k]];
        }
        std::sort(sol.begin(), sol.end());
        H.offer(sol, cost);
      }
      if (exhausted) break;  // this column set has no partition at all
    }
    return true;
  }

  // Feasibility pump (Fischetti, Glover & Lodi 2005): alternate between rounding the LP point and
  // the LP point closest (in the L1 distance over the binaries) to the rounded one. A stalled
  // rounding gets its most contested entries flipped; a repeated one gets a random shake.
  // Returns false on a stop request. The costs of the LP are restored before returning.
  bool feasibility_pump(double time_frac) {
    if (lp->solve() != LpSolver::OPTIMAL) return !stop();
    ++R.nodes;
    lp->get_x(x);
    std::mt19937 rng(o.seed * 40503u + 977u);
    std::vector<char> xt(nk), prev1, prev2;
    for (int k = 0; k < nk; ++k) xt[k] = x[k] > 0.5;
    double cmax = 0.0;
    for (double c : P.cost) cmax = std::max(cmax, std::fabs(c));
    if (cmax == 0.0) cmax = 1.0;
    const double t0 = now();
    const double budget = time_frac * o.time_limit;
    std::vector<std::pair<double, int>> sc;
    bool ok = true, found = false;
    for (int it = 0; it < 300 && !found; ++it) {
      if (stop()) {
        ok = false;
        break;
      }
      if (now() - t0 > budget) break;
      for (int k = 0; k < nk; ++k) lp->set_col_cost(k, xt[k] ? -cmax : cmax);
      if (lp->solve() != LpSolver::OPTIMAL) break;
      ++R.pump_iters;
      lp->get_x(x);
      bool integral_point = true, same = true;
      for (int k = 0; k < nk; ++k) {
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
        for (int k = 0; k < nk; ++k) xt[k] = x[k] > 0.5;
        if (xt != prev2) continue;
      }
      // stalled (same rounding) or a cycle of length two: flip the entries the LP disagrees with most
      sc.clear();
      for (int k = 0; k < nk; ++k) {
        const double dist = std::fabs(x[k] - (xt[k] ? 1.0 : 0.0));
        if (dist > 1e-9) sc.push_back({-dist, k});
      }
      std::sort(sc.begin(), sc.end());
      const size_t flips = std::min<size_t>(sc.size(), 10 + rng() % 20);
      for (size_t q = 0; q < flips; ++q) xt[sc[q].second] ^= 1;
      if (sc.empty() || it % 30 == 29)  // nothing to flip, or no progress for a while: random shake
        for (int k = 0; k < nk; ++k)
          if (x[k] > FRAC && rng() % 4 == 0) xt[k] ^= 1;
    }
    for (int k = 0; k < nk; ++k) lp->set_col_cost(k, P.cost[cols[k]]);
    if (found) {
      offer_integral();
      ++R.pump_found;
    }
    return ok;
  }

  // Restarted randomized diving while there is no incumbent: fix a variable with a large LP value
  // to one, solve, repeat; an infeasible LP flips the last fixing, and after a few flips the dive is
  // abandoned and a new one starts from the root (chronological backtracking alone keeps repairing
  // the bottom of a dive whose mistake is near the top). Returns false on a stop request.
  bool dive_for_incumbent(int max_dives, double time_frac) {
    std::mt19937 rng(o.seed * 2654435761u + 12345u);
    std::vector<int> fix, cand;
    const double t0 = now();
    const double budget = time_frac * o.time_limit;
    for (int dive = 0; dive < max_dives && H.thr() == INF; ++dive) {
      fix.clear();
      int flips = 0;
      const int max_flips = 10 + 5 * std::min(dive, 40);
      for (;;) {
        if (stop()) return false;
        if (now() - t0 > budget) {  // leave the rest of the time to the complete search
          switch_to({});
          return true;
        }
        switch_to(fix);
        const LpSolver::Status st = lp->solve();
        ++R.nodes;
        ++R.dive_nodes;
        if (st != LpSolver::OPTIMAL) {
          if (st != LpSolver::INFEASIBLE && stop()) return false;
          while (!fix.empty() && (fix.back() & 1) == 0) fix.pop_back();
          if (fix.empty()) {
            if (dive == 0 && flips == 0) {  // the root itself is infeasible: the tree search will report it
              switch_to({});
              return true;
            }
            break;
          }
          if (++flips > max_flips) break;
          fix.back() -= 1;  // the last variable fixed to one goes to zero instead
          continue;
        }
        lp->get_x(x);
        cand.clear();
        for (int k = 0; k < nk; ++k)
          if (x[k] > FRAC && x[k] < 1.0 - FRAC) cand.push_back(k);
        if (cand.empty()) {
          offer_integral();
          break;
        }
        if (nonneg && fix.empty()) round_greedy();
        // one of the three largest values, the largest most often
        std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(3, cand.size()), cand.end(), [&](int a, int b) {
          return x[a] > x[b] || (x[a] == x[b] && a < b);
        });
        const unsigned r = dive == 0 ? 0 : rng() % 6;
        const size_t pick = std::min<size_t>(r < 3 ? 0 : (r < 5 ? 1 : 2), cand.size() - 1);
        fix.push_back(cand[pick] * 2 + 1);
      }
      ++R.dives;
    }
    switch_to({});
    return true;
  }

  // Greedy rounding for set models: take columns by decreasing LP value while they
  // serve an unmet row and fit every row's upper side, then drop redundant ones.
  void round_greedy() {
    order.clear();
    for (int k = 0; k < nk; ++k)
      if (x[k] > FRAC) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (x[a] != x[b]) return x[a] > x[b];
      if (P.cost[cols[a]] != P.cost[cols[b]]) return P.cost[cols[a]] < P.cost[cols[b]];
      return a < b;
    });
    std::fill(act.begin(), act.end(), 0.0);
    chosen.clear();
    for (int k : order) {
      const int j = cols[k];
      bool helps = false, fits = true;
      for (int q = P.cptr[j]; q < P.cptr[j + 1] && fits; ++q) {
        const int i = P.cidx[q];
        if (act[i] < P.rlo[i] - 1e-9) helps = true;
        if (act[i] + P.cval[q] > P.rup[i] + 1e-9) fits = false;
      }
      if (!helps || !fits) continue;
      chosen.push_back(j);
      for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) act[P.cidx[q]] += P.cval[q];
    }
    for (int i = 0; i < P.m; ++i)
      if (act[i] < P.rlo[i] - 1e-9) return;
    std::sort(chosen.begin(), chosen.end(), [&](int a, int b) { return P.cost[a] > P.cost[b] || (P.cost[a] == P.cost[b] && a < b); });
    std::vector<int> sol;
    double cost = 0.0;
    for (int j : chosen) {
      bool redundant = true;
      for (int q = P.cptr[j]; q < P.cptr[j + 1] && redundant; ++q)
        if (act[P.cidx[q]] - P.cval[q] < P.rlo[P.cidx[q]] - 1e-9) redundant = false;
      if (redundant) {
        for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) act[P.cidx[q]] -= P.cval[q];
      } else {
        sol.push_back(j);
        cost += P.cost[j];
      }
    }
    if (cost <= H.thr()) {
      std::sort(sol.begin(), sol.end());
      H.offer(sol, cost);
    }
  }

  // Ryan-Foster: two rows r, s covered together by a fractional amount. One child forbids the
  // columns that cover exactly one of them, the other forbids the columns that cover both.
  bool branch_pair(Node& cur, double z, const std::vector<int>& cand) {
    const size_t m = static_cast<size_t>(P.m);
    touched_pairs.clear();
    for (int k : cand) {
      const int j = cols[k];
      for (int a = P.cptr[j]; a < P.cptr[j + 1]; ++a)
        for (int b = a + 1; b < P.cptr[j + 1]; ++b) {
          const size_t id = static_cast<size_t>(P.cidx[a]) * m + P.cidx[b];
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
    if (best < 0) return false;  // only duplicate columns are fractional: branch on a variable instead
    const int r = best / P.m, s = best % P.m;
    rmark[r] = rmark[s] = 1;
    Node* differ = new Node();
    differ->bound = z;
    differ->depth = cur.depth + 1;
    differ->fix = cur.fix;
    for (int k = 0; k < nk; ++k) {
      if (fv[k] != -1 || gfix[k]) continue;
      const int j = cols[k];
      int hit = 0;
      for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) hit += rmark[P.cidx[q]];
      if (hit == 1) cur.fix.push_back(k * 2);
      else if (hit == 2) differ->fix.push_back(k * 2);
    }
    rmark[r] = rmark[s] = 0;
    push(differ);
    cur.bound = z;
    ++cur.depth;
    return true;
  }

  // Reliability branching. Returns the branching column, -1 after fixing a variable whose
  // other side is impossible (the node must be solved again), -2 if the node is finished.
  int branch_var(Node& cur, double z, double thr, std::vector<int>& cand) {
    std::sort(cand.begin(), cand.end(), [&](int a, int b) {
      const double fa = std::fabs(x[a] - 0.5), fb = std::fabs(x[b] - 0.5);
      return fa < fb || (fa == fb && a < b);
    });
    double avg[2] = {1.0, 1.0};
    for (int d = 0; d < 2; ++d) {
      double s = 0.0;
      long c = 0;
      for (int k = 0; k < nk; ++k)
        if (pcn[d][k]) s += pcs[d][k] / pcn[d][k], ++c;
      if (c) avg[d] = s / c;
    }
    const bool may_strong = o.strong_cands > 0 && (cur.depth <= 8 || sb_iters * 3 < lp->iterations());
    const double cut = thr == INF ? INF : thr + P.offset + 1e-9 * std::max(1.0, std::fabs(thr));
    std::vector<char> basis;
    int best = -1, tried = 0;
    double best_score = -1.0;
    const std::vector<double> x0 = x;
    for (int k : cand) {
      const double f = x0[k];
      double d0, d1;
      const bool unreliable = pcn[0][k] < o.reliability || pcn[1][k] < o.reliability;
      if (may_strong && unreliable && tried < o.strong_cands) {
        ++tried;
        if (basis.empty()) basis = lp->get_basis();
        double zc[2];
        for (int v = 0; v < 2; ++v) {
          const long it0 = lp->iterations();
          lp->set_col_bounds(k, v, v);
          const LpSolver::Status st = lp->solve(cut, o.strong_iters);
          ++R.strong_lps;
          zc[v] = (st == LpSolver::INFEASIBLE || st == LpSolver::CUTOFF) ? INF
                  : (st == LpSolver::OPTIMAL || st == LpSolver::LIMIT) ? std::max(lp->obj() - P.offset, z) : z;
          if (zc[v] < INF && round_up(zc[v]) > thr) zc[v] = INF;
          sb_iters += lp->iterations() - it0;
          lp->set_col_bounds(k, 0.0, 1.0);
          lp->set_basis(basis);
        }
        if (zc[0] == INF && zc[1] == INF) return -2;
        if (zc[0] == INF || zc[1] == INF) {
          cur.fix.push_back(k * 2 + (zc[0] == INF ? 1 : 0));
          return -1;
        }
        d0 = zc[0] - z;
        d1 = zc[1] - z;
        pcs[0][k] += d0 / std::max(f, 1e-9);
        ++pcn[0][k];
        pcs[1][k] += d1 / std::max(1.0 - f, 1e-9);
        ++pcn[1][k];
      } else {
        d0 = (pcn[0][k] ? pcs[0][k] / pcn[0][k] : avg[0]) * f;
        d1 = (pcn[1][k] ? pcs[1][k] / pcn[1][k] : avg[1]) * (1.0 - f);
      }
      const double score = std::max(d0, 1e-6) * std::max(d1, 1e-6);
      if (score > best_score) best_score = score, best = k;
    }
    Node* down = new Node();
    down->bound = z;
    down->depth = cur.depth + 1;
    down->fix = cur.fix;
    down->fix.push_back(best * 2);
    down->bvar = best;
    down->bdir = 0;
    down->bfrac = x0[best];
    down->pz = z;
    push(down);
    cur.fix.push_back(best * 2 + 1);
    cur.bound = z;
    ++cur.depth;
    cur.bvar = best;
    cur.bdir = 1;
    cur.bfrac = 1.0 - x0[best];
    cur.pz = z;
    return best;
  }
};

}  // namespace

BnbResult branch_and_bound(const LpProblem& P, const BnbHooks& H, const BnbOptions& opt) {
  BnbResult R;
  const auto t0 = Clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  const int n = P.n, m = P.m;

  LpOptions lo;
  lo.seed = opt.seed;
  lo.time_limit = opt.time_limit;
  const LpResult root = solve_lp(P, lo);
  R.root_time = elapsed();
  if (root.status == "infeasible") {
    R.status = BnbResult::LP_INFEASIBLE;
    return R;
  }
  if (root.status != "optimal") {
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
