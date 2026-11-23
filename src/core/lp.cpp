#include "scsp/lp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

#include "scsp/lu.hpp"

namespace scsp {

struct LpSolver::Snapshot {
  const void* owner = nullptr;
  uint64_t cost_epoch = 0;
  uint64_t active_epoch = 0;
  BasisLU lu;
  std::vector<int> head, bpos;
  std::vector<double> d, y, x;
  double cur_obj = 0.0;
};

namespace {

const double INF = std::numeric_limits<double>::infinity();
const double PIV = 1e-9;  // smallest pivot element accepted by the ratio tests

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double sec() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

// Bounded simplex on  [A -I] (x, r) = 0  with bounds on the structural variables x
// and on the row activities r (the logical variables).
// Variable index v: v < m is the logical of row v, v >= m is structural column v - m.
class Simplex {
 public:
  enum : char { BASIC = 0, LOWER = 1, UPPER = 2, FREE = 3, MARK = 4 };
  enum { OPTIMAL = 0, INFEASIBLE = 1, UNBOUNDED = 2, LIMIT = 3, FAILED = 4, CUTOFF = 5 };
  double cutoff = INF;  // dual(): stop once the dual objective (scaled costs) exceeds this
  long iter_cap = -1;   // dual(): iterations allowed in one call (-1 = no cap)

  int m = 0, nc = 0;
  std::vector<double> c, c0, lo, up, x, d, y;
  std::vector<char> st;
  std::vector<int> head, bpos;
  long it_dual = 0, it_primal = 0, n_refactor = 0, n_flip = 0;

  Simplex(int rows, const double* rlo, const double* rup, const LpOptions& o, const Timer& t)
      : m(rows), opt(o), timer(t), rng(o.seed) {
    c.assign(m, 0.0);
    c0 = c;
    lo.assign(rlo, rlo + m);
    up.assign(rup, rup + m);
    x.assign(m, 0.0);
    d.assign(m, 0.0);
    y.assign(m, 0.0);
    st.assign(m, BASIC);
    head.resize(m);
    bpos.resize(m);
    lidx.resize(m);
    std::iota(head.begin(), head.end(), 0);
    std::iota(bpos.begin(), bpos.end(), 0);
    std::iota(lidx.begin(), lidx.end(), 0);
    dse.assign(m, 1.0);
    rho.assign(m, 0.0);
    aq.assign(m, 0.0);
    tau.assign(m, 0.0);
    spike.assign(m, 0.0);
    rowv.assign(m, 0.0);
    posv.assign(m, 0.0);
    rmark.assign(m, 0);
    cptr.assign(1, 0);
  }

  void add_column(const int* idx, const double* val, int len, double cost, double lb, double ub) {
    cidx.insert(cidx.end(), idx, idx + len);
    cval.insert(cval.end(), val, val + len);
    cptr.push_back(static_cast<int>(cidx.size()));
    c.push_back(cost);
    c0.push_back(cost);
    lo.push_back(lb);
    up.push_back(ub);
    const char s = lb > -INF ? LOWER : (ub < INF ? UPPER : FREE);
    st.push_back(s);
    x.push_back(s == LOWER ? lb : (s == UPPER ? ub : 0.0));
    d.push_back(0.0);
    bpos.push_back(-1);
    ++nc;
    factored = false;  // the new column has no reduced cost yet
    primal_valid = false;
    dual_valid = false;
  }

  // Call after adding columns and before solving.
  void finalize() {
    rebuild_rows();
    arow.assign(m + nc, 0.0);
    touched.resize(m + nc);
    n_touched = 0;
    cands.resize(m + nc);
    devex.assign(m + nc, 1.0);
  }

  void rebuild_rows(const std::vector<char>* iw = nullptr) {
    rptr.assign(m + 1, 0);
    if (!iw || iw->empty()) {
      use_active = false;
      active_list.clear();
      for (int i : cidx) ++rptr[i + 1];
      for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
      ridx.resize(cidx.size());
      rval.resize(cidx.size());
      row_unit.assign(m, 1);
      std::vector<int> pos(rptr.begin(), rptr.end() - 1);
      for (int j = 0; j < nc; ++j)
        for (int q = cptr[j]; q < cptr[j + 1]; ++q) {
          const int i = cidx[q];
          const double v = cval[q];
          if (v != 1.0) row_unit[i] = 0;
          ridx[pos[i]] = m + j;
          rval[pos[i]++] = v;
        }
    } else {
      use_active = true;
      active_list.clear();
      size_t nnz_act = 0;
      for (int j = 0; j < nc; ++j) {
        if ((*iw)[j]) {
          active_list.push_back(j);
          for (int q = cptr[j]; q < cptr[j + 1]; ++q) {
            ++rptr[cidx[q] + 1];
            ++nnz_act;
          }
        }
      }
      for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
      ridx.resize(nnz_act);
      rval.resize(nnz_act);
      row_unit.assign(m, 1);
      std::vector<int> pos(rptr.begin(), rptr.end() - 1);
      for (int j : active_list)
        for (int q = cptr[j]; q < cptr[j + 1]; ++q) {
          const int i = cidx[q];
          const double v = cval[q];
          if (v != 1.0) row_unit[i] = 0;
          ridx[pos[i]] = m + j;
          rval[pos[i]++] = v;
        }
    }
  }

  void set_active(const std::vector<char>& iw, const std::vector<double>& ext_lb, const std::vector<double>& ext_ub) {
    ++active_epoch;
    if (iw.empty()) {
      for (int j = 0; j < nc; ++j) {
        lo[m + j] = ext_lb[j];
        up[m + j] = ext_ub[j];
      }
      rebuild_rows(nullptr);
      return;
    }
    for (int j = 0; j < nc; ++j) {
      if (iw[j]) {
        lo[m + j] = ext_lb[j];
        up[m + j] = ext_ub[j];
      } else {
        lo[m + j] = 0.0;
        up[m + j] = 0.0;
        st[m + j] = LOWER;
        x[m + j] = 0.0;
        d[m + j] = 0.0;
      }
    }
    rebuild_rows(&iw);
  }

  void activate_subset(const std::vector<int>& added, const std::vector<double>& ext_lb,
                       const std::vector<double>& ext_ub, const std::vector<char>& iw) {
    if (added.empty()) return;
    ++active_epoch;
    if (cost_mod) {
      c = c0;
      cost_mod = false;
    }
    for (int k : added) {
      active_list.push_back(k);
      const int v = m + k;
      lo[v] = ext_lb[k];
      up[v] = ext_ub[k];
      double dv = c0[v];
      for (int q = cptr[k]; q < cptr[k + 1]; ++q) dv -= cval[q] * y[cidx[q]];
      d[v] = dv;
      if (lo[v] == up[v]) {
        st[v] = LOWER;
        x[v] = lo[v];
        if (x[v] != 0.0) primal_valid = false;
      } else if (dv < -opt.opt_tol && up[v] < INF) {
        st[v] = UPPER;
        x[v] = up[v];
        primal_valid = false;
      } else {
        st[v] = lo[v] > -INF ? LOWER : (up[v] < INF ? UPPER : FREE);
        x[v] = st[v] == LOWER ? lo[v] : (st[v] == UPPER ? up[v] : 0.0);
        if (x[v] != 0.0) primal_valid = false;
      }
    }
    rebuild_rows(&iw);
  }

  void set_cost(int v, double cost) {
    c[v] = c0[v] = cost;
    ++cost_epoch;
  }

  // Dual simplex with perturbed costs, then the primal simplex with the true costs.
  int solve(bool pert = true) {
    if (!factored || pert || lu.updates() >= opt.refactor) {
      if (!refactor()) return FAILED;
      compute_dual();
      dual_valid = false;
      primal_valid = false;
    }
    if (!dual_valid) {
      if (fix_dual()) primal_valid = false;
      dual_valid = true;
    }
    if (pert && opt.perturb) perturb();
    int s = dual();
    if (cost_mod) {  // perturbed or shifted costs: back to the true ones, the primal simplex cleans up
      c = c0;
      cost_mod = false;
      factored = false;
      primal_valid = false;
      dual_valid = false;
      if (s == OPTIMAL) s = primal();
    }
    return s;
  }

  // ---- incremental use (branch and bound) ----
  void set_bounds(int v, double l, double u) {
    const bool was_fixed = (lo[v] == up[v]);
    lo[v] = l;
    up[v] = u;
    if (st[v] == BASIC) return;
    const char old_st = st[v];
    const double old_x = x[v];
    if (l == u || st[v] == LOWER) {
      if (l > -INF) st[v] = LOWER, x[v] = l;
      else if (u < INF) st[v] = UPPER, x[v] = u;
      else st[v] = FREE, x[v] = 0.0;
    } else if (st[v] == UPPER) {
      if (u < INF) x[v] = u;
      else if (l > -INF) st[v] = LOWER, x[v] = l;
      else st[v] = FREE, x[v] = 0.0;
    }
    if (x[v] != old_x) primal_valid = false;
    if (l != u && (was_fixed || st[v] != old_st)) dual_valid = false;
  }

  void change_cost(int v, double cost) {
    c[v] = c0[v] = cost;
    ++cost_epoch;
    factored = false;  // the reduced costs have to be recomputed
    dual_valid = false;
    primal_valid = false;
  }

  const std::vector<char>& basis() const { return st; }

  void sync_bounds_after_restore() {
    // Only preserve primal/dual validity if already valid prior to bound checks
    // (e.g. set_basis may hit b == st after set_bounds already invalidated primal_valid).
    bool pok = primal_valid, dok = dual_valid;
    auto check_v = [&](int v) {
      const char s = st[v];
      if (s == BASIC) return;
      if (s == LOWER) {
        if (lo[v] > -INF) {
          if (x[v] != lo[v]) x[v] = lo[v], pok = false;
        } else {
          set_nonbasic(v);
          pok = false;
          dok = false;
        }
      } else if (s == UPPER) {
        if (up[v] < INF) {
          if (x[v] != up[v]) x[v] = up[v], pok = false;
        } else {
          set_nonbasic(v);
          pok = false;
          dok = false;
        }
      } else {
        if (lo[v] > -INF || up[v] < INF) {
          set_nonbasic(v);
          pok = false;
          dok = false;
        }
      }
      if (dok && lo[v] != up[v]) {
        const double dv = d[v];
        const char ns = st[v];
        if ((ns == LOWER && dv < -opt.opt_tol) || (ns == UPPER && dv > opt.opt_tol) ||
            (ns == FREE && std::fabs(dv) > opt.opt_tol))
          dok = false;
      }
    };
    if (!use_active) {
      const int nv = m + nc;
      for (int v = 0; v < nv; ++v) check_v(v);
    } else {
      for (int v = 0; v < m; ++v) check_v(v);
      for (int j : active_list) check_v(m + j);
    }
    primal_valid = pok;
    dual_valid = dok;
    if (pok) cur_obj = objective();
  }

  bool set_basis(const std::vector<char>& b) {
    if (static_cast<int>(b.size()) != m + nc) return false;
    if (factored && !cost_mod && b == st) {
      std::fill(dse.begin(), dse.end(), 1.0);
      sync_bounds_after_restore();
      return true;
    }
    int k = 0;
    for (int v = 0; v < m + nc; ++v)
      if (b[v] == BASIC) ++k;
    if (k != m) return false;
    st = b;
    k = 0;
    for (int v = 0; v < m + nc; ++v) {
      if (st[v] == BASIC) {
        head[k] = v;
        bpos[v] = k++;
      } else {
        bpos[v] = -1;
        if (st[v] == LOWER && lo[v] > -INF) x[v] = lo[v];
        else if (st[v] == UPPER && up[v] < INF) x[v] = up[v];
        else set_nonbasic(v);
      }
    }
    std::fill(dse.begin(), dse.end(), 1.0);
    factored = false;
    primal_valid = false;
    dual_valid = false;
    return true;
  }

  // The dual steepest edge weights belong to the basis: with them a restored basis prices as well as before.
  const std::vector<double>& weights() const { return dse; }
  void set_weights(const std::vector<double>& w) {
    if (static_cast<int>(w.size()) == m) dse = w;
  }

  bool can_snapshot() const { return factored && !cost_mod && primal_valid && dual_valid; }

  double objective() const {
    double s = 0.0;
    if (!use_active) {
      for (int v = m; v < m + nc; ++v) s += c0[v] * x[v];
    } else {
      for (int j : active_list) s += c0[m + j] * x[m + j];
    }
    return s;
  }

  // Dual simplex from the current basis (made dual feasible by bound flips and cost shifts).
  int dual() {
    if (!factored || lu.updates() >= opt.refactor) {
      if (!refactor()) return FAILED;
      compute_dual();
      dual_valid = false;
      primal_valid = false;
    }
    if (!dual_valid) {
      if (fix_dual()) primal_valid = false;
      dual_valid = true;
    }
    if (!primal_valid) {
      compute_primal();
      primal_valid = true;
    }
    bool need = false;
    int trouble = 0;
    const long it_start = it_dual;
    bool stall_pert = false;
    auto cmp = [](const Cand& a, const Cand& b) { return a.theta > b.theta; };
    for (;;) {
      if (lu.updates() >= opt.refactor || need) {
        if (!refactor()) return FAILED;
        compute_dual();
        fix_dual();
        compute_primal();
        dual_valid = true;
        primal_valid = true;
        need = false;
      }
      if ((it_dual & 31) == 0 && (timer.sec() > opt.time_limit || (opt.stop && opt.stop->load(std::memory_order_relaxed))))
        return LIMIT;
      if (it_dual + it_primal >= opt.max_iter) return LIMIT;
      if (iter_cap >= 0 && it_dual - it_start >= iter_cap) return LIMIT;
      if (cutoff < INF && !cost_mod && cur_obj > cutoff - 1e-6 * (1.0 + std::fabs(cutoff))) {
        const double ob = objective();
        cur_obj = ob;
        if (ob > cutoff) return CUTOFF;
      }
      if (!stall_pert && opt.perturb && it_dual - it_start > 2L * m + 200) {  // degenerate stalling: perturb the costs now
        perturb();
        stall_pert = true;
      }

      // pricing: dual steepest edge (hoisted pointers)
      int r = -1;
      double best = 0.0, delta = 0.0;
      {
        const int* const hp = head.data();
        const double* const xp = x.data();
        const double* const lop = lo.data();
        const double* const upp = up.data();
        const double* const dsep = dse.data();
        const double ftol = opt.feas_tol;
        for (int p = 0; p < m; ++p) {
          const int v = hp[p];
          const double xv = xp[v];
          double inf;
          if (xv < lop[v] - ftol) inf = xv - lop[v];
          else if (xv > upp[v] + ftol) inf = xv - upp[v];
          else continue;
          const double score = inf * inf / dsep[p];
          if (score > best) best = score, r = p, delta = inf;
        }
      }
      if (r < 0) {
        if (lu.updates() == 0) return OPTIMAL;
        need = true;
        continue;
      }
      const int pv = head[r];
      posv[r] = 1.0;
      posv_idx.assign(1, r);
      const bool rho_sparse = lu.btran_sparse(posv, posv_idx, true, rho, rho_idx);
      pivot_row(rho_sparse);

      // bound flipping ratio test (hoisted pointers + zero-flip fast path)
      const double sgn = delta > 0 ? 1.0 : -1.0;
      int n_cands = 0;
      int min_ci = -1;
      double min_th = INF;
      {
        const double* const lop = lo.data();
        const double* const upp = up.data();
        const double* const dp = d.data();
        const double* const arp = arow.data();
        const char* const stp = st.data();
        const int* const tp = touched.data();
        Cand* const cp = cands.data();
        const int nt = n_touched;
        for (int i = 0; i < nt; ++i) {
          const int v = tp[i];
          if (lop[v] == upp[v]) continue;
          const double a = sgn * arp[v];
          const char s = static_cast<char>(stp[v] & 3);
          if ((s == LOWER && a > PIV) || (s == UPPER && a < -PIV) || (s == FREE && std::fabs(a) > PIV)) {
            double th = s == FREE ? 0.0 : dp[v] / a;
            if (th < 0.0) th = 0.0;
            const double aa = std::fabs(a);
            if (th < min_th) min_th = th, min_ci = n_cands;
            cp[n_cands++] = {th, aa, v};
          }
        }
      }
      int q = -1;
      double thq = 0.0, aabs = 0.0, slope = std::fabs(delta);
      flips.clear();
      fth.clear();
      if (n_cands > 0) {
        const Cand& mcd = cands[min_ci];
        const double mrange = up[mcd.v] - lo[mcd.v];
        if (mrange == INF || slope - mcd.a * mrange <= opt.feas_tol) {
          // Zero-flip fast path: skip heap construction, scan linearly for largest pivot among ties.
          q = mcd.v, thq = mcd.theta, aabs = mcd.a;
          const double tie = 1e-12 * (1.0 + thq);
          const Cand* const cp = cands.data();
          for (int i = 0; i < n_cands; ++i) {
            if (cp[i].theta - thq <= tie && cp[i].a > aabs) q = cp[i].v, aabs = cp[i].a;
          }
        } else {
          size_t hn = static_cast<size_t>(n_cands);
          std::make_heap(cands.begin(), cands.begin() + hn, cmp);
          while (hn > 0) {
            std::pop_heap(cands.begin(), cands.begin() + hn, cmp);
            const Cand cd = cands[--hn];
            const double range = up[cd.v] - lo[cd.v];
            if (range == INF || slope - cd.a * range <= opt.feas_tol) {
              q = cd.v, thq = cd.theta, aabs = cd.a;
              break;
            }
            slope -= cd.a * range;
            flips.push_back(cd.v);
            fth.push_back(cd.theta);
          }
          if (q >= 0) {
            const double tie = 1e-12 * (1.0 + thq);
            while (hn > 0) {
              std::pop_heap(cands.begin(), cands.begin() + hn, cmp);
              const Cand cd = cands[--hn];
              if (cd.theta - thq > tie) break;
              if (cd.a > aabs) q = cd.v, aabs = cd.a;
            }
            int unflip = -1;
            for (size_t k = flips.size(); k-- > 0;) {
              if (thq - fth[k] > tie) break;
              const double a = std::fabs(arow[flips[k]]);
              if (a > aabs) q = flips[k], aabs = a, unflip = static_cast<int>(k);
            }
            if (unflip >= 0) flips.erase(flips.begin() + unflip);
          }
        }
      }
      if (q < 0) {  // the dual is unbounded along this row
        clear_row();
        clear_vec(rho, rho_idx, rho_sparse);
        if (lu.updates() == 0) return INFEASIBLE;
        need = true;
        continue;
      }
      const double arq = arow[q];

      // entering column (hypersparse FTRAN)
      load_column_sparse(q);
      bool spike_sparse = false;
      const bool aq_sparse = lu.ftran_sparse(rowv, rowv_idx, true, aq, aq_idx, &spike, &spike_idx, &spike_sparse);
      const double alpha = aq[r];
      if (std::fabs(alpha) < PIV || std::fabs(alpha - arq) > 1e-6 * (1.0 + std::fabs(alpha))) {
        clear_row();
        clear_vec(rho, rho_idx, rho_sparse);
        clear_vec(aq, aq_idx, aq_sparse);
        clear_vec(spike, spike_idx, spike_sparse);
        if (lu.updates() > 0) {
          need = true;
          continue;
        }
        if (++trouble > 50) return FAILED;
        dse[r] *= 1e6;  // look at another row first
        continue;
      }

      // dual step fused with clearing arow and st MARK bits (single pass over touched)
      const double thD = d[q] / arq;
      {
        double* const dp = d.data();
        double* const arp = arow.data();
        char* const stp = st.data();
        const int* const tp = touched.data();
        const int nt = n_touched;
        for (int i = 0; i < nt; ++i) {
          const int v = tp[i];
          dp[v] -= thD * arp[v];
          arp[v] = 0.0;
          stp[v] = static_cast<char>(stp[v] & 3);
        }
        n_touched = 0;
      }
      d[q] = 0.0;
      d[pv] = -thD;

      // bound flips (hypersparse FTRAN)
      if (!flips.empty()) {
        const int g = next_rmark();
        rowv_idx.clear();
        for (int v : flips) {
          double dx;
          if (st[v] == LOWER) dx = up[v] - lo[v], st[v] = UPPER, x[v] = up[v];
          else dx = lo[v] - up[v], st[v] = LOWER, x[v] = lo[v];
          cur_obj += c[v] * dx;
          if (v < m) {
            rowv[v] += dx;
            if (rmark[v] != g) rmark[v] = g, rowv_idx.push_back(v);
          } else {
            for (int k = cptr[v - m]; k < cptr[v - m + 1]; ++k) {
              const int i = cidx[k];
              rowv[i] -= cval[k] * dx;
              if (rmark[i] != g) rmark[i] = g, rowv_idx.push_back(i);
            }
          }
        }
        const bool flip_sparse = lu.ftran_sparse(rowv, rowv_idx, true, posv, posv_idx);
        if (flip_sparse) {
          for (int p : posv_idx) {
            const double dp = posv[p];
            posv[p] = 0.0;
            const int hv = head[p];
            x[hv] += dp;
            cur_obj += c[hv] * dp;
          }
          posv_idx.clear();
        } else {
          for (int p = 0; p < m; ++p) {
            const double dp = posv[p];
            posv[p] = 0.0;
            if (dp != 0.0) {
              const int hv = head[p];
              x[hv] += dp;
              cur_obj += c[hv] * dp;
            }
          }
        }
        n_flip += static_cast<long>(flips.size());
      }

      // primal step (sparse over aq_idx when hypersparse)
      const double bound = delta > 0 ? up[pv] : lo[pv];
      const double thP = (x[pv] - bound) / alpha;
      if (aq_sparse) {
        for (int p : aq_idx) {
          const int hv = head[p];
          const double dx = -thP * aq[p];
          x[hv] += dx;
          cur_obj += c[hv] * dx;
        }
      } else {
        for (int p = 0; p < m; ++p) {
          if (aq[p] != 0.0) {
            const int hv = head[p];
            const double dx = -thP * aq[p];
            x[hv] += dx;
            cur_obj += c[hv] * dx;
          }
        }
      }
      x[q] += thP;
      cur_obj += c[q] * thP;
      cur_obj += c[pv] * (bound - x[pv]);
      x[pv] = bound;

      // dual steepest edge weights (sparse over rho_idx and aq_idx; rho is zeroed by ftran_sparse)
      double wr = 0.0;
      if (rho_sparse) {
        for (int i : rho_idx) wr += rho[i] * rho[i];
      } else {
        for (int i = 0; i < m; ++i) wr += rho[i] * rho[i];
      }
      const bool tau_sparse = lu.ftran_sparse(rho, rho_idx, rho_sparse, tau, tau_idx);
      if (aq_sparse) {
        for (int p : aq_idx) {
          if (p == r) continue;
          const double k = aq[p] / alpha;
          dse[p] = std::max(dse[p] - 2.0 * k * tau[p] + k * k * wr, 1e-4);
        }
      } else {
        for (int p = 0; p < m; ++p) {
          if (p == r || aq[p] == 0.0) continue;
          const double k = aq[p] / alpha;
          dse[p] = std::max(dse[p] - 2.0 * k * tau[p] + k * k * wr, 1e-4);
        }
      }
      dse[r] = std::max(wr / (alpha * alpha), 1e-4);
      clear_vec(tau, tau_idx, tau_sparse);
      clear_vec(aq, aq_idx, aq_sparse);

      st[pv] = delta > 0 ? UPPER : LOWER;
      bpos[pv] = -1;
      st[q] = BASIC;
      bpos[q] = r;
      head[r] = q;
      if (!lu.update(r, spike, alpha, spike_sparse ? &spike_idx : nullptr)) need = true;
      clear_vec(spike, spike_idx, spike_sparse);
      ++it_dual;
    }
  }

  // Primal simplex from the current (primal feasible) basis. The bounds are
  // perturbed against degeneracy; the dual simplex removes what the perturbation leaves.
  int primal() {
    for (int rep = 0; rep < 6; ++rep) {
      int s = primal_core(opt.perturb && rep < 3);
      if (s != OPTIMAL) {
        primal_valid = false;
        dual_valid = false;
        return s;
      }
      if (primal_feasible()) {
        primal_valid = true;
        dual_valid = true;
        return OPTIMAL;
      }
      dual_valid = false;
      primal_valid = false;
      s = dual();
      c = c0;
      cost_mod = false;
      if (s != OPTIMAL) {
        factored = false;
        primal_valid = false;
        dual_valid = false;
        return s;
      }
      compute_dual();
      if (dual_feasible()) {
        primal_valid = true;
        dual_valid = true;
        return OPTIMAL;
      }
    }
    primal_valid = false;
    dual_valid = false;
    return FAILED;
  }

  // Fresh factorization and exact primal and dual values for the current basis.
  bool refresh() {
    if (!refactor()) return false;
    compute_primal();
    compute_dual();
    primal_valid = true;
    dual_valid = false;
    return true;
  }

  // Extracts the simplex tableau row for basic structural column k (0 <= k < nc).
  // Leaves all internal scratch vectors cleanly zeroed.
  bool tableau_row(int k, std::vector<int>& nb_var, std::vector<double>& nb_val, std::vector<char>& nb_st,
                   std::vector<double>& nb_lo, std::vector<double>& nb_up) {
    if (!factored || k < 0 || k >= nc) return false;
    const int r = bpos[m + k];
    if (r < 0) return false;
    posv[r] = 1.0;
    posv_idx.assign(1, r);
    const bool rho_sparse = lu.btran_sparse(posv, posv_idx, true, rho, rho_idx);
    pivot_row(rho_sparse);
    clear_vec(rho, rho_idx, rho_sparse);
    nb_var.clear();
    nb_val.clear();
    nb_st.clear();
    nb_lo.clear();
    nb_up.clear();
    double* const arp = arow.data();
    char* const stp = st.data();
    const int* const tp = touched.data();
    const int nt = n_touched;
    for (int i = 0; i < nt; ++i) {
      const int v = tp[i];
      const double av = arp[v];
      arp[v] = 0.0;
      const char sv = static_cast<char>(stp[v] & 3);
      stp[v] = sv;
      if (std::fabs(av) > 1e-13) {
        nb_var.push_back(v);
        nb_val.push_back(av);
        nb_st.push_back(sv);
        nb_lo.push_back(lo[v]);
        nb_up.push_back(up[v]);
      }
    }
    n_touched = 0;
    return true;
  }

  void row_entries(int i, std::vector<int>& col_idx, std::vector<double>& col_val) const {
    col_idx.clear();
    col_val.clear();
    if (i < 0 || i >= m) return;
    const int beg = rptr[i], end = rptr[i + 1];
    col_idx.reserve(static_cast<size_t>(end - beg));
    col_val.reserve(static_cast<size_t>(end - beg));
    for (int q = beg; q < end; ++q) {
      col_idx.push_back(ridx[q] - m);
      col_val.push_back(rval[q]);
    }
  }

  void fill_snapshot(LpSolver::Snapshot& sd) const {
    sd.owner = this;
    sd.cost_epoch = cost_epoch;
    sd.active_epoch = active_epoch;
    sd.lu = lu;
    sd.head = head;
    sd.bpos = bpos;
    sd.d = d;
    sd.y = y;
    sd.x = x;
    sd.cur_obj = cur_obj;
  }

  bool restore_snapshot(const std::vector<char>& b, const std::vector<double>& w, const LpSolver::Snapshot& sd) {
    if (sd.owner != this || sd.cost_epoch != cost_epoch || sd.active_epoch != active_epoch ||
        static_cast<int>(b.size()) != m + nc)
      return false;
    if (cost_mod) {
      c = c0;
      cost_mod = false;
    }
    st = b;
    if (static_cast<int>(w.size()) == m) dse = w;
    lu = sd.lu;
    head = sd.head;
    bpos = sd.bpos;
    d = sd.d;
    y = sd.y;
    x = sd.x;
    cur_obj = sd.cur_obj;
    factored = true;
    primal_valid = true;
    dual_valid = true;
    sync_bounds_after_restore();
    return true;
  }

  bool is_dual_feasible() const { return dual_feasible(); }
  int active_count() const { return use_active ? static_cast<int>(active_list.size()) : nc; }

 private:
  struct Cand {
    double theta, a;
    int v;
  };

  static void clear_vec(std::vector<double>& v, std::vector<int>& idx, bool sparse) {
    if (sparse) {
      for (int k : idx) v[k] = 0.0;
      idx.clear();
    } else {
      std::fill(v.begin(), v.end(), 0.0);
    }
  }

  int next_rmark() {
    if (++rmark_gen == 0x7fffffff) {
      std::fill(rmark.begin(), rmark.end(), 0);
      rmark_gen = 1;
    }
    return rmark_gen;
  }

  bool primal_feasible() const {
    for (int p = 0; p < m; ++p) {
      const int v = head[p];
      if (x[v] < lo[v] - opt.feas_tol || x[v] > up[v] + opt.feas_tol) return false;
    }
    return true;
  }

  bool dual_feasible() const {
    auto ok_v = [&](int v) {
      const char s = st[v];
      if (s == BASIC || lo[v] == up[v]) return true;
      const double viol = s == LOWER ? -d[v] : (s == UPPER ? d[v] : std::fabs(d[v]));
      return viol <= opt.opt_tol;
    };
    if (!use_active) {
      const int nv = m + nc;
      for (int v = 0; v < nv; ++v)
        if (!ok_v(v)) return false;
    } else {
      for (int v = 0; v < m; ++v)
        if (!ok_v(v)) return false;
      for (int j : active_list)
        if (!ok_v(m + j)) return false;
    }
    return true;
  }

  void perturb_bounds() {
    lo_save = lo;
    up_save = up;
    std::uniform_real_distribution<double> u(0.5, 1.0);
    const int nv = m + nc;
    for (int v = 0; v < nv; ++v) {
      if (lo[v] == up[v]) continue;
      if (lo[v] > -INF) lo[v] -= 1e-6 * (1.0 + std::fabs(lo[v])) * u(rng);
      if (up[v] < INF) up[v] += 1e-6 * (1.0 + std::fabs(up[v])) * u(rng);
      if (st[v] == LOWER) x[v] = lo[v];
      else if (st[v] == UPPER) x[v] = up[v];
    }
  }

  void restore_bounds(bool recompute) {
    lo = lo_save;
    up = up_save;
    const int nv = m + nc;
    for (int v = 0; v < nv; ++v) {
      if (st[v] == LOWER) x[v] = lo[v];
      else if (st[v] == UPPER) x[v] = up[v];
    }
    if (recompute) compute_primal();
  }

  int primal_core(bool pert) {
    if (!refactor()) return FAILED;
    compute_primal();
    compute_dual();
    if (dual_feasible()) return OPTIMAL;
    if (pert) {
      perturb_bounds();
      compute_primal();
    }
    const int s = primal_loop();
    if (pert) restore_bounds(s == OPTIMAL);
    return s;
  }

  // Devex pricing, Harris ratio test (hypersparse FTRAN/BTRAN).
  int primal_loop() {
    std::fill(devex.begin(), devex.end(), 1.0);
    bool need = false;
    int trouble = 0;
    const int nv = m + nc;
    for (;;) {
      if (lu.updates() >= opt.refactor || need) {
        if (!refactor()) return FAILED;
        compute_primal();
        compute_dual();
        need = false;
      }
      if ((it_primal & 31) == 0 && (timer.sec() > opt.time_limit || (opt.stop && opt.stop->load(std::memory_order_relaxed))))
        return LIMIT;
      if (it_dual + it_primal >= opt.max_iter) return LIMIT;

      int q = -1;
      double best = 0.0;
      auto check_q = [&](int v) {
        const char s = st[v];
        if (s == BASIC || lo[v] == up[v]) return;
        const double dv = d[v];
        const double viol = s == LOWER ? -dv : (s == UPPER ? dv : std::fabs(dv));
        if (viol <= opt.opt_tol) return;
        const double score = dv * dv / devex[v];
        if (score > best) best = score, q = v;
      };
      if (!use_active) {
        for (int v = 0; v < nv; ++v) check_q(v);
      } else {
        for (int v = 0; v < m; ++v) check_q(v);
        for (int j : active_list) check_q(m + j);
      }
      if (q < 0) {
        if (lu.updates() == 0) return OPTIMAL;
        need = true;
        continue;
      }
      const double sg = d[q] < 0 ? 1.0 : -1.0;  // direction of x_q
      load_column_sparse(q);
      bool spike_sparse = false;
      const bool aq_sparse = lu.ftran_sparse(rowv, rowv_idx, true, aq, aq_idx, &spike, &spike_idx, &spike_sparse);

      // Harris ratio test (sparse over aq_idx when hypersparse)
      double tmax = INF;
      auto pass1 = [&](int p) {
        const double a = sg * aq[p];
        if (std::fabs(a) <= PIV) return;
        const int v = head[p];
        const double slack = a > 0 ? x[v] - lo[v] : up[v] - x[v];
        if (slack == INF) return;
        tmax = std::min(tmax, (std::max(slack, 0.0) + opt.feas_tol) / std::fabs(a));
      };
      if (aq_sparse) {
        for (int p : aq_idx) pass1(p);
      } else {
        for (int p = 0; p < m; ++p) pass1(p);
      }
      int r = -1;
      double amax = 0.0, tr = INF;
      if (tmax < INF) {
        auto pass2 = [&](int p) {
          const double a = sg * aq[p];
          if (std::fabs(a) <= PIV) return;
          const int v = head[p];
          const double slack = a > 0 ? x[v] - lo[v] : up[v] - x[v];
          if (slack == INF) return;
          const double t = std::max(slack, 0.0) / std::fabs(a);
          if (t <= tmax && std::fabs(a) > amax) amax = std::fabs(a), r = p, tr = t;
        };
        if (aq_sparse) {
          for (int p : aq_idx) pass2(p);
        } else {
          for (int p = 0; p < m; ++p) pass2(p);
        }
      }
      const double range = up[q] - lo[q];
      if (range < INF && (r < 0 || range <= tr)) {  // the entering variable reaches its other bound
        if (aq_sparse) {
          for (int p : aq_idx) x[head[p]] -= sg * range * aq[p];
        } else {
          for (int p = 0; p < m; ++p)
            if (aq[p] != 0.0) x[head[p]] -= sg * range * aq[p];
        }
        clear_vec(aq, aq_idx, aq_sparse);
        clear_vec(spike, spike_idx, spike_sparse);
        if (st[q] == LOWER) st[q] = UPPER, x[q] = up[q];
        else st[q] = LOWER, x[q] = lo[q];
        ++n_flip;
        ++it_primal;
        continue;
      }
      if (r < 0) {
        clear_vec(aq, aq_idx, aq_sparse);
        clear_vec(spike, spike_idx, spike_sparse);
        if (lu.updates() == 0) return UNBOUNDED;
        need = true;
        continue;
      }
      const double alpha = aq[r];
      const int pv = head[r];
      if (aq_sparse) {
        for (int p : aq_idx) x[head[p]] -= sg * tr * aq[p];
      } else {
        for (int p = 0; p < m; ++p)
          if (aq[p] != 0.0) x[head[p]] -= sg * tr * aq[p];
      }
      clear_vec(aq, aq_idx, aq_sparse);
      x[q] += sg * tr;
      const bool to_lower = sg * alpha > 0;
      x[pv] = to_lower ? lo[pv] : up[pv];

      // pivot row for the reduced cost and Devex updates
      posv[r] = 1.0;
      posv_idx.assign(1, r);
      const bool rho_sparse = lu.btran_sparse(posv, posv_idx, true, rho, rho_idx);
      pivot_row(rho_sparse);
      clear_vec(rho, rho_idx, rho_sparse);
      if (std::fabs(arow[q] - alpha) > 1e-6 * (1.0 + std::fabs(alpha)) && lu.updates() > 0 && ++trouble < 1000) {
        // inaccurate factors: restore nothing, recompute everything from the new basis
        need = true;
      }
      const double thD = d[q] / alpha;
      const double wq = devex[q];
      {
        double* const dp = d.data();
        double* const dvx = devex.data();
        double* const arp = arow.data();
        char* const stp = st.data();
        const int* const tp = touched.data();
        const int nt = n_touched;
        for (int i = 0; i < nt; ++i) {
          const int v = tp[i];
          const double av = arp[v];
          arp[v] = 0.0;
          stp[v] = static_cast<char>(stp[v] & 3);
          if (v == q) continue;
          dp[v] -= thD * av;
          const double k = av / alpha;
          dvx[v] = std::max(dvx[v], k * k * wq);
        }
        n_touched = 0;
      }
      d[q] = 0.0;
      d[pv] = -thD;
      devex[pv] = std::max(wq / (alpha * alpha), 1.0);
      const bool reset = devex[pv] > 1e8;

      st[pv] = to_lower ? LOWER : UPPER;
      bpos[pv] = -1;
      st[q] = BASIC;
      bpos[q] = r;
      head[r] = q;
      if (reset) std::fill(devex.begin(), devex.end(), 1.0);
      if (!lu.update(r, spike, alpha, spike_sparse ? &spike_idx : nullptr)) need = true;
      clear_vec(spike, spike_idx, spike_sparse);
      ++it_primal;
    }
  }

  const LpOptions& opt;
  const Timer& timer;
  std::mt19937 rng;
  std::vector<int> cptr, cidx, rptr, ridx, lidx;
  std::vector<double> cval, rval;
  BasisLU lu;
  std::vector<double> dse, devex, rho, aq, tau, spike, rowv, posv, arow, fth, lo_save, up_save;
  std::vector<int> posv_idx, rowv_idx, rho_idx, aq_idx, tau_idx, spike_idx, rmark;
  int rmark_gen = 0;
  std::vector<char> row_unit;
  std::vector<int> touched, flips, active_list;
  bool use_active = false;
  int n_touched = 0;
  std::vector<Cand> cands;
  std::vector<std::pair<int, int>> bad;
  const double minus_one = -1.0;
  uint64_t cost_epoch = 0;
  uint64_t active_epoch = 0;
  double cur_obj = 0.0;
  bool factored = false;      // the LU factors belong to the current basis and d is current
  bool cost_mod = false;      // c differs from c0 (perturbation or shifting)
  bool primal_valid = false;  // x_B matches current basis & nonbasic bounds
  bool dual_valid = false;    // nonbasic variables are dual feasible for current bounds & d

  BasisLU::Col column(int v) const {
    if (v < m) return {&lidx[v], &minus_one, 1};
    const int j = v - m;
    return {cidx.data() + cptr[j], cval.data() + cptr[j], cptr[j + 1] - cptr[j]};
  }

  void load_column_sparse(int v) {
    rowv_idx.clear();
    const BasisLU::Col col = column(v);
    for (int k = 0; k < col.len; ++k) {
      const double val = col.val[k];
      if (val != 0.0) {
        rowv[col.idx[k]] = val;
        rowv_idx.push_back(col.idx[k]);
      }
    }
  }

  void set_nonbasic(int v) {
    if (lo[v] > -INF) st[v] = LOWER, x[v] = lo[v];
    else if (up[v] < INF) st[v] = UPPER, x[v] = up[v];
    else st[v] = FREE, x[v] = 0.0;
    bpos[v] = -1;
  }

  bool refactor() {
    for (int attempt = 0; attempt < 4; ++attempt) {
      const int nbad = lu.factor(m, [this](int p) { return column(head[p]); }, &bad);
      ++n_refactor;
      if (nbad == 0) {
        factored = true;
        return true;
      }
      for (const auto& pr : bad) {  // replace dependent columns by logicals
        set_nonbasic(head[pr.first]);
        head[pr.first] = pr.second;
        st[pr.second] = BASIC;
        bpos[pr.second] = pr.first;
        dse[pr.first] = 1.0;
      }
    }
    return false;
  }

  void compute_primal() {
    for (int i = 0; i < m; ++i)
      if (st[i] != BASIC) rowv[i] += x[i];
    auto add_col = [&](int j) {
      const int v = m + j;
      const double xv = x[v];
      if (st[v] == BASIC || xv == 0.0) return;
      for (int q = cptr[j]; q < cptr[j + 1]; ++q) rowv[cidx[q]] -= cval[q] * xv;
    };
    if (!use_active) {
      for (int j = 0; j < nc; ++j) add_col(j);
    } else {
      for (int j : active_list) add_col(j);
    }
    lu.ftran(rowv, posv);  // clears rowv to zero
    for (int p = 0; p < m; ++p) {
      x[head[p]] = posv[p];
      posv[p] = 0.0;
    }
    double ob = 0.0;
    if (!use_active) {
      for (int v = m; v < m + nc; ++v) ob += c[v] * x[v];
    } else {
      for (int j : active_list) ob += c[m + j] * x[m + j];
    }
    cur_obj = ob;
  }

  void compute_dual() {
    for (int p = 0; p < m; ++p) posv[p] = c[head[p]];
    lu.btran(posv, y);  // clears posv to zero
    for (int i = 0; i < m; ++i) d[i] = st[i] == BASIC ? 0.0 : c[i] + y[i];
    auto dual_col = [&](int j) {
      const int v = m + j;
      if (st[v] == BASIC) {
        d[v] = 0.0;
        return;
      }
      double s = c[v];
      for (int q = cptr[j]; q < cptr[j + 1]; ++q) s -= cval[q] * y[cidx[q]];
      d[v] = s;
    };
    if (!use_active) {
      for (int j = 0; j < nc; ++j) dual_col(j);
    } else {
      for (int j : active_list) dual_col(j);
    }
  }

  // Makes the nonbasic variables dual feasible: boxed variables move to the other
  // bound, the others get their cost shifted (removed again before the primal phase).
  // Returns true iff any nonbasic x[v] changed.
  bool fix_dual() {
    bool x_changed = false;
    auto fix_v = [&](int v) {
      const char s = st[v];
      if (s == BASIC || lo[v] == up[v]) return;
      const double dv = d[v];
      if (s == LOWER && dv < -opt.opt_tol) {
        if (up[v] < INF) {
          if (x[v] != up[v]) x_changed = true;
          st[v] = UPPER, x[v] = up[v];
        } else {
          c[v] -= dv, d[v] = 0.0, cost_mod = true;
        }
      } else if (s == UPPER && dv > opt.opt_tol) {
        if (lo[v] > -INF) {
          if (x[v] != lo[v]) x_changed = true;
          st[v] = LOWER, x[v] = lo[v];
        } else {
          c[v] -= dv, d[v] = 0.0, cost_mod = true;
        }
      } else if (s == FREE && std::fabs(dv) > opt.opt_tol) {
        c[v] -= dv, d[v] = 0.0, cost_mod = true;
      }
    };
    if (!use_active) {
      const int nv = m + nc;
      for (int v = 0; v < nv; ++v) fix_v(v);
    } else {
      for (int v = 0; v < m; ++v) fix_v(v);
      for (int j : active_list) fix_v(m + j);
    }
    return x_changed;
  }

  void perturb() {
    double sum = 0.0;
    const int n_act = use_active ? static_cast<int>(active_list.size()) : nc;
    if (!use_active) {
      for (int j = 0; j < nc; ++j) sum += std::fabs(c0[m + j]);
    } else {
      for (int j : active_list) sum += std::fabs(c0[m + j]);
    }
    const double avg = n_act > 0 ? sum / n_act : 1.0;
    std::uniform_real_distribution<double> u(0.5, 1.0);
    auto pert_j = [&](int j) {
      const int v = m + j;
      if (st[v] == BASIC || lo[v] == up[v]) return;
      const double eps = 1e-6 * (std::fabs(c0[v]) + avg) * u(rng);
      if (st[v] == LOWER) c[v] += eps, d[v] += eps;
      else if (st[v] == UPPER) c[v] -= eps, d[v] -= eps;
    };
    if (!use_active) {
      for (int j = 0; j < nc; ++j) pert_j(j);
    } else {
      for (int j : active_list) pert_j(j);
    }
    cost_mod = true;
  }

  // Row r of the tableau for the nonbasic variables: arow[v] = rho' a_v.
  // Hoists raw pointers, uses bit-4 of st[v] as visited mark, and skips rval loads on unit rows.
  void pivot_row(bool rho_sparse) {
    const double* const rhop = rho.data();
    const int* const rp = rptr.data();
    const int* const ri_arr = ridx.data();
    const double* const rv_arr = rval.data();
    const char* const ru = row_unit.data();
    char* const stp = st.data();
    double* const arp = arow.data();
    int* const tp = touched.data();
    int nt = n_touched;

    auto process = [&](int i) {
      const double ri = rhop[i];
      if (std::fabs(ri) < 1e-14) return;
      const char si = stp[i];
      if (si != BASIC) {
        arp[i] = -ri;
        if (!(si & MARK)) {
          stp[i] = static_cast<char>(si | MARK);
          tp[nt++] = i;
        }
      }
      const int beg = rp[i], end = rp[i + 1];
      if (ru[i]) {
        for (int q = beg; q < end; ++q) {
          const int v = ri_arr[q];
          const char sv = stp[v];
          if (sv == BASIC) continue;
          if (!(sv & MARK)) {
            stp[v] = static_cast<char>(sv | MARK);
            tp[nt++] = v;
          }
          arp[v] += ri;
        }
      } else {
        for (int q = beg; q < end; ++q) {
          const int v = ri_arr[q];
          const char sv = stp[v];
          if (sv == BASIC) continue;
          if (!(sv & MARK)) {
            stp[v] = static_cast<char>(sv | MARK);
            tp[nt++] = v;
          }
          arp[v] += ri * rv_arr[q];
        }
      }
    };
    if (rho_sparse) {
      const int* const idxp = rho_idx.data();
      const size_t n = rho_idx.size();
      for (size_t k = 0; k < n; ++k) process(idxp[k]);
    } else {
      for (int i = 0; i < m; ++i) process(i);
    }
    n_touched = nt;
  }

  void clear_row() {
    double* const arp = arow.data();
    char* const stp = st.data();
    const int* const tp = touched.data();
    const int nt = n_touched;
    for (int i = 0; i < nt; ++i) {
      const int v = tp[i];
      arp[v] = 0.0;
      stp[v] = static_cast<char>(stp[v] & 3);
    }
    n_touched = 0;
  }
};

const char* status_name(int s) {
  switch (s) {
    case Simplex::OPTIMAL: return "optimal";
    case Simplex::INFEASIBLE: return "infeasible";
    case Simplex::UNBOUNDED: return "unbounded";
    case Simplex::LIMIT: return "limit";
    default: return "failed";
  }
}

// Fills x-dependent result fields and the infeasibility measures on the full problem.
void finish(const LpProblem& P, const std::vector<double>& y, LpResult& R) {
  const int n = P.n, m = P.m;
  R.row_dual = y;
  R.red_cost.assign(n, 0.0);
  std::vector<double> act(m, 0.0);
  double obj = P.offset, pinf = 0.0, dinf = 0.0;
  for (int j = 0; j < n; ++j) {
    const double xj = R.x[j];
    double dj = P.cost[j];
    for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) {
      dj -= P.cval[q] * y[P.cidx[q]];
      act[P.cidx[q]] += P.cval[q] * xj;
    }
    R.red_cost[j] = dj;
    obj += P.cost[j] * xj;
    pinf = std::max(pinf, std::max(P.lb[j] - xj, xj - P.ub[j]));
    if (P.lb[j] == P.ub[j]) continue;
    const double tol = 1e-7;
    if (xj <= P.lb[j] + tol) dinf = std::max(dinf, -dj);
    else if (xj >= P.ub[j] - tol) dinf = std::max(dinf, dj);
    else dinf = std::max(dinf, std::fabs(dj));
  }
  for (int i = 0; i < m; ++i) {
    pinf = std::max(pinf, std::max(P.rlo[i] - act[i], act[i] - P.rup[i]));
    if (P.rlo[i] == P.rup[i]) continue;
    const double tol = 1e-7;
    if (act[i] <= P.rlo[i] + tol) dinf = std::max(dinf, -y[i]);
    else if (act[i] >= P.rup[i] - tol) dinf = std::max(dinf, y[i]);
    else dinf = std::max(dinf, std::fabs(y[i]));
  }
  R.obj = obj;
  R.max_primal_infeas = std::max(pinf, 0.0);
  R.max_dual_infeas = std::max(dinf, 0.0);
}

}  // namespace

LpProblem LpProblem::from_csr(int n, int m, const double* cost, const double* lb, const double* ub,
                              const double* rlo, const double* rup, const std::int64_t* row_ptr,
                              const std::int32_t* col_idx, const double* val, double offset) {
  LpProblem p;
  p.n = n;
  p.m = m;
  p.offset = offset;
  p.cost.assign(cost, cost + n);
  p.lb.assign(lb, lb + n);
  p.ub.assign(ub, ub + n);
  p.rlo.assign(rlo, rlo + m);
  p.rup.assign(rup, rup + m);
  const std::int64_t nnz = row_ptr[m];
  p.cptr.assign(n + 1, 0);
  for (std::int64_t q = 0; q < nnz; ++q) ++p.cptr[col_idx[q] + 1];
  for (int j = 0; j < n; ++j) p.cptr[j + 1] += p.cptr[j];
  p.cidx.resize(nnz);
  p.cval.resize(nnz);
  std::vector<int> pos(p.cptr.begin(), p.cptr.end() - 1);
  for (int i = 0; i < m; ++i)
    for (std::int64_t q = row_ptr[i]; q < row_ptr[i + 1]; ++q) {
      const int j = col_idx[q];
      p.cidx[pos[j]] = i;
      p.cval[pos[j]++] = val[q];
    }
  return p;
}

// ------------------------------------------------------------------ LpSolver
struct LpSolver::Impl {
  Timer timer;
  LpOptions opt;
  double cs = 1.0, offset = 0.0;
  int m = 0, nc = 0;
  bool first = true;
  double cutoff = INF;
  long iters_base = 0;  // iterations of the simplex objects replaced by add_rows / remove_rows
  // the loaded problem (rows grow with add_rows): row sides, columns with scaled costs
  std::vector<double> rlo, rup, cost, ext_lb, ext_ub;
  std::vector<char> in_work;
  std::vector<int> cptr, cidx;
  std::vector<double> cval;
  std::unique_ptr<Simplex> S;

  Impl(const LpProblem& P, const std::vector<int>& cols, const LpOptions& o)
      : opt(o), offset(P.offset), m(P.m), rlo(P.rlo), rup(P.rup) {
    double cmax = 0.0;
    for (double cj : P.cost) cmax = std::max(cmax, std::fabs(cj));
    cs = cmax > 0.0 ? cmax : 1.0;
    cptr.assign(1, 0);
    auto add = [&](int j) {
      cidx.insert(cidx.end(), P.cidx.begin() + P.cptr[j], P.cidx.begin() + P.cptr[j + 1]);
      cval.insert(cval.end(), P.cval.begin() + P.cptr[j], P.cval.begin() + P.cptr[j + 1]);
      cptr.push_back(static_cast<int>(cidx.size()));
      cost.push_back(P.cost[j] / cs);
      ext_lb.push_back(P.lb[j]);
      ext_ub.push_back(P.ub[j]);
    };
    if (cols.empty())
      for (int j = 0; j < P.n; ++j) add(j);
    else
      for (int j : cols) add(j);
    nc = static_cast<int>(cost.size());
    build(ext_lb, ext_ub, nullptr);
  }

  Impl(const Impl& src, unsigned seed, double time_limit, const std::atomic<bool>* stop)
      : opt(src.opt), cs(src.cs), offset(src.offset), m(src.m), nc(src.nc), cutoff(src.cutoff),
        rlo(src.rlo), rup(src.rup), cost(src.cost), ext_lb(src.ext_lb), ext_ub(src.ext_ub),
        in_work(src.in_work), cptr(src.cptr), cidx(src.cidx), cval(src.cval) {
    opt.seed = seed;
    opt.time_limit = time_limit;
    opt.stop = stop;
    const std::vector<char>& basis = src.S->basis();
    build(ext_lb, ext_ub, src.first ? nullptr : &basis);
    if (!first) S->set_weights(src.S->weights());
  }

  // A new simplex object for the current rows and columns; `basis` (rows, then columns) is taken over if given.
  void build(const std::vector<double>& lb, const std::vector<double>& ub, const std::vector<char>* basis) {
    if (S) iters_base += S->it_dual + S->it_primal;
    S.reset(new Simplex(m, rlo.data(), rup.data(), opt, timer));
    if (in_work.empty()) {
      for (int k = 0; k < nc; ++k)
        S->add_column(cidx.data() + cptr[k], cval.data() + cptr[k], cptr[k + 1] - cptr[k], cost[k], lb[k], ub[k]);
      S->finalize();
    } else {
      for (int k = 0; k < nc; ++k)
        S->add_column(cidx.data() + cptr[k], cval.data() + cptr[k], cptr[k + 1] - cptr[k], cost[k],
                      in_work[k] ? lb[k] : 0.0, in_work[k] ? ub[k] : 0.0);
      S->finalize();
      S->set_active(in_work, lb, ub);
    }
    first = !(basis && S->set_basis(*basis));
  }

  void column_bounds(std::vector<double>& lb, std::vector<double>& ub) const {
    lb = ext_lb;
    ub = ext_ub;
  }
};

LpSolver::LpSolver(const LpProblem& p, const std::vector<int>& cols, const LpOptions& opt)
    : impl_(new Impl(p, cols, opt)) {}
LpSolver::LpSolver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LpSolver::~LpSolver() = default;

std::unique_ptr<LpSolver> LpSolver::clone(unsigned seed, double time_limit, const std::atomic<bool>* stop) const {
  std::unique_ptr<Impl> copy(new Impl(*impl_, seed, time_limit, stop));
  return std::unique_ptr<LpSolver>(new LpSolver(std::move(copy)));
}

LpSolver::Status LpSolver::solve() { return solve(impl_->cutoff, -1); }
void LpSolver::set_cutoff(double cutoff) { impl_->cutoff = cutoff; }
double LpSolver::cutoff() const { return impl_->cutoff; }

// New rows start with their row variable basic, so the old basis stays a basis and stays dual feasible:
// the next solve() is a dual simplex run that only has to repair the rows the current point violates.
void LpSolver::add_rows(const std::vector<int>& row_ptr, const std::vector<int>& col_idx, const std::vector<double>& val,
                        const std::vector<double>& row_lo, const std::vector<double>& row_up) {
  Impl& I = *impl_;
  const int r = static_cast<int>(row_lo.size());
  if (static_cast<int>(row_ptr.size()) != r + 1 || static_cast<int>(row_up.size()) != r || row_ptr.front() != 0 ||
      col_idx.size() != val.size() || row_ptr.back() != static_cast<int>(col_idx.size()))
    throw std::invalid_argument("LpSolver::add_rows: inconsistent arrays");
  for (int i = 0; i < r; ++i)
    if (row_ptr[i] > row_ptr[i + 1]) throw std::invalid_argument("LpSolver::add_rows: row_ptr must not decrease");
  for (int k : col_idx)
    if (k < 0 || k >= I.nc) throw std::invalid_argument("LpSolver::add_rows: column index out of range");
  if (r == 0) return;
  std::vector<double> lb, ub;
  I.column_bounds(lb, ub);
  // merge the new entries into the column-wise matrix (rows inside a column stay sorted: the new rows come last)
  std::vector<int> extra(I.nc, 0);
  for (int k : col_idx) ++extra[k];
  std::vector<int> nptr(I.nc + 1, 0);
  for (int k = 0; k < I.nc; ++k) nptr[k + 1] = nptr[k] + (I.cptr[k + 1] - I.cptr[k]) + extra[k];
  std::vector<int> nidx(nptr.back());
  std::vector<double> nval(nptr.back());
  std::vector<int> pos(I.nc);
  for (int k = 0; k < I.nc; ++k) {
    const int len = I.cptr[k + 1] - I.cptr[k];
    std::copy(I.cidx.begin() + I.cptr[k], I.cidx.begin() + I.cptr[k + 1], nidx.begin() + nptr[k]);
    std::copy(I.cval.begin() + I.cptr[k], I.cval.begin() + I.cptr[k + 1], nval.begin() + nptr[k]);
    pos[k] = nptr[k] + len;
  }
  for (int i = 0; i < r; ++i)
    for (int q = row_ptr[i]; q < row_ptr[i + 1]; ++q) {
      const int k = col_idx[q];
      nidx[pos[k]] = I.m + i;
      nval[pos[k]++] = val[q];
    }
  const std::vector<char>& old = I.S->basis();
  std::vector<char> basis(old.begin(), old.begin() + I.m);
  basis.insert(basis.end(), r, static_cast<char>(Simplex::BASIC));
  basis.insert(basis.end(), old.begin() + I.m, old.end());
  const bool had = !I.first;
  I.cptr.swap(nptr);
  I.cidx.swap(nidx);
  I.cval.swap(nval);
  I.rlo.insert(I.rlo.end(), row_lo.begin(), row_lo.end());
  I.rup.insert(I.rup.end(), row_up.begin(), row_up.end());
  I.m += r;
  I.build(lb, ub, had ? &basis : nullptr);
}

// The basis survives when every removed row has its row variable basic (a row that is not tight); otherwise the
// next solve() starts from scratch.
void LpSolver::remove_rows(const std::vector<int>& rows) {
  Impl& I = *impl_;
  std::vector<char> gone(I.m, 0);
  for (int i : rows) {
    if (i < 0 || i >= I.m) throw std::invalid_argument("LpSolver::remove_rows: row index out of range");
    gone[i] = 1;
  }
  std::vector<int> map(I.m, -1);
  int nm = 0;
  for (int i = 0; i < I.m; ++i)
    if (!gone[i]) map[i] = nm++;
  if (nm == I.m) return;
  std::vector<double> lb, ub;
  I.column_bounds(lb, ub);
  const std::vector<char>& old = I.S->basis();
  std::vector<char> basis;
  bool keep = !I.first;
  for (int i = 0; i < I.m; ++i) {
    if (!gone[i]) basis.push_back(old[i]);
    else if (old[i] != Simplex::BASIC) keep = false;
  }
  basis.insert(basis.end(), old.begin() + I.m, old.end());
  std::vector<int> nptr(1, 0), nidx;
  std::vector<double> nval, nlo, nup;
  for (int k = 0; k < I.nc; ++k) {
    for (int q = I.cptr[k]; q < I.cptr[k + 1]; ++q)
      if (map[I.cidx[q]] >= 0) {
        nidx.push_back(map[I.cidx[q]]);
        nval.push_back(I.cval[q]);
      }
    nptr.push_back(static_cast<int>(nidx.size()));
  }
  for (int i = 0; i < I.m; ++i)
    if (!gone[i]) {
      nlo.push_back(I.rlo[i]);
      nup.push_back(I.rup[i]);
    }
  I.cptr.swap(nptr);
  I.cidx.swap(nidx);
  I.cval.swap(nval);
  I.rlo.swap(nlo);
  I.rup.swap(nup);
  I.m = nm;
  I.build(lb, ub, keep ? &basis : nullptr);
}

void LpSolver::remove_rows(int first_row) {
  if (first_row < 0 || first_row > impl_->m) throw std::invalid_argument("LpSolver::remove_rows: row index out of range");
  std::vector<int> rows;
  for (int i = first_row; i < impl_->m; ++i) rows.push_back(i);
  remove_rows(rows);
}

LpSolver::Status LpSolver::solve(double cutoff, long max_iter) {
  Impl& I = *impl_;
  I.S->cutoff = cutoff < INF ? (cutoff - I.offset) / I.cs : INF;
  I.S->iter_cap = max_iter;
  const int s = I.S->solve(I.first);
  I.first = false;
  return static_cast<Status>(s);
}
void LpSolver::set_col_bounds(int k, double lb, double ub) {
  Impl& I = *impl_;
  I.ext_lb[k] = lb;
  I.ext_ub[k] = ub;
  if (I.in_work.empty() || I.in_work[k]) {
    I.S->set_bounds(I.m + k, lb, ub);
  } else if (lb > 0.0) {
    activate_cols({k});
  }
}
void LpSolver::set_col_cost(int k, double cost) {
  impl_->cost[k] = cost / impl_->cs;
  impl_->S->change_cost(impl_->m + k, cost / impl_->cs);
}
std::vector<char> LpSolver::get_basis() const { return impl_->S->basis(); }
bool LpSolver::set_basis(const std::vector<char>& b) {
  Impl& I = *impl_;
  if (!I.in_work.empty() && static_cast<int>(b.size()) == I.m + I.nc) {
    std::vector<int> need;
    for (int k = 0; k < I.nc; ++k)
      if (!I.in_work[k] && b[I.m + k] != Simplex::LOWER) need.push_back(k);
    if (!need.empty()) activate_cols(need);
  }
  return I.S->set_basis(b);
}
void LpSolver::save_state(State& s) const {
  s.basis = impl_->S->basis();
  s.weights = impl_->S->weights();
  if (impl_->S->can_snapshot()) {
    auto snap = std::make_shared<Snapshot>();
    impl_->S->fill_snapshot(*snap);
    s.snap = std::move(snap);
  } else {
    s.snap.reset();
  }
}
bool LpSolver::restore_state(const State& s) {
  if (s.snap && impl_->S->restore_snapshot(s.basis, s.weights, *s.snap)) return true;
  if (!impl_->S->set_basis(s.basis)) return false;
  impl_->S->set_weights(s.weights);
  return true;
}
double LpSolver::obj() const { return impl_->S->objective() * impl_->cs + impl_->offset; }
int LpSolver::cols() const { return impl_->nc; }
int LpSolver::rows() const { return impl_->m; }
long LpSolver::iterations() const { return impl_->iters_base + impl_->S->it_dual + impl_->S->it_primal; }

void LpSolver::set_working_cols(const std::vector<char>& active) {
  Impl& I = *impl_;
  if (active.empty()) {
    I.in_work.clear();
    I.S->set_active(I.in_work, I.ext_lb, I.ext_ub);
    return;
  }
  I.in_work = active;
  for (int k = 0; k < I.nc; ++k)
    if (I.S->st[I.m + k] == Simplex::BASIC || I.ext_lb[k] > 0.0 || I.S->x[I.m + k] != 0.0) I.in_work[k] = 1;
  I.S->set_active(I.in_work, I.ext_lb, I.ext_ub);
}

void LpSolver::activate_cols(const std::vector<int>& add_cols) {
  Impl& I = *impl_;
  if (I.in_work.empty() || add_cols.empty()) return;
  std::vector<int> added;
  for (int k : add_cols)
    if (k >= 0 && k < I.nc && !I.in_work[k]) {
      I.in_work[k] = 1;
      added.push_back(k);
    }
  if (!added.empty()) I.S->activate_subset(added, I.ext_lb, I.ext_ub, I.in_work);
}

void LpSolver::activate_all_unfixed() {
  Impl& I = *impl_;
  if (I.in_work.empty()) return;
  std::vector<int> added;
  for (int k = 0; k < I.nc; ++k)
    if (!I.in_work[k] && I.ext_ub[k] > I.ext_lb[k]) {
      I.in_work[k] = 1;
      added.push_back(k);
    }
  if (!added.empty()) I.S->activate_subset(added, I.ext_lb, I.ext_ub, I.in_work);
}

bool LpSolver::price_inactive(std::vector<int>& neg_cols, double tol) {
  neg_cols.clear();
  Impl& I = *impl_;
  if (I.in_work.empty()) return true;
  if (!I.S->refresh() || !I.S->is_dual_feasible()) return false;
  const double* const yp = I.S->y.data();
  const double st_tol = std::max(I.opt.opt_tol, tol / std::max(1.0, I.cs));
  std::vector<std::pair<double, int>> neg;
  for (int k = 0; k < I.nc; ++k) {
    if (I.in_work[k] || I.ext_lb[k] >= I.ext_ub[k]) continue;
    double dk = I.cost[k];
    for (int q = I.cptr[k]; q < I.cptr[k + 1]; ++q) dk -= I.cval[q] * yp[I.cidx[q]];
    if (dk < -st_tol) neg.push_back({dk, k});
  }
  std::sort(neg.begin(), neg.end());
  neg_cols.reserve(neg.size());
  for (const auto& pr : neg) neg_cols.push_back(pr.second);
  return true;
}

int LpSolver::active_cols_count() const {
  const Impl& I = *impl_;
  return I.in_work.empty() ? I.nc : I.S->active_count();
}

bool LpSolver::has_inactive_cols() const {
  const Impl& I = *impl_;
  return !I.in_work.empty() && I.S->active_count() < I.nc;
}
void LpSolver::get_x(std::vector<double>& x) const {
  const Impl& I = *impl_;
  x.assign(I.S->x.begin() + I.m, I.S->x.end());
}
void LpSolver::get_row_activity(std::vector<double>& a) const {
  const Impl& I = *impl_;
  a.assign(I.S->x.begin(), I.S->x.begin() + I.m);
}
std::vector<char> LpSolver::row_status() const {
  const Impl& I = *impl_;
  return std::vector<char>(I.S->basis().begin(), I.S->basis().begin() + I.m);
}
void LpSolver::get_red_cost(std::vector<double>& d) const {
  const Impl& I = *impl_;
  d.resize(I.nc);
  for (int k = 0; k < I.nc; ++k) d[k] = I.S->d[I.m + k] * I.cs;
}
bool LpSolver::get_row_dual(std::vector<double>& y) {
  Impl& I = *impl_;
  if (!I.S->refresh()) return false;
  y.resize(I.m);
  for (int i = 0; i < I.m; ++i) y[i] = I.S->y[i] * I.cs;
  return true;
}
bool LpSolver::is_col_basic(int k) const {
  const Impl& I = *impl_;
  return k >= 0 && k < I.nc && I.S->st[I.m + k] == Simplex::BASIC;
}
bool LpSolver::get_tableau_row(int k, std::vector<int>& nb_var, std::vector<double>& nb_val,
                               std::vector<char>& nb_st, std::vector<double>& nb_lo,
                               std::vector<double>& nb_up) {
  return impl_->S->tableau_row(k, nb_var, nb_val, nb_st, nb_lo, nb_up);
}
void LpSolver::get_row(int i, std::vector<int>& col_idx, std::vector<double>& col_val) const {
  impl_->S->row_entries(i, col_idx, col_val);
}

LpResult solve_lp(const LpProblem& P, const LpOptions& opt) {
  Timer timer;
  LpResult R;
  const int n = P.n, m = P.m;
  // costs are scaled so that the largest one is 1: tolerances are then relative
  double cmax = 0.0;
  for (double cj : P.cost) cmax = std::max(cmax, std::fabs(cj));
  const double cs = cmax > 0.0 ? cmax : 1.0;

  // Sifting pays off when columns greatly outnumber rows. With equality rows the restricted
  // problems are harder to keep feasible, so the ratio has to be larger (measured in docs/p2_report.md).
  bool has_eq = false;
  for (int i = 0; i < m && !has_eq; ++i) has_eq = P.rlo[i] == P.rup[i];
  const bool wide = has_eq ? n >= 20 * m : (n >= 5 * m && n >= 2000);
  const bool sift = opt.method == "sifting" || (opt.method == "auto" && wide);
  R.method = sift ? "sifting" : "dual";
  auto S = std::make_unique<Simplex>(m, P.rlo.data(), P.rup.data(), opt, timer);
  std::vector<int> wcol;  // structural column of S -> column of P, or -1 - row for an artificial
  int status;
  long it_dual0 = 0, it_primal0 = 0, refactor0 = 0, flip0 = 0;  // work of an abandoned sifting run

  auto report = [&](int s) {
    R.dual_iters = it_dual0 + S->it_dual;
    R.primal_iters = it_primal0 + S->it_primal;
    R.refactors = refactor0 + S->n_refactor;
    R.flips = flip0 + S->n_flip;
    R.work_cols = S->nc;
    R.runtime = timer.sec();
    if (s == Simplex::LIMIT) R.status = timer.sec() > opt.time_limit ? "time_limit" : "iteration_limit";
    else R.status = status_name(s);
  };

  auto direct = [&]() {
    wcol.clear();
    for (int j = 0; j < n; ++j) {
      S->add_column(P.cidx.data() + P.cptr[j], P.cval.data() + P.cptr[j], P.cptr[j + 1] - P.cptr[j], P.cost[j] / cs,
                    P.lb[j], P.ub[j]);
      wcol.push_back(j);
    }
    S->finalize();
    return S->solve(true);
  };

  if (!sift) {
    status = direct();
  } else {
    // row-wise pattern of the full matrix for the first working set
    std::vector<int> rptr(m + 1, 0), ridx(P.cidx.size());
    for (int i : P.cidx) ++rptr[i + 1];
    for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
    {
      std::vector<int> pos(rptr.begin(), rptr.end() - 1);
      for (int j = 0; j < n; ++j)
        for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) ridx[pos[P.cidx[q]]++] = j;
    }
    std::vector<char> inw(n, 0);
    auto score = [&](int j) { return P.cost[j] / std::max(1, P.cptr[j + 1] - P.cptr[j]); };
    std::vector<int> tmp;
    const int per_row = 5;
    for (int i = 0; i < m; ++i) {
      tmp.assign(ridx.begin() + rptr[i], ridx.begin() + rptr[i + 1]);
      if (static_cast<int>(tmp.size()) > per_row) {
        std::nth_element(tmp.begin(), tmp.begin() + per_row, tmp.end(),
                         [&](int a, int b) { return score(a) < score(b) || (score(a) == score(b) && a < b); });
        tmp.resize(per_row);
      }
      for (int j : tmp) inw[j] = 1;
    }
    for (int j = 0; j < n; ++j)
      if (P.lb[j] != 0.0) inw[j] = 1;  // columns outside the working set must sit at zero
    auto add = [&](int j) {
      S->add_column(P.cidx.data() + P.cptr[j], P.cval.data() + P.cptr[j], P.cptr[j + 1] - P.cptr[j], P.cost[j] / cs,
                   P.lb[j], P.ub[j]);
      wcol.push_back(j);
    };
    for (int j = 0; j < n; ++j)
      if (inw[j]) add(j);
    // artificial columns keep the restricted problem feasible
    double big = 10.0;  // in scaled cost units
    std::vector<int> art;
    for (int i = 0; i < m; ++i) {
      const double coef = P.rlo[i] > 0.0 ? 1.0 : (P.rup[i] < 0.0 ? -1.0 : 0.0);
      if (coef == 0.0) continue;
      art.push_back(m + S->nc);
      S->add_column(&i, &coef, 1, big, 0.0, INF);
      wcol.push_back(-1 - i);
    }
    S->finalize();
    const int cap = opt.sift_cap > 0 ? opt.sift_cap : std::max(2 * m, 500);
    std::vector<std::pair<double, int>> neg;
    status = S->solve(true);
    for (;;) {
      ++R.sift_rounds;
      if (status == Simplex::UNBOUNDED && !art.empty() && big <= 1e9) {
        // an artificial column that is too cheap can open a ray that the real problem does not have
        big *= 100.0;
        for (int v : art) S->set_cost(v, big);
        status = S->solve(true);
        continue;
      }
      if (status != Simplex::OPTIMAL) break;
      // price the columns outside the working set
      neg.clear();
      for (int j = 0; j < n; ++j) {
        if (inw[j]) continue;
        double dj = P.cost[j] / cs;
        for (int q = P.cptr[j]; q < P.cptr[j + 1]; ++q) dj -= P.cval[q] * S->y[P.cidx[q]];
        if (dj < -opt.opt_tol) neg.push_back({dj, j});
      }
      if (neg.empty()) {
        double amax = 0.0;
        for (int v : art) amax = std::max(amax, S->x[v]);
        if (amax <= 1e-7) break;
        if (big > 1e9) {
          status = Simplex::INFEASIBLE;
          break;
        }
        big *= 100.0;  // the penalty was too small: the artificials still carry value
        for (int v : art) S->set_cost(v, big);
        status = S->solve(true);
        continue;
      }
      if (static_cast<int>(neg.size()) > cap) {
        std::nth_element(neg.begin(), neg.begin() + cap, neg.end());
        neg.resize(cap);
      }
      std::sort(neg.begin(), neg.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
      for (const auto& pr : neg) {
        inw[pr.second] = 1;
        add(pr.second);
      }
      S->finalize();
      // The new columns have negative reduced cost. Boxed ones move to their upper bound, which keeps
      // the basis dual feasible, and the dual simplex (bound flipping) repairs primal feasibility.
      status = S->solve(true);
    }
    if (status == Simplex::INFEASIBLE || status == Simplex::UNBOUNDED) {
      // The artificial columns blur these two answers (an infeasible problem with a cost ray looks
      // unbounded). Decide on the full problem without artificials.
      it_dual0 = S->it_dual, it_primal0 = S->it_primal, refactor0 = S->n_refactor, flip0 = S->n_flip;
      S = std::make_unique<Simplex>(m, P.rlo.data(), P.rup.data(), opt, timer);
      status = direct();
    }
  }

  report(status);
  if (status == Simplex::OPTIMAL && S->refresh()) {
    R.x.assign(n, 0.0);
    for (int k = 0; k < S->nc; ++k)
      if (wcol[k] >= 0) R.x[wcol[k]] = S->x[m + k];
    std::vector<double> y(m);
    for (int i = 0; i < m; ++i) y[i] = S->y[i] * cs;
    finish(P, y, R);
  }
  R.runtime = timer.sec();
  return R;
}

}  // namespace scsp
