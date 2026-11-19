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
  enum : char { BASIC = 0, LOWER = 1, UPPER = 2, FREE = 3 };
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
  }

  // Call after adding columns and before solving.
  void finalize() {
    rptr.assign(m + 1, 0);
    for (int i : cidx) ++rptr[i + 1];
    for (int i = 0; i < m; ++i) rptr[i + 1] += rptr[i];
    ridx.resize(cidx.size());
    rval.resize(cidx.size());
    std::vector<int> pos(rptr.begin(), rptr.end() - 1);
    for (int j = 0; j < nc; ++j)
      for (int q = cptr[j]; q < cptr[j + 1]; ++q) {
        const int i = cidx[q];
        ridx[pos[i]] = j;
        rval[pos[i]++] = cval[q];
      }
    arow.assign(m + nc, 0.0);
    mark.assign(m + nc, 0);
    devex.assign(m + nc, 1.0);
  }

  void set_cost(int v, double cost) { c[v] = c0[v] = cost; }

  // Dual simplex with perturbed costs, then the primal simplex with the true costs.
  int solve(bool pert = true) {
    if (!factored || pert || lu.updates() > opt.refactor / 2) {
      if (!refactor()) return FAILED;
      compute_dual();
    }
    fix_dual();
    if (pert && opt.perturb) perturb();
    int s = dual();
    if (cost_mod) {  // perturbed or shifted costs: back to the true ones, the primal simplex cleans up
      c = c0;
      cost_mod = false;
      factored = false;
      if (s == OPTIMAL) s = primal();
    }
    return s;
  }

  // ---- incremental use (branch and bound) ----
  void set_bounds(int v, double l, double u) {
    lo[v] = l;
    up[v] = u;
    if (st[v] == BASIC) return;
    if (l == u || st[v] == LOWER) {
      if (l > -INF) st[v] = LOWER, x[v] = l;
      else if (u < INF) st[v] = UPPER, x[v] = u;
      else st[v] = FREE, x[v] = 0.0;
    } else if (st[v] == UPPER) {
      if (u < INF) x[v] = u;
      else if (l > -INF) st[v] = LOWER, x[v] = l;
      else st[v] = FREE, x[v] = 0.0;
    }
  }

  void change_cost(int v, double cost) {
    c[v] = c0[v] = cost;
    factored = false;  // the reduced costs have to be recomputed
  }

  const std::vector<char>& basis() const { return st; }

  bool set_basis(const std::vector<char>& b) {
    if (static_cast<int>(b.size()) != m + nc) return false;
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
    return true;
  }

  // The dual steepest edge weights belong to the basis: with them a restored basis prices as well as before.
  const std::vector<double>& weights() const { return dse; }
  void set_weights(const std::vector<double>& w) {
    if (static_cast<int>(w.size()) == m) dse = w;
  }

  double objective() const {
    double s = 0.0;
    for (int v = m; v < m + nc; ++v) s += c0[v] * x[v];
    return s;
  }

  // Dual simplex from the current basis (made dual feasible by bound flips and cost shifts).
  int dual() {
    if (!factored || lu.updates() > opt.refactor / 2) {
      if (!refactor()) return FAILED;
      compute_dual();
    }
    fix_dual();
    compute_primal();
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
        need = false;
      }
      if ((it_dual & 31) == 0 && (timer.sec() > opt.time_limit || (opt.stop && opt.stop->load(std::memory_order_relaxed))))
        return LIMIT;
      if (it_dual + it_primal >= opt.max_iter) return LIMIT;
      if (iter_cap >= 0 && it_dual - it_start >= iter_cap) return LIMIT;
      if (cutoff < INF && !cost_mod && (it_dual & 7) == 7) {
        double ob = 0.0;
        for (int v = m; v < m + nc; ++v) ob += c[v] * x[v];
        if (ob > cutoff) return CUTOFF;
      }
      if (!stall_pert && opt.perturb && it_dual - it_start > 2L * m + 200) {  // degenerate stalling: perturb the costs now
        perturb();
        stall_pert = true;
      }

      // pricing: dual steepest edge
      int r = -1;
      double best = 0.0, delta = 0.0;
      for (int p = 0; p < m; ++p) {
        const int v = head[p];
        const double xv = x[v];
        double inf;
        if (xv < lo[v] - opt.feas_tol) inf = xv - lo[v];
        else if (xv > up[v] + opt.feas_tol) inf = xv - up[v];
        else continue;
        const double score = inf * inf / dse[p];
        if (score > best) best = score, r = p, delta = inf;
      }
      if (r < 0) {
        if (lu.updates() == 0) return OPTIMAL;
        need = true;
        continue;
      }
      const int pv = head[r];
      std::fill(posv.begin(), posv.end(), 0.0);
      posv[r] = 1.0;
      lu.btran(posv, rho);
      pivot_row();

      // bound flipping ratio test
      const double sgn = delta > 0 ? 1.0 : -1.0;
      cands.clear();
      for (int v : touched) {
        if (lo[v] == up[v]) continue;
        const double a = sgn * arow[v];
        const char s = st[v];
        if ((s == LOWER && a > PIV) || (s == UPPER && a < -PIV) || (s == FREE && std::fabs(a) > PIV)) {
          double th = s == FREE ? 0.0 : d[v] / a;
          if (th < 0.0) th = 0.0;
          cands.push_back({th, std::fabs(a), v});
        }
      }
      int q = -1;
      double thq = 0.0, aabs = 0.0, slope = std::fabs(delta);
      flips.clear();
      fth.clear();
      size_t hn = cands.size();
      std::make_heap(cands.begin(), cands.end(), cmp);
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
      if (q < 0) {  // the dual is unbounded along this row
        clear_row();
        if (lu.updates() == 0) return INFEASIBLE;
        need = true;
        continue;
      }
      // among candidates with the same ratio prefer the largest pivot element
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
      const double arq = arow[q];

      // entering column
      load_column(q);
      lu.ftran(rowv, aq, &spike);
      const double alpha = aq[r];
      if (std::fabs(alpha) < PIV || std::fabs(alpha - arq) > 1e-6 * (1.0 + std::fabs(alpha))) {
        clear_row();
        if (lu.updates() > 0) {
          need = true;
          continue;
        }
        if (++trouble > 50) return FAILED;
        dse[r] *= 1e6;  // look at another row first
        continue;
      }

      // dual step
      const double thD = d[q] / arq;
      for (int v : touched) d[v] -= thD * arow[v];
      d[q] = 0.0;
      d[pv] = -thD;

      // bound flips
      if (!flips.empty()) {
        std::fill(rowv.begin(), rowv.end(), 0.0);
        for (int v : flips) {
          double dx;
          if (st[v] == LOWER) dx = up[v] - lo[v], st[v] = UPPER, x[v] = up[v];
          else dx = lo[v] - up[v], st[v] = LOWER, x[v] = lo[v];
          if (v < m) rowv[v] += dx;
          else
            for (int k = cptr[v - m]; k < cptr[v - m + 1]; ++k) rowv[cidx[k]] -= cval[k] * dx;
        }
        lu.ftran(rowv, posv);
        for (int p = 0; p < m; ++p) x[head[p]] += posv[p];
        n_flip += static_cast<long>(flips.size());
      }

      // primal step
      const double bound = delta > 0 ? up[pv] : lo[pv];
      const double thP = (x[pv] - bound) / alpha;
      for (int p = 0; p < m; ++p)
        if (aq[p] != 0.0) x[head[p]] -= thP * aq[p];
      x[q] += thP;
      x[pv] = bound;

      // dual steepest edge weights
      double wr = 0.0;
      for (int i = 0; i < m; ++i) wr += rho[i] * rho[i];
      rowv = rho;
      lu.ftran(rowv, tau);
      for (int p = 0; p < m; ++p) {
        if (p == r || aq[p] == 0.0) continue;
        const double k = aq[p] / alpha;
        dse[p] = std::max(dse[p] - 2.0 * k * tau[p] + k * k * wr, 1e-4);
      }
      dse[r] = std::max(wr / (alpha * alpha), 1e-4);

      st[pv] = delta > 0 ? UPPER : LOWER;
      bpos[pv] = -1;
      st[q] = BASIC;
      bpos[q] = r;
      head[r] = q;
      clear_row();
      if (!lu.update(r, spike, alpha)) need = true;
      ++it_dual;
    }
  }

  // Primal simplex from the current (primal feasible) basis. The bounds are
  // perturbed against degeneracy; the dual simplex removes what the perturbation leaves.
  int primal() {
    for (int rep = 0; rep < 6; ++rep) {
      int s = primal_core(opt.perturb && rep < 3);
      if (s != OPTIMAL) return s;
      if (primal_feasible()) return OPTIMAL;
      s = dual();
      c = c0;
      cost_mod = false;
      if (s != OPTIMAL) {
        factored = false;
        return s;
      }
      compute_dual();
      if (dual_feasible()) return OPTIMAL;
    }
    return FAILED;
  }

  // Fresh factorization and exact primal and dual values for the current basis.
  bool refresh() {
    if (!refactor()) return false;
    compute_primal();
    compute_dual();
    return true;
  }

 private:
  struct Cand {
    double theta, a;
    int v;
  };

  bool primal_feasible() const {
    for (int p = 0; p < m; ++p) {
      const int v = head[p];
      if (x[v] < lo[v] - opt.feas_tol || x[v] > up[v] + opt.feas_tol) return false;
    }
    return true;
  }

  bool dual_feasible() const {
    const int nv = m + nc;
    for (int v = 0; v < nv; ++v) {
      const char s = st[v];
      if (s == BASIC || lo[v] == up[v]) continue;
      const double viol = s == LOWER ? -d[v] : (s == UPPER ? d[v] : std::fabs(d[v]));
      if (viol > opt.opt_tol) return false;
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

  // Devex pricing, Harris ratio test.
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
      for (int v = 0; v < nv; ++v) {
        const char s = st[v];
        if (s == BASIC || lo[v] == up[v]) continue;
        const double dv = d[v];
        const double viol = s == LOWER ? -dv : (s == UPPER ? dv : std::fabs(dv));
        if (viol <= opt.opt_tol) continue;
        const double score = dv * dv / devex[v];
        if (score > best) best = score, q = v;
      }
      if (q < 0) {
        if (lu.updates() == 0) return OPTIMAL;
        need = true;
        continue;
      }
      const double sg = d[q] < 0 ? 1.0 : -1.0;  // direction of x_q
      load_column(q);
      lu.ftran(rowv, aq, &spike);

      // Harris ratio test
      double tmax = INF;
      for (int p = 0; p < m; ++p) {
        const double a = sg * aq[p];
        if (std::fabs(a) <= PIV) continue;
        const int v = head[p];
        const double slack = a > 0 ? x[v] - lo[v] : up[v] - x[v];
        if (slack == INF) continue;
        tmax = std::min(tmax, (std::max(slack, 0.0) + opt.feas_tol) / std::fabs(a));
      }
      int r = -1;
      double amax = 0.0, tr = INF;
      if (tmax < INF)
        for (int p = 0; p < m; ++p) {
          const double a = sg * aq[p];
          if (std::fabs(a) <= PIV) continue;
          const int v = head[p];
          const double slack = a > 0 ? x[v] - lo[v] : up[v] - x[v];
          if (slack == INF) continue;
          const double t = std::max(slack, 0.0) / std::fabs(a);
          if (t <= tmax && std::fabs(a) > amax) amax = std::fabs(a), r = p, tr = t;
        }
      const double range = up[q] - lo[q];
      if (range < INF && (r < 0 || range <= tr)) {  // the entering variable reaches its other bound
        for (int p = 0; p < m; ++p)
          if (aq[p] != 0.0) x[head[p]] -= sg * range * aq[p];
        if (st[q] == LOWER) st[q] = UPPER, x[q] = up[q];
        else st[q] = LOWER, x[q] = lo[q];
        ++n_flip;
        ++it_primal;
        continue;
      }
      if (r < 0) {
        if (lu.updates() == 0) return UNBOUNDED;
        need = true;
        continue;
      }
      const double alpha = aq[r];
      const int pv = head[r];
      for (int p = 0; p < m; ++p)
        if (aq[p] != 0.0) x[head[p]] -= sg * tr * aq[p];
      x[q] += sg * tr;
      const bool to_lower = sg * alpha > 0;
      x[pv] = to_lower ? lo[pv] : up[pv];

      // pivot row for the reduced cost and Devex updates
      std::fill(posv.begin(), posv.end(), 0.0);
      posv[r] = 1.0;
      lu.btran(posv, rho);
      pivot_row();
      if (std::fabs(arow[q] - alpha) > 1e-6 * (1.0 + std::fabs(alpha)) && lu.updates() > 0 && ++trouble < 1000) {
        // inaccurate factors: restore nothing, recompute everything from the new basis
        need = true;
      }
      const double thD = d[q] / alpha;
      const double wq = devex[q];
      for (int v : touched) {
        if (v == q) continue;
        d[v] -= thD * arow[v];
        const double k = arow[v] / alpha;
        devex[v] = std::max(devex[v], k * k * wq);
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
      clear_row();
      if (reset) std::fill(devex.begin(), devex.end(), 1.0);
      if (!lu.update(r, spike, alpha)) need = true;
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
  std::vector<char> mark;
  std::vector<int> touched, flips;
  std::vector<Cand> cands;
  std::vector<std::pair<int, int>> bad;
  const double minus_one = -1.0;
  bool factored = false;  // the LU factors belong to the current basis and d is current
  bool cost_mod = false;  // c differs from c0 (perturbation or shifting)

  BasisLU::Col column(int v) const {
    if (v < m) return {&lidx[v], &minus_one, 1};
    const int j = v - m;
    return {cidx.data() + cptr[j], cval.data() + cptr[j], cptr[j + 1] - cptr[j]};
  }

  void load_column(int v) {
    std::fill(rowv.begin(), rowv.end(), 0.0);
    const BasisLU::Col col = column(v);
    for (int k = 0; k < col.len; ++k) rowv[col.idx[k]] = col.val[k];
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
    std::fill(rowv.begin(), rowv.end(), 0.0);
    for (int i = 0; i < m; ++i)
      if (st[i] != BASIC) rowv[i] += x[i];
    for (int j = 0; j < nc; ++j) {
      const int v = m + j;
      const double xv = x[v];
      if (st[v] == BASIC || xv == 0.0) continue;
      for (int q = cptr[j]; q < cptr[j + 1]; ++q) rowv[cidx[q]] -= cval[q] * xv;
    }
    lu.ftran(rowv, posv);
    for (int p = 0; p < m; ++p) x[head[p]] = posv[p];
  }

  void compute_dual() {
    for (int p = 0; p < m; ++p) posv[p] = c[head[p]];
    lu.btran(posv, y);
    for (int i = 0; i < m; ++i) d[i] = st[i] == BASIC ? 0.0 : c[i] + y[i];
    for (int j = 0; j < nc; ++j) {
      const int v = m + j;
      if (st[v] == BASIC) {
        d[v] = 0.0;
        continue;
      }
      double s = c[v];
      for (int q = cptr[j]; q < cptr[j + 1]; ++q) s -= cval[q] * y[cidx[q]];
      d[v] = s;
    }
  }

  // Makes the nonbasic variables dual feasible: boxed variables move to the other
  // bound, the others get their cost shifted (removed again before the primal phase).
  void fix_dual() {
    const int nv = m + nc;
    for (int v = 0; v < nv; ++v) {
      const char s = st[v];
      if (s == BASIC || lo[v] == up[v]) continue;
      const double dv = d[v];
      if (s == LOWER && dv < -opt.opt_tol) {
        if (up[v] < INF) st[v] = UPPER, x[v] = up[v];
        else c[v] -= dv, d[v] = 0.0, cost_mod = true;
      } else if (s == UPPER && dv > opt.opt_tol) {
        if (lo[v] > -INF) st[v] = LOWER, x[v] = lo[v];
        else c[v] -= dv, d[v] = 0.0, cost_mod = true;
      } else if (s == FREE && std::fabs(dv) > opt.opt_tol) {
        c[v] -= dv, d[v] = 0.0, cost_mod = true;
      }
    }
  }

  void perturb() {
    double sum = 0.0;
    for (int j = 0; j < nc; ++j) sum += std::fabs(c0[m + j]);
    const double avg = nc > 0 ? sum / nc : 1.0;
    std::uniform_real_distribution<double> u(0.5, 1.0);
    for (int j = 0; j < nc; ++j) {
      const int v = m + j;
      if (st[v] == BASIC || lo[v] == up[v]) continue;
      const double eps = 1e-6 * (std::fabs(c0[v]) + avg) * u(rng);
      if (st[v] == LOWER) c[v] += eps, d[v] += eps;
      else if (st[v] == UPPER) c[v] -= eps, d[v] -= eps;
    }
    cost_mod = true;
  }

  // Row r of the tableau for the nonbasic variables: arow[v] = rho' a_v.
  void pivot_row() {
    for (int i = 0; i < m; ++i) {
      const double ri = rho[i];
      if (std::fabs(ri) < 1e-14) continue;
      if (st[i] != BASIC) {
        arow[i] = -ri;
        mark[i] = 1;
        touched.push_back(i);
      }
      for (int q = rptr[i]; q < rptr[i + 1]; ++q) {
        const int v = m + ridx[q];
        if (st[v] == BASIC) continue;
        if (!mark[v]) {
          mark[v] = 1;
          touched.push_back(v);
        }
        arow[v] += ri * rval[q];
      }
    }
  }

  void clear_row() {
    for (int v : touched) arow[v] = 0.0, mark[v] = 0;
    touched.clear();
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
  std::vector<double> rlo, rup, cost;
  std::vector<int> cptr, cidx;
  std::vector<double> cval;
  std::unique_ptr<Simplex> S;

  Impl(const LpProblem& P, const std::vector<int>& cols, const LpOptions& o)
      : opt(o), offset(P.offset), m(P.m), rlo(P.rlo), rup(P.rup) {
    double cmax = 0.0;
    for (double cj : P.cost) cmax = std::max(cmax, std::fabs(cj));
    cs = cmax > 0.0 ? cmax : 1.0;
    std::vector<double> lb, ub;
    cptr.assign(1, 0);
    auto add = [&](int j) {
      cidx.insert(cidx.end(), P.cidx.begin() + P.cptr[j], P.cidx.begin() + P.cptr[j + 1]);
      cval.insert(cval.end(), P.cval.begin() + P.cptr[j], P.cval.begin() + P.cptr[j + 1]);
      cptr.push_back(static_cast<int>(cidx.size()));
      cost.push_back(P.cost[j] / cs);
      lb.push_back(P.lb[j]);
      ub.push_back(P.ub[j]);
    };
    if (cols.empty())
      for (int j = 0; j < P.n; ++j) add(j);
    else
      for (int j : cols) add(j);
    nc = static_cast<int>(cost.size());
    build(lb, ub, nullptr);
  }

  Impl(const Impl& src, unsigned seed, double time_limit, const std::atomic<bool>* stop)
      : opt(src.opt), cs(src.cs), offset(src.offset), m(src.m), nc(src.nc), cutoff(src.cutoff),
        rlo(src.rlo), rup(src.rup), cost(src.cost), cptr(src.cptr), cidx(src.cidx), cval(src.cval) {
    opt.seed = seed;
    opt.time_limit = time_limit;
    opt.stop = stop;
    std::vector<double> lb, ub;
    src.column_bounds(lb, ub);
    const std::vector<char>& basis = src.S->basis();
    build(lb, ub, src.first ? nullptr : &basis);
    if (!first) S->set_weights(src.S->weights());
  }

  // A new simplex object for the current rows and columns; `basis` (rows, then columns) is taken over if given.
  void build(const std::vector<double>& lb, const std::vector<double>& ub, const std::vector<char>* basis) {
    if (S) iters_base += S->it_dual + S->it_primal;
    S.reset(new Simplex(m, rlo.data(), rup.data(), opt, timer));
    for (int k = 0; k < nc; ++k)
      S->add_column(cidx.data() + cptr[k], cval.data() + cptr[k], cptr[k + 1] - cptr[k], cost[k], lb[k], ub[k]);
    S->finalize();
    first = !(basis && S->set_basis(*basis));
  }

  void column_bounds(std::vector<double>& lb, std::vector<double>& ub) const {
    lb.assign(S->lo.begin() + m, S->lo.end());
    ub.assign(S->up.begin() + m, S->up.end());
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
void LpSolver::set_col_bounds(int k, double lb, double ub) { impl_->S->set_bounds(impl_->m + k, lb, ub); }
void LpSolver::set_col_cost(int k, double cost) {
  impl_->cost[k] = cost / impl_->cs;
  impl_->S->change_cost(impl_->m + k, cost / impl_->cs);
}
std::vector<char> LpSolver::get_basis() const { return impl_->S->basis(); }
bool LpSolver::set_basis(const std::vector<char>& b) { return impl_->S->set_basis(b); }
void LpSolver::save_state(State& s) const {
  s.basis = impl_->S->basis();
  s.weights = impl_->S->weights();
}
bool LpSolver::restore_state(const State& s) {
  if (!impl_->S->set_basis(s.basis)) return false;
  impl_->S->set_weights(s.weights);
  return true;
}
double LpSolver::obj() const { return impl_->S->objective() * impl_->cs + impl_->offset; }
int LpSolver::cols() const { return impl_->nc; }
int LpSolver::rows() const { return impl_->m; }
long LpSolver::iterations() const { return impl_->iters_base + impl_->S->it_dual + impl_->S->it_primal; }
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
