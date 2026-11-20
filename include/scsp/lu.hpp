// Sparse LU factorization of a simplex basis with Forrest-Tomlin updates and
// adaptive hypersparse FTRAN/BTRAN.
//
// Sources (public literature only):
//   Markowitz (1957) pivot count with threshold pivoting; Suhl & Suhl (1990) for
//   the singleton-first ordering, row-count buckets, cached column maxima, and flat
//   dynamic storage; Forrest & Tomlin (1972) for the update; Gilbert & Peierls (1988)
//   and Hall & McKinnon (2005) for adaptive hypersparse FTRAN/BTRAN via reachability DFS.
//
// Index spaces: a basis has m columns ("positions" 0..m-1) and m rows.
//   ftran solves B x = rhs   (rhs in row space, x in position space)
//   btran solves B' z = rhs  (rhs in position space, z in row space)
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace scsp {

class BasisLU {
 public:
  struct Col {
    const int* idx;
    const double* val;
    int len;
  };

  int factor(int m, const std::function<Col(int)>& get, std::vector<std::pair<int, int>>* bad);

  // Dense FTRAN/BTRAN. Input vector `rhs` is cleared to zero on return; all m entries
  // of output (`x` or `z`) are written.
  void ftran(std::vector<double>& rhs, std::vector<double>& x, std::vector<double>* spike = nullptr) const;
  void btran(std::vector<double>& rhs, std::vector<double>& z) const;

  // Adaptive hypersparse FTRAN: solves B x = y.
  // Preconditions:
  //   - If `y_sparse` is true, `y` must be zero outside `y_idx`, and `x` (and `*spike` if given)
  //     must be all-zero on entry.
  // Postconditions:
  //   - `y` is cleared to all-zero on return.
  //   - Returns true iff `x` was computed sparsely (`x_idx` covers all nonzeros of `x`).
  bool ftran_sparse(std::vector<double>& y, std::vector<int>& y_idx, bool y_sparse, std::vector<double>& x,
                    std::vector<int>& x_idx, std::vector<double>* spike = nullptr,
                    std::vector<int>* spike_idx = nullptr, bool* spike_sparse = nullptr) const;

  // Adaptive hypersparse BTRAN: solves B' z = w.
  // Preconditions:
  //   - If `w_sparse` is true, `w` must be zero outside `w_idx`, and `z` must be all-zero on entry.
  // Postconditions:
  //   - `w` is cleared to all-zero on return.
  //   - Returns true iff `z` was computed sparsely (`z_idx` covers all nonzeros of `z`).
  bool btran_sparse(std::vector<double>& w, const std::vector<int>& w_idx, bool w_sparse, std::vector<double>& z,
                    std::vector<int>& z_idx) const;

  bool update(int pos, const std::vector<double>& spike, double alpha,
              const std::vector<int>* spike_idx = nullptr);

  int updates() const { return nupd_; }
  long nnz() const {
    const long lsz = lf_ ? static_cast<long>(lf_->Lidx.size()) : 0L;
    return lsz + static_cast<long>(Ridx.size()) + u_nnz_ + m_;
  }
  long factor_nnz() const { return factor_nnz_; }

 private:
  // Immutable between refactors: shared across snapshots (copy-on-write on factor()).
  struct LFactor {
    // L: column etas from the factorization. y[Lidx] -= Lval * y[Lrow].
    std::vector<int> Lrow, Lstart, Lidx, lcol_of_row;
    std::vector<double> Lval;
    // L^T: row-wise transpose of L for hypersparse BTRAN.
    std::vector<int> Lt_start, Lt_idx;
    std::vector<double> Lt_val;
  };

  // Non-copied working buffers and adaptive hypersparsity state.
  struct Scratch {
    std::vector<double> work;
    std::vector<int> mark, dfs_node, dfs_edge, topo, seed;
    int gen = 0;
    // Adaptive hypersparsity backoff (Hall & McKinnon 2005): skip DFS when recent solves were dense.
    int ftran_streak = 0, ftran_skip = 0;
    int btran_streak = 0, btran_skip = 0;
    int upd_streak = 0, upd_skip = 0;

    Scratch() = default;
    Scratch(const Scratch&) {}
    Scratch& operator=(const Scratch&) { return *this; }
    Scratch(Scratch&&) = default;
    Scratch& operator=(Scratch&&) = default;

    void ensure(int m) {
      if (static_cast<int>(work.size()) < m) {
        work.assign(m, 0.0);
        mark.assign(m, 0);
        dfs_node.resize(m);
        dfs_edge.resize(m);
        gen = 0;
      }
    }

    void reset_adaptive() {
      ftran_streak = ftran_skip = 0;
      btran_streak = btran_skip = 0;
      upd_streak = upd_skip = 0;
    }

    int next_gen() {
      if (++gen == 0x7fffffff) {
        std::fill(mark.begin(), mark.end(), 0);
        gen = 1;
      }
      return gen;
    }
  };

  // Persistent factorization buffers (also non-copied across snapshots).
  struct FactorWork {
    std::vector<int> rstart, rlen, rcap, rcol;
    std::vector<double> rval;
    std::vector<int> cstart, clen, ccap, crow;
    std::vector<int> rhead_cnt, rprev, rnext;
    std::vector<char> ractive, cactive;
    std::vector<int> posin, rstack, cstack;
    std::vector<int> ucol_cnt;

    FactorWork() = default;
    FactorWork(const FactorWork&) {}
    FactorWork& operator=(const FactorWork&) { return *this; }
    FactorWork(FactorWork&&) = default;
    FactorWork& operator=(FactorWork&&) = default;
  };

  void u_row_push(int id, int c, double v);
  bool dfs_l(const LFactor& L, const std::vector<int>& seeds, int thresh, int& out_gen) const;
  bool dfs_lt(const LFactor& L, const std::vector<int>& seeds, int thresh) const;
  bool dfs_ucols(const std::vector<int>& seeds, int thresh) const;
  bool dfs_urows(const std::vector<int>& seeds, int thresh) const;

  int m_ = 0, nupd_ = 0;
  long factor_nnz_ = 0, u_nnz_ = 0;
  std::shared_ptr<LFactor> lf_;
  // R: row etas from the updates. y[Rrow] -= sum Rval * y[Ridx].
  std::vector<int> Rrow, Rstart, Ridx;
  std::vector<double> Rval;
  // U by rows (flat dynamic arena, indexed by pivot id): entries in columns of later pivots.
  std::vector<int> u_rstart, u_rlen, u_rcap, u_rcol;
  std::vector<double> u_rval;
  // U by columns (flat array, indexed by column position c): target column pcol[id] of earlier pivots.
  std::vector<int> u_cstart, u_clen, u_ccol;
  std::vector<double> u_cval;
  std::vector<double> diag;
  std::vector<int> prow, pcol;      // pivot id -> row, position
  std::vector<int> rowpiv, colpiv;  // row -> pivot id, position -> pivot id
  std::vector<int> order, opos;     // elimination order of pivot ids and its inverse
  mutable Scratch sc_;
  FactorWork fw_;
};

}  // namespace scsp
