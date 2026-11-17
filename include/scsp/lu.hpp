// Sparse LU factorization of a simplex basis with Forrest-Tomlin updates.
//
// Sources (public literature only):
//   Markowitz (1957) pivot count with threshold pivoting; Suhl & Suhl (1990) for
//   the singleton-first ordering; Forrest & Tomlin (1972) for the update.
//
// Index spaces: a basis has m columns ("positions" 0..m-1) and m rows.
//   ftran solves B x = rhs   (rhs in row space, x in position space)
//   btran solves B' z = rhs  (rhs in position space, z in row space)
#pragma once

#include <functional>
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

  // Factorizes the matrix whose column at position p is get(p). Returns the number
  // of positions left without a pivot (0 = regular). For a singular matrix, `bad`
  // receives (position, row) pairs: putting the unit column of `row` at `position`
  // makes the matrix regular.
  int factor(int m, const std::function<Col(int)>& get, std::vector<std::pair<int, int>>* bad);

  // rhs is overwritten with intermediate values. `spike` (optional) receives the
  // partially transformed vector that update() needs.
  void ftran(std::vector<double>& rhs, std::vector<double>& x, std::vector<double>* spike = nullptr) const;
  void btran(std::vector<double>& rhs, std::vector<double>& z) const;

  // Replaces the column at position `pos` by the column whose ftran spike is given.
  // `alpha` is the pivot element (entry `pos` of the transformed column) and is used
  // as an accuracy check. Returns false when the update is unstable; the caller
  // must then refactorize.
  bool update(int pos, const std::vector<double>& spike, double alpha);

  int updates() const { return nupd_; }
  long nnz() const;
  long factor_nnz() const { return factor_nnz_; }

 private:
  struct Ent {
    int c;
    double v;
  };
  int m_ = 0, nupd_ = 0;
  long factor_nnz_ = 0;
  // L: column etas from the factorization. y[Lidx] -= Lval * y[Lrow].
  std::vector<int> Lrow, Lstart, Lidx;
  std::vector<double> Lval;
  // R: row etas from the updates. y[Rrow] -= sum Rval * y[Ridx].
  std::vector<int> Rrow, Rstart, Ridx;
  std::vector<double> Rval;
  // U: one row per pivot, entries in columns (positions) of later pivots.
  std::vector<std::vector<Ent>> Urows;
  std::vector<double> diag;
  std::vector<int> prow, pcol;   // pivot id -> row, position
  std::vector<int> order, opos;  // elimination order of pivot ids and its inverse
  std::vector<int> colpiv;       // position -> pivot id
  std::vector<double> work_;     // position space, all zero between calls
};

}  // namespace scsp
