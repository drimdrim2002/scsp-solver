#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "scsp/bin_solver.hpp"
#include "scsp/lp.hpp"
#include "scsp/parallel.hpp"
#include "scsp/presolve.hpp"
#include "scsp/problem.hpp"
#include "scsp/sc_solver.hpp"
#include "scsp/version.hpp"

namespace py = pybind11;

namespace {

using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using LArr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

scsp::Problem make_problem(const DArr& cost, const LArr& row_ptr, const IArr& col_idx, const LArr& side_ptr,
                           const IArr& side_idx, const DArr& side_val, const DArr& side_rhs, double offset) {
  const int n = static_cast<int>(cost.size());
  const int m = static_cast<int>(row_ptr.size()) - 1;
  const int ns = static_cast<int>(side_rhs.size());
  if (m < 0 || side_ptr.size() != ns + 1) throw py::value_error("bad pointer arrays");
  if (row_ptr.at(m) != col_idx.size() || side_ptr.at(ns) != side_idx.size() || side_idx.size() != side_val.size())
    throw py::value_error("index arrays do not match the pointer arrays");
  return scsp::Problem::from_rows(n, cost.data(), m, row_ptr.data(), col_idx.data(), ns, side_ptr.data(),
                                  side_idx.data(), side_val.data(), side_rhs.data(), offset);
}

py::object num(double v) { return std::isnan(v) ? py::object(py::none()) : py::object(py::float_(v)); }

py::dict result_dict(const scsp::SolveResult& r) {
  py::dict d;
  d["status"] = r.status;
  d["cols"] = r.cols;
  d["obj"] = r.status == "no_solution" || r.status == "infeasible" ? py::object(py::none()) : num(r.obj);
  d["bound"] = num(r.bound);
  d["runtime"] = r.runtime;
  py::list tr;
  for (const auto& t : r.trace) tr.append(py::make_tuple(t.t, num(t.obj), num(t.bound), t.kind == 0 ? "sol" : "bound"));
  d["trace"] = tr;
  py::dict st;
  st["red_cols"] = r.red_cols;
  st["red_rows"] = r.red_rows;
  st["fixed_cols"] = r.fixed_cols;
  st["presolve_time"] = r.presolve_time;
  st["sub_iters"] = r.sub_iters;
  st["greedy_calls"] = r.greedy_calls;
  st["ls_moves"] = r.ls_moves;
  st["enum_nodes"] = r.enum_nodes;
  st["passes"] = r.passes;
  st["warm_used"] = r.warm_used;
  st["bnb_status"] = r.bnb_status;
  st["lp_root"] = num(r.lp_root);
  st["lp_root_time"] = r.lp_root_time;
  st["bnb_nodes"] = r.bnb_nodes;
  st["bnb_lp_iters"] = r.bnb_lp_iters;
  st["bnb_strong"] = r.bnb_strong;
  st["bnb_cols"] = r.bnb_cols;
  st["tree_gave_up"] = r.tree_gave_up;
  st["lp_root_cut"] = r.lp_root_cut;
  st["cuts"] = r.cuts;
  st["cut_rounds"] = r.cut_rounds;
  st["cut_time"] = r.cut_time;
  st["node_fixed"] = r.node_fixed;
  st["rins_calls"] = r.rins_calls;
  st["rins_found"] = r.rins_found;
  d["stats"] = st;
  return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "scsp native core (P1: set covering heuristic, P2: LP engine)";
  m.attr("__version__") = scsp::version();
  m.def("version", &scsp::version, "Library version string.");
  m.def("build_info", &scsp::build_info, "Compiler, architecture and OS of this build.");
  m.def("hardware_threads", &scsp::hardware_threads, "Hardware threads seen by std::thread.");
  m.def(
      "parallel_sum",
      [](std::uint64_t n, unsigned threads) {
        py::gil_scoped_release release;
        return scsp::parallel_sum(n, threads);
      },
      py::arg("n"), py::arg("threads") = 4, "Sum 1..n with std::thread workers (threading smoke test).");

  m.def(
      "presolve_sc",
      [](DArr cost, LArr row_ptr, IArr col_idx, LArr side_ptr, IArr side_idx, DArr side_val, DArr side_rhs,
         double offset) {
        const scsp::Problem p = make_problem(cost, row_ptr, col_idx, side_ptr, side_idx, side_val, side_rhs, offset);
        scsp::PresolveResult r;
        {
          py::gil_scoped_release release;
          r = scsp::presolve(p, true);
        }
        py::dict d;
        d["infeasible"] = r.infeasible;
        d["n_cols"] = r.red.n;
        d["n_rows"] = r.red.m;
        d["offset"] = r.red.offset;
        d["col_orig"] = r.col_orig;
        d["fixed_one"] = r.fixed_one;
        d["rep"] = r.rep;
        d["rounds"] = r.rounds;
        d["side_rhs"] = r.red.side_rhs;
        return d;
      },
      "Presolve a set covering model; returns the size of the reduced model and the column maps.");

  m.def(
      "solve_sc",
      [](DArr cost, LArr row_ptr, IArr col_idx, LArr side_ptr, IArr side_idx, DArr side_val, DArr side_rhs,
         double offset, double time_limit, int threads, std::uint64_t seed, double mip_gap, py::object warm,
         bool presolve, bool local_search, bool enumerate, bool bnb, double tree_giveup, int cuts) {
        const scsp::Problem p = make_problem(cost, row_ptr, col_idx, side_ptr, side_idx, side_val, side_rhs, offset);
        scsp::SolveOptions o;
        o.time_limit = time_limit;
        o.threads = threads;
        o.seed = seed;
        o.mip_gap = mip_gap;
        o.presolve = presolve;
        o.local_search = local_search;
        o.enumerate = enumerate;
        o.bnb = bnb;
        o.tree_giveup = tree_giveup;
        o.cuts = cuts;
        std::vector<int> w;
        const bool has_warm = !warm.is_none();
        if (has_warm) {
          w = warm.cast<std::vector<int>>();
          for (int j : w)
            if (j < 0 || j >= p.n) throw py::value_error("warm start index out of range");
        }
        scsp::SolveResult r;
        {
          py::gil_scoped_release release;
          r = scsp::solve_sc(p, o, has_warm ? &w : nullptr);
        }
        return result_dict(r);
      },
      py::arg("cost"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("side_ptr"), py::arg("side_idx"),
      py::arg("side_val"), py::arg("side_rhs"), py::arg("offset") = 0.0, py::arg("time_limit") = 10.0,
      py::arg("threads") = 4, py::arg("seed") = 1, py::arg("mip_gap") = 0.0, py::arg("warm") = py::none(),
      py::arg("presolve") = true, py::arg("local_search") = true, py::arg("enumerate") = true, py::arg("bnb") = true,
      py::arg("tree_giveup") = 3.0, py::arg("cuts") = -1,
      "Set covering heuristic (cover rows + packing side constraints).");

  m.def(
      "solve_bin",
      [](DArr cost, LArr row_ptr, IArr col_idx, DArr val, DArr row_lo, DArr row_up, double offset, double time_limit,
         int threads, std::uint64_t seed, double mip_gap, py::object warm, bool local_search, int cuts) {
        const int n = static_cast<int>(cost.size());
        const int nr = static_cast<int>(row_ptr.size()) - 1;
        if (nr < 0 || row_lo.size() != nr || row_up.size() != nr) throw py::value_error("array sizes do not match");
        if (row_ptr.at(nr) != col_idx.size() || col_idx.size() != val.size())
          throw py::value_error("index arrays do not match the pointer array");
        for (py::ssize_t q = 0; q < col_idx.size(); ++q)
          if (col_idx.at(q) < 0 || col_idx.at(q) >= n) throw py::value_error("column index out of range");
        const std::vector<double> lb(n, 0.0), ub(n, 1.0);
        const scsp::LpProblem p = scsp::LpProblem::from_csr(n, nr, cost.data(), lb.data(), ub.data(), row_lo.data(),
                                                            row_up.data(), row_ptr.data(), col_idx.data(), val.data(), offset);
        scsp::SolveOptions o;
        o.time_limit = time_limit;
        o.threads = threads;
        o.seed = seed;
        o.mip_gap = mip_gap;
        o.local_search = local_search;
        o.cuts = cuts;
        std::vector<int> w;
        const bool has_warm = !warm.is_none();
        if (has_warm) w = warm.cast<std::vector<int>>();
        scsp::SolveResult r;
        {
          py::gil_scoped_release release;
          r = scsp::solve_bin(p, o, has_warm ? &w : nullptr);
        }
        return result_dict(r);
      },
      py::arg("cost"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("val"), py::arg("row_lo"), py::arg("row_up"),
      py::arg("offset") = 0.0, py::arg("time_limit") = 10.0, py::arg("threads") = 4, py::arg("seed") = 1,
      py::arg("mip_gap") = 0.0, py::arg("warm") = py::none(), py::arg("local_search") = true, py::arg("cuts") = -1,
      "General 0-1 path: LP-based branch and bound for models with partitioning or packing rows.");

  py::class_<scsp::LpSolver>(m, "LpSolver",
                             "An LP that stays loaded: change column bounds and solve again from the last basis "
                             "(dual simplex), save and restore a basis, stop at a cutoff.")
      .def(py::init([](DArr cost, LArr row_ptr, IArr col_idx, DArr val, DArr row_lo, DArr row_up, DArr col_lb,
                       DArr col_ub, double offset, unsigned seed, double time_limit) {
             const int n = static_cast<int>(cost.size());
             const int nr = static_cast<int>(row_ptr.size()) - 1;
             if (nr < 0 || row_lo.size() != nr || row_up.size() != nr || col_lb.size() != n || col_ub.size() != n)
               throw py::value_error("array sizes do not match");
             if (row_ptr.at(nr) != col_idx.size() || col_idx.size() != val.size())
               throw py::value_error("index arrays do not match the pointer array");
             for (py::ssize_t q = 0; q < col_idx.size(); ++q)
               if (col_idx.at(q) < 0 || col_idx.at(q) >= n) throw py::value_error("column index out of range");
             const scsp::LpProblem p =
                 scsp::LpProblem::from_csr(n, nr, cost.data(), col_lb.data(), col_ub.data(), row_lo.data(),
                                           row_up.data(), row_ptr.data(), col_idx.data(), val.data(), offset);
             scsp::LpOptions o;
             o.seed = seed;
             o.time_limit = time_limit;
             return new scsp::LpSolver(p, {}, o);
           }),
           py::arg("cost"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("val"), py::arg("row_lo"),
           py::arg("row_up"), py::arg("col_lb"), py::arg("col_ub"), py::arg("offset") = 0.0, py::arg("seed") = 1,
           py::arg("time_limit") = 1e30)
      .def(
          "solve",
          [](scsp::LpSolver& s, long max_iter) {
            scsp::LpSolver::Status st;
            {
              py::gil_scoped_release release;
              st = s.solve(s.cutoff(), max_iter);
            }
            static const char* names[] = {"optimal", "infeasible", "unbounded", "limit", "failed", "cutoff"};
            return std::string(names[st]);
          },
          py::arg("max_iter") = -1,
          "Solve (first call) or re-solve from the last basis. Returns optimal | infeasible | unbounded | limit | "
          "failed | cutoff.")
      .def("set_cutoff", [](scsp::LpSolver& s, py::object z) {
        s.set_cutoff(z.is_none() ? std::numeric_limits<double>::infinity() : z.cast<double>());
      }, py::arg("cutoff"), "Stop with 'cutoff' once the dual objective exceeds this value (None switches it off).")
      .def("set_col_bounds", [](scsp::LpSolver& s, int j, double lb, double ub) {
        if (j < 0 || j >= s.cols()) throw py::index_error("column index out of range");
        if (!(lb <= ub)) throw py::value_error("lower bound above upper bound");
        s.set_col_bounds(j, lb, ub);
      }, py::arg("j"), py::arg("lb"), py::arg("ub"))
      .def("get_basis", [](const scsp::LpSolver& s) {
        const std::vector<char> b = s.get_basis();
        py::array_t<std::int8_t> a(b.size());
        auto w = a.mutable_unchecked<1>();
        for (size_t i = 0; i < b.size(); ++i) w(i) = b[i];
        return a;
      }, "Status of every row, then every column: 0 basic, 1 at lower bound, 2 at upper bound, 3 free.")
      .def("set_basis", [](scsp::LpSolver& s, py::array_t<std::int8_t, py::array::c_style | py::array::forcecast> a) {
        std::vector<char> b(a.size());
        for (py::ssize_t i = 0; i < a.size(); ++i) {
          if (a.at(i) < 0 || a.at(i) > 3) throw py::value_error("basis status out of range");
          b[i] = static_cast<char>(a.at(i));
        }
        if (!s.set_basis(b)) throw py::value_error("not a basis: need one status per row and column and exactly n_rows basic entries");
      }, py::arg("basis"))
      .def("add_rows", [](scsp::LpSolver& s, std::vector<int> row_ptr, std::vector<int> col_idx, std::vector<double> val,
                          std::vector<double> row_lo, std::vector<double> row_up) {
        try {
          s.add_rows(row_ptr, col_idx, val, row_lo, row_up);
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        }
      }, py::arg("row_ptr"), py::arg("col_idx"), py::arg("val"), py::arg("row_lo"), py::arg("row_up"),
           "Append rows (row-wise CSR over the loaded columns); the next solve() continues from the last basis.")
      .def("remove_rows", [](scsp::LpSolver& s, py::object rows) {
        try {
          if (py::isinstance<py::int_>(rows)) s.remove_rows(rows.cast<int>());
          else s.remove_rows(rows.cast<std::vector<int>>());
        } catch (const std::invalid_argument& e) {
          throw py::value_error(e.what());
        }
      }, py::arg("rows"), "Remove the listed rows, or with an int that row and every later one.")
      .def("row_status", [](const scsp::LpSolver& s) {
        const std::vector<char> b = s.row_status();
        py::array_t<std::int8_t> a(b.size());
        auto w = a.mutable_unchecked<1>();
        for (size_t i = 0; i < b.size(); ++i) w(i) = static_cast<std::int8_t>(b[i]);
        return a;
      }, "0 where the row variable is basic (the row is not tight), 1 / 2 at the lower / upper side.")
      .def_property_readonly("row_activity", [](const scsp::LpSolver& s) {
        std::vector<double> v;
        s.get_row_activity(v);
        return py::array_t<double>(v.size(), v.data());
      })
      .def_property_readonly("obj", &scsp::LpSolver::obj)
      .def_property_readonly("x", [](const scsp::LpSolver& s) {
        std::vector<double> v;
        s.get_x(v);
        return py::array_t<double>(v.size(), v.data());
      })
      .def_property_readonly("red_cost", [](const scsp::LpSolver& s) {
        std::vector<double> v;
        s.get_red_cost(v);
        return py::array_t<double>(v.size(), v.data());
      })
      .def("row_dual", [](scsp::LpSolver& s) {
        std::vector<double> v;
        if (!s.get_row_dual(v)) throw std::runtime_error("the basis could not be factorized");
        return py::array_t<double>(v.size(), v.data());
      }, "Row duals of the current basis (refactorizes).")
      .def_property_readonly("iterations", &scsp::LpSolver::iterations)
      .def_property_readonly("n_rows", &scsp::LpSolver::rows)
      .def_property_readonly("n_cols", &scsp::LpSolver::cols);

  m.def(
      "solve_lp",
      [](DArr cost, LArr row_ptr, IArr col_idx, DArr val, DArr row_lo, DArr row_up, DArr col_lb, DArr col_ub,
         double offset, const std::string& method, double time_limit, double feas_tol, double opt_tol, int refactor,
         bool perturb, unsigned seed, int sift_cap) {
        const int n = static_cast<int>(cost.size());
        const int nr = static_cast<int>(row_ptr.size()) - 1;
        if (nr < 0 || row_lo.size() != nr || row_up.size() != nr || col_lb.size() != n || col_ub.size() != n)
          throw py::value_error("array sizes do not match");
        if (row_ptr.at(nr) != col_idx.size() || col_idx.size() != val.size())
          throw py::value_error("index arrays do not match the pointer array");
        for (py::ssize_t q = 0; q < col_idx.size(); ++q)
          if (col_idx.at(q) < 0 || col_idx.at(q) >= n) throw py::value_error("column index out of range");
        if (method != "auto" && method != "dual" && method != "sifting") throw py::value_error("unknown method");
        const scsp::LpProblem p =
            scsp::LpProblem::from_csr(n, nr, cost.data(), col_lb.data(), col_ub.data(), row_lo.data(), row_up.data(),
                                      row_ptr.data(), col_idx.data(), val.data(), offset);
        scsp::LpOptions o;
        o.method = method;
        o.time_limit = time_limit;
        o.feas_tol = feas_tol;
        o.opt_tol = opt_tol;
        o.refactor = refactor;
        o.perturb = perturb;
        o.seed = seed;
        o.sift_cap = sift_cap;
        scsp::LpResult r;
        {
          py::gil_scoped_release release;
          r = scsp::solve_lp(p, o);
        }
        py::dict d;
        d["status"] = r.status;
        d["obj"] = num(r.obj);
        d["x"] = py::array_t<double>(r.x.size(), r.x.data());
        d["row_dual"] = py::array_t<double>(r.row_dual.size(), r.row_dual.data());
        d["red_cost"] = py::array_t<double>(r.red_cost.size(), r.red_cost.data());
        d["runtime"] = r.runtime;
        py::dict st;
        st["method"] = r.method;
        st["dual_iters"] = r.dual_iters;
        st["primal_iters"] = r.primal_iters;
        st["refactors"] = r.refactors;
        st["flips"] = r.flips;
        st["sift_rounds"] = r.sift_rounds;
        st["work_cols"] = r.work_cols;
        st["max_primal_infeas"] = r.max_primal_infeas;
        st["max_dual_infeas"] = r.max_dual_infeas;
        d["stats"] = st;
        return d;
      },
      py::arg("cost"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("val"), py::arg("row_lo"), py::arg("row_up"),
      py::arg("col_lb"), py::arg("col_ub"), py::arg("offset") = 0.0, py::arg("method") = "auto",
      py::arg("time_limit") = 1e30, py::arg("feas_tol") = 1e-9, py::arg("opt_tol") = 1e-9, py::arg("refactor") = 100,
      py::arg("perturb") = true, py::arg("seed") = 1, py::arg("sift_cap") = 0,
      "LP engine: bounded dual simplex, primal simplex and sifting.");
}
