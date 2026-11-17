// General 0-1 path (P3): any mix of covering, partitioning and packing rows plus side
// constraints, solved by LP-based branch and bound (bnb.hpp). Used for the models the
// set covering heuristic cannot take, in particular set partitioning.
#pragma once

#include <vector>

#include "scsp/lp.hpp"
#include "scsp/sc_solver.hpp"

namespace scsp {

// L: every column binary with bounds [0, 1]. `warm`: optional feasible solution (columns at one).
SolveResult solve_bin(const LpProblem& L, const SolveOptions& opt, const std::vector<int>* warm = nullptr);

}  // namespace scsp
