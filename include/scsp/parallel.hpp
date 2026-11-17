#ifndef SCSP_PARALLEL_HPP
#define SCSP_PARALLEL_HPP

#include <cstdint>

namespace scsp {

// Number of hardware threads reported by the standard library (>= 1).
unsigned hardware_threads();

// Sums 1..n using `threads` std::thread workers. It exists only to prove that
// the threading runtime compiles and links on every target platform (P0).
std::uint64_t parallel_sum(std::uint64_t n, unsigned threads);

}  // namespace scsp

#endif  // SCSP_PARALLEL_HPP
