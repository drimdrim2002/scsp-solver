#include "scsp/parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace scsp {

unsigned hardware_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1u : n;
}

std::uint64_t parallel_sum(std::uint64_t n, unsigned threads) {
  threads = std::max(1u, threads);
  std::vector<std::uint64_t> partial(threads, 0);
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([t, threads, n, &partial]() {
      std::uint64_t s = 0;
      for (std::uint64_t v = t + 1; v <= n; v += threads) s += v;
      partial[t] = s;
    });
  }
  for (auto& th : pool) th.join();
  std::uint64_t total = 0;
  for (std::uint64_t s : partial) total += s;
  return total;
}

}  // namespace scsp
