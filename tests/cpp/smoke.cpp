#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "scsp/parallel.hpp"
#include "scsp/version.hpp"

int main() {
  const std::uint64_t n = 1000000;
  const std::uint64_t got = scsp::parallel_sum(n, 4);
  const std::uint64_t want = n * (n + 1) / 2;
  std::cout << "scsp " << scsp::version() << " (" << scsp::build_info() << "), "
            << scsp::hardware_threads() << " hardware threads, sum=" << got << "\n";
  return got == want ? EXIT_SUCCESS : EXIT_FAILURE;
}
