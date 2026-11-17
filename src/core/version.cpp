#include "scsp/version.hpp"

#include <sstream>
#include <string>

#ifndef SCSP_VERSION
#define SCSP_VERSION "0.0.0"
#endif

namespace scsp {

std::string version() { return SCSP_VERSION; }

std::string build_info() {
  std::ostringstream os;
#if defined(__clang__)
  os << "clang " << __clang_major__ << '.' << __clang_minor__ << '.' << __clang_patchlevel__;
#elif defined(__GNUC__)
  os << "gcc " << __GNUC__ << '.' << __GNUC_MINOR__ << '.' << __GNUC_PATCHLEVEL__;
#else
  os << "unknown-compiler";
#endif
#if defined(__aarch64__) || defined(__arm64__)
  os << " arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  os << " x86_64";
#else
  os << " unknown-arch";
#endif
#if defined(__APPLE__)
  os << " darwin";
#elif defined(__linux__)
  os << " linux";
#endif
  os << " c++" << __cplusplus;
  return os.str();
}

}  // namespace scsp
