#ifndef SCSP_VERSION_HPP
#define SCSP_VERSION_HPP

#include <string>

namespace scsp {

// Library version, e.g. "0.0.1".
std::string version();

// Compiler / platform description, used to tell the Mac and Linux builds apart.
std::string build_info();

}  // namespace scsp

#endif  // SCSP_VERSION_HPP
