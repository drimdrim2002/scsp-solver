cmake_minimum_required(VERSION 3.18)
project(scsp VERSION 0.0.1 LANGUAGES CXX)

# C++17, standard headers only (no <bits/stdc++.h>), std::thread instead of OpenMP.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SCSP_BUILD_PYTHON "Build the pybind11 module scsp._core" ON)
option(SCSP_BUILD_TESTS "Build the native C++ smoke test" OFF)
option(SCSP_WERROR "Treat warnings as errors" OFF)

find_package(Threads REQUIRED)

add_library(scsp_core STATIC
  src/core/version.cpp
  src/core/parallel.cpp
  src/core/problem.cpp
  src/core/presolve.cpp
  src/core/sc_solver.cpp
  src/core/lu.cpp
  src/core/lp.cpp
  src/core/cuts.cpp
  src/core/bnb.cpp
  src/core/bin_solver.cpp
)
target_include_directories(scsp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(scsp_core PUBLIC Threads::Threads)
target_compile_definitions(scsp_core PRIVATE SCSP_VERSION="${PROJECT_VERSION}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(scsp_core PRIVATE -Wall -Wextra -Wpedantic)
  if(SCSP_WERROR)
    target_compile_options(scsp_core PRIVATE -Werror)
  endif()
endif()

if(SCSP_BUILD_TESTS)
  enable_testing()
  add_executable(scsp_smoke tests/cpp/smoke.cpp)
  target_link_libraries(scsp_smoke PRIVATE scsp_core)
  add_test(NAME scsp_smoke COMMAND scsp_smoke)
endif()

if(SCSP_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE src/bindings/module.cpp)
  target_link_libraries(_core PRIVATE scsp_core)
  install(TARGETS _core LIBRARY DESTINATION scsp)
endif()
