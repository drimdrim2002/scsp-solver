#!/usr/bin/env bash
# Build and test scsp directly on a Linux machine (no docker).
#
#   scripts/linux_native_build.sh              # creates .venv-linux, installs the build tools into it, builds, tests
#   scripts/linux_native_build.sh --no-install # uses the current Python environment as it is (nothing is installed)
#
# Needs: g++ (C++17), python3 >= 3.9 with venv. With --no-install the environment must already have
# scikit-build-core, pybind11, cmake, ninja, numpy, scipy and pytest.
# Not run on the development Mac: written for a Linux x86-64 host (same steps as docker/Dockerfile.linux-amd64).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL=1
[ "${1:-}" = "--no-install" ] && INSTALL=0

if [ "$(uname -s)" != "Linux" ]; then
  echo "this script is meant for Linux (found $(uname -s)); on the Mac use: pip install -e . --no-build-isolation" >&2
  exit 1
fi
command -v g++ >/dev/null || { echo "g++ not found" >&2; exit 1; }

if [ "$INSTALL" = 1 ]; then
  python3 -m venv "$ROOT/.venv-linux"
  # shellcheck disable=SC1091
  . "$ROOT/.venv-linux/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install "scikit-build-core>=0.10" "pybind11>=2.12" cmake ninja numpy scipy pytest
fi
PY="$(command -v python || command -v python3)"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# 1) native library + C++ smoke test, warnings as errors
cmake -S "$ROOT" -B "$BUILD/native" -G Ninja -DSCSP_BUILD_PYTHON=OFF -DSCSP_BUILD_TESTS=ON -DSCSP_WERROR=ON
cmake --build "$BUILD/native"
ctest --test-dir "$BUILD/native" --output-on-failure

# 2) Python wheel through scikit-build-core, installed and imported outside the source tree
"$PY" -m pip wheel "$ROOT" --no-build-isolation --no-deps -w "$BUILD/wheels"
"$PY" -m pip install --force-reinstall --no-deps "$BUILD"/wheels/scsp-*.whl
(cd "$BUILD" && "$PY" -c "import scsp; print(scsp.__file__); print(scsp.version(), '|', scsp.build_info(), '|', scsp.hardware_threads(), scsp.parallel_sum(10**6, 4))")

# 3) tests (those that need Gurobi, HiGHS or the downloaded instances skip themselves)
(cd "$ROOT" && "$PY" -m pytest -q tests)
