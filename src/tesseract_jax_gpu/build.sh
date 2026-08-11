#!/usr/bin/env bash
# Build the native FFI shim as an importable extension module.
#
# We compile against XLA's header-only FFI API (shipped inside jaxlib) and
# pybind11. We do NOT link the CUDA runtime -- it is dlopen'd at runtime -- so
# the resulting .so carries no compile-time CUDA-version dependency (see
# _cuda_shim.cc). Plain g++ is enough; no CMake / nvcc required.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$HERE/../../.venv/bin/python}"

JAXLIB_INC="$("$PY" -c 'import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), "include"))')"
PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PY_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
EXT_SUFFIX="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

OUT="$HERE/_cuda_shim${EXT_SUFFIX}"

echo "jaxlib include: $JAXLIB_INC"
echo "output:         $OUT"

g++ -O3 -Wall -shared -std=c++17 -fPIC -fvisibility=hidden \
    -I"$JAXLIB_INC" -I"$PYBIND_INC" -I"$PY_INC" \
    "$HERE/_cuda_shim.cc" \
    -o "$OUT" \
    -ldl

echo "built $OUT"
