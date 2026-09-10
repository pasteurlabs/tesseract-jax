"""Custom Hatchling build hook that compiles the native FFI shim.

Hatchling has no native support for building C/C++ extensions, so this hook
compiles ``tesseract_jax/_cuda_shim.cc`` during the wheel build. It is
the moral equivalent of what ``scikit-build-core`` would do, scoped down to
"compile + place the extension into the package tree".

The shim links no CUDA library at build time (it ``dlopen``s the CUDA runtime at
import), so the only build-time inputs are a C++ compiler, nanobind's headers
and bundled sources, and the XLA FFI headers that ship inside jaxlib. The
compiled module is placed next to its sources in ``tesseract_jax`` so the wheel
picks it up via ``[tool.hatch.build.targets.wheel].artifacts``.

The module is built against the CPython stable ABI (``Py_LIMITED_API``) via
nanobind, so it produces a single ``cp312-abi3`` wheel per platform that serves
every supported CPython version, rather than one wheel per version. The wheel is
still platform-specific (not ``py3-none-any``), as intended for a native module.

The GPU-direct feature is optional: if the extension fails to build (no
compiler, headers missing) the package still installs and imports; the
GPU-direct path simply reports itself unavailable and callers fall back to the
host-callback transport. A source install therefore degrades gracefully by
default. To make a build failure fatal -- as CI must, so a broken shim never
ships silently -- set ``TESSERACT_JAX_GPU_REQUIRED=1``.

The shim is only built on platforms it supports (Linux and macOS). On others
(e.g. Windows) compilation is skipped and a pure-Python wheel is produced.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE_DIR = Path("tesseract_jax")
SOURCE = PACKAGE_DIR / "_cuda_shim.cc"

# CPython stable-ABI floor the module targets: 3.12 (0x030C0000). Must match the
# project's minimum supported Python; nanobind adapts its bindings to the Limited
# API when this is defined at compile time.
PY_LIMITED_API = "0x030C0000"

# Platforms whose linker invocation and Python-symbol resolution this hook knows
# how to drive (see _platform_link_args). Others get a pure-Python wheel.
SUPPORTED_PLATFORMS = ("linux", "darwin")


class CudaShimBuildHook(BuildHookInterface):
    """Compile the native FFI shim before the wheel is assembled."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Compile the shim and register it for inclusion in the wheel."""
        # Only relevant for the wheel target; the sdist ships sources instead.
        if self.target_name != "wheel":
            return

        # Explicit opt-out: build the pure-Python (``py3-none-any``) wheel even on
        # a platform that can compile the shim. The release pipeline uses this to
        # publish a universal fallback wheel alongside the native ones, so users
        # on platforms/architectures without a matching native wheel still get an
        # installable (host-callback-only) package. Refuse the contradictory
        # combination rather than silently picking one.
        if os.environ.get("TESSERACT_JAX_PURE_PYTHON"):
            if os.environ.get("TESSERACT_JAX_GPU_REQUIRED"):
                raise RuntimeError(
                    "TESSERACT_JAX_PURE_PYTHON and TESSERACT_JAX_GPU_REQUIRED are "
                    "mutually exclusive: one skips the native shim, the other "
                    "requires it."
                )
            self.app.display_info(
                "TESSERACT_JAX_PURE_PYTHON set; skipping the native FFI shim and "
                "producing a pure-Python wheel."
            )
            return

        # Skip compilation on platforms this hook does not know how to build for
        # (e.g. Windows). The package still installs as a pure-Python wheel; the
        # GPU-direct path reports itself unavailable and callers fall back to the
        # host-callback transport.
        if not sys.platform.startswith(SUPPORTED_PLATFORMS):
            self.app.display_info(
                f"native FFI shim not built on {sys.platform!r} "
                "(unsupported platform); producing a pure-Python wheel."
            )
            return

        root = Path(self.root)
        source = root / SOURCE
        if not source.is_file():
            raise RuntimeError(f"native shim source not found: {source}")

        out = root / PACKAGE_DIR / f"_cuda_shim{_abi3_ext_suffix()}"

        try:
            self._compile(root, source, out)
        except Exception as exc:
            if not os.environ.get("TESSERACT_JAX_GPU_REQUIRED"):
                self.app.display_warning(
                    f"native FFI shim build failed ({exc}); GPU-direct dispatch "
                    "will be unavailable and callers fall back to the "
                    "host-callback transport. Set TESSERACT_JAX_GPU_REQUIRED=1 "
                    "to make this failure fatal (as CI does)."
                )
                return
            raise

        # Force-include the freshly built binary in the wheel even though it is
        # git-ignored, and tag the wheel platform-specific + stable-ABI (abi3):
        # one wheel per platform for all CPython >= the Limited API floor,
        # instead of one per version.
        rel = out.relative_to(root)
        build_data.setdefault("force_include", {})[str(out)] = str(rel)
        build_data["pure_python"] = False
        build_data["tag"] = self._abi3_wheel_tag()

        self.app.display_info(f"Built native FFI shim: {rel} (tag {build_data['tag']})")

    def _abi3_wheel_tag(self) -> str:
        """The wheel tag for the abi3 shim: ``cp3XX-abi3-<platform>``.

        Reuses Hatchling's own platform-tag resolution (which skips the
        many/musl aliases and applies the macOS-compat processing) and overrides
        the interpreter/ABI parts. The interpreter tag is the *Limited API floor*
        (``cp312``), not the building interpreter -- a wheel built on 3.13 still
        installs on 3.12 because it only uses stable-ABI symbols from >= 3.12.
        """
        base = self.build_config.builder.get_best_matching_tag()  # cpXY-cpYY-plat
        platform_tag = base.rsplit("-", 1)[-1]
        return f"{_abi3_cpython_tag()}-abi3-{platform_tag}"

    def clean(self, versions: list[str]) -> None:
        """Remove the compiled shim so a rebuild starts from a clean slate."""
        out = Path(self.root) / PACKAGE_DIR / f"_cuda_shim{_abi3_ext_suffix()}"
        if out.exists():
            out.unlink()

    def _compile(self, root: Path, source: Path, out: Path) -> None:
        jaxlib_inc = _jaxlib_include()
        nb_inc, nb_robin_inc, nb_combined = _nanobind_paths()
        py_inc = sysconfig.get_path("include")
        cxx = os.environ.get("CXX", "c++")

        args = [
            cxx,
            "-O3",
            "-Wall",
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-fvisibility=hidden",
            # Target the CPython stable ABI so the module is abi3: nanobind
            # adapts its bindings to the Limited API when this is defined.
            f"-DPy_LIMITED_API={PY_LIMITED_API}",
            f"-I{jaxlib_inc}",
            f"-I{nb_inc}",
            f"-I{nb_robin_inc}",
            f"-I{py_inc}",
            str(source),
            # nanobind ships its runtime as sources (not header-only); combined
            # mode amalgamates them into one translation unit compiled with the
            # module, so the wheel needs no separate libnanobind.
            str(nb_combined),
            "-o",
            str(out),
            *_platform_link_args(),
        ]
        self.app.display_info("Compiling native FFI shim: " + " ".join(args))
        subprocess.run(args, check=True, env=os.environ.copy())


def _abi3_cpython_tag() -> str:
    """CPython interpreter tag for the Limited API floor, e.g. ``cp312``."""
    hex_ver = int(PY_LIMITED_API, 16)
    major = (hex_ver >> 24) & 0xFF
    minor = (hex_ver >> 16) & 0xFF
    return f"cp{major}{minor}"


def _abi3_ext_suffix() -> str:
    """Extension filename suffix for the abi3 module (e.g. ``.abi3.so``).

    Unlike the version-specific ``EXT_SUFFIX`` (``.cpython-312-...``), the abi3
    suffix carries no interpreter version, so the same file loads on every
    supported CPython. Windows uses ``.pyd``; other platforms ``.so`` (the shim
    is not built on Windows, but the suffix is kept correct for completeness).
    """
    return ".abi3.pyd" if sys.platform == "win32" else ".abi3.so"


def _platform_link_args() -> list[str]:
    """Linker args for building a Python extension module, per platform.

    The extension references Python C-API symbols (``PyBaseObject_Type`` etc.)
    and nanobind's, which live in the interpreter and are only available once the
    module is loaded, not at link time. Each platform expresses "leave these
    undefined, resolve them at load" differently:

    * macOS: ``-undefined dynamic_lookup``. Without it, ``ld`` errors on every
      Python symbol. ``dlopen`` is in libSystem, so no ``-ldl`` is needed.
    * Linux/other ELF: undefined symbols in a shared object are permitted by
      default (resolved against the loading process at ``dlopen`` time), so no
      special flag is required. ``-ldl`` provides ``dlopen`` for the CUDA runtime
      lookup in the shim.
    """
    if sys.platform == "darwin":
        return ["-undefined", "dynamic_lookup"]
    return ["-ldl"]


def _nanobind_paths() -> tuple[str, str, Path]:
    """Locate nanobind's include dir, its bundled robin_map, and combined source.

    nanobind is not header-only: it ships C++ sources that must be compiled with
    the module. Combined mode compiles a single amalgamated ``nb_combined.cpp``.
    """
    import nanobind

    nb_root = Path(nanobind.__file__).parent
    inc = nb_root / "include"
    robin = nb_root / "ext" / "robin_map" / "include"
    combined = nb_root / "src" / "nb_combined.cpp"
    for path in (inc / "nanobind" / "nanobind.h", robin, combined):
        if not path.exists():
            raise RuntimeError(f"nanobind layout unexpected: missing {path}")
    return str(inc), str(robin), combined


def _jaxlib_include() -> str:
    """Locate the XLA FFI headers shipped inside jaxlib."""
    import jaxlib

    inc = Path(jaxlib.__file__).parent / "include"
    if not (inc / "xla" / "ffi" / "api" / "ffi.h").is_file():
        raise RuntimeError(
            f"XLA FFI headers not found under {inc}; is jaxlib installed?"
        )
    return str(inc)
