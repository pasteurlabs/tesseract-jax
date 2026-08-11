"""Custom Hatchling build hook that compiles the native FFI shim.

Hatchling has no native support for building C/C++ extensions, so this hook
compiles ``tesseract_jax/_cuda_shim.cc`` during the wheel build. It is
the moral equivalent of what ``scikit-build-core`` would do, scoped down to
"compile + place the extension into the package tree".

The shim links no CUDA library at build time (it ``dlopen``s the CUDA runtime at
import), so the only build-time inputs are a C++ compiler, pybind11's headers,
and the XLA FFI headers that ship inside jaxlib. The compiled module is placed
next to its sources in ``tesseract_jax`` so the wheel picks it up via
``[tool.hatch.build.targets.wheel].artifacts``. Producing a wheel therefore
yields a platform-specific (non-``py3-none-any``) wheel, as intended.

The GPU-direct feature is optional: if the extension fails to build (no
compiler, headers missing), the package still installs and imports; the
GPU-direct path simply reports itself unavailable and callers fall back to the
host-callback transport. To keep an install from failing on machines that can't
compile it, set ``TESSERACT_JAX_GPU_OPTIONAL=1`` and a build failure is
downgraded to a warning.
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


class CudaShimBuildHook(BuildHookInterface):
    """Compile the native FFI shim before the wheel is assembled."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        # Only relevant for the wheel target; the sdist ships sources instead.
        if self.target_name != "wheel":
            return

        root = Path(self.root)
        source = root / SOURCE
        if not source.is_file():
            raise RuntimeError(f"native shim source not found: {source}")

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        out = root / PACKAGE_DIR / f"_cuda_shim{ext_suffix}"

        try:
            self._compile(root, source, out)
        except Exception as exc:
            if os.environ.get("TESSERACT_JAX_GPU_OPTIONAL"):
                self.app.display_warning(
                    f"native FFI shim build failed ({exc}); GPU-direct dispatch "
                    "will be unavailable. Continuing because "
                    "TESSERACT_JAX_GPU_OPTIONAL is set."
                )
                return
            raise

        # Force-include the freshly built binary in the wheel even though it is
        # git-ignored, and mark the wheel platform-specific.
        rel = out.relative_to(root)
        build_data.setdefault("force_include", {})[str(out)] = str(rel)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        self.app.display_info(f"Built native FFI shim: {rel}")

    def clean(self, versions: list[str]) -> None:
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        out = Path(self.root) / PACKAGE_DIR / f"_cuda_shim{ext_suffix}"
        if out.exists():
            out.unlink()

    def _compile(self, root: Path, source: Path, out: Path) -> None:
        import pybind11

        jaxlib_inc = _jaxlib_include()
        pybind_inc = pybind11.get_include()
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
            f"-I{jaxlib_inc}",
            f"-I{pybind_inc}",
            f"-I{py_inc}",
            str(source),
            "-o",
            str(out),
            "-ldl",
        ]
        self.app.display_info("Compiling native FFI shim: " + " ".join(args))
        subprocess.run(args, check=True, env=os.environ.copy())


def _jaxlib_include() -> str:
    """Locate the XLA FFI headers shipped inside jaxlib."""
    import jaxlib

    inc = Path(jaxlib.__file__).parent / "include"
    if not (inc / "xla" / "ffi" / "api" / "ffi.h").is_file():
        raise RuntimeError(
            f"XLA FFI headers not found under {inc}; is jaxlib installed?"
        )
    return str(inc)
