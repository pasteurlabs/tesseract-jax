"""Custom Hatchling build hook that compiles the native FFI shim.

Hatchling has no native support for building C/C++ extensions, so this hook
compiles ``src/tesseract_jax_gpu/_cuda_shim.cc`` during the wheel build. It is
the moral equivalent of what ``scikit-build-core`` would do, scoped down to
"compile + place the extension into the package tree".

The shim links no CUDA library at build time (it ``dlopen``s the CUDA runtime at
import), so the only build-time inputs are a C++ compiler, pybind11's headers,
and the XLA FFI headers that ship inside jaxlib. The compiled module is placed
next to its sources in ``src/tesseract_jax_gpu`` so the wheel picks it up via
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
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE_DIR = Path("src") / "tesseract_jax_gpu"
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

        artifacts: list[Path] = []
        optional = bool(os.environ.get("TESSERACT_JAX_GPU_OPTIONAL"))

        # Strategy A: the C++/pybind11 shim.
        try:
            self._compile(root, source, out)
            artifacts.append(out)
        except Exception as exc:
            if not optional:
                raise
            self.app.display_warning(
                f"Strategy A shim build failed ({exc}); GPU-direct dispatch "
                "(A) unavailable. Continuing (TESSERACT_JAX_GPU_OPTIONAL set)."
            )

        # Strategy B: the Rust/PyO3 crate. The module name must be top-level
        # (matches the #[pymodule] name), so it lands in src/.
        try:
            artifacts.append(self._compile_rust(root))
        except Exception as exc:
            if not optional:
                raise
            self.app.display_warning(
                f"Strategy B crate build failed ({exc}); GPU-direct dispatch "
                "(B) unavailable. Continuing (TESSERACT_JAX_GPU_OPTIONAL set)."
            )

        if not artifacts:
            return

        # Force-include the freshly built binaries in the wheel even though they
        # are git-ignored, and mark the wheel platform-specific.
        force_include = build_data.setdefault("force_include", {})
        for path in artifacts:
            force_include[str(path)] = str(path.relative_to(root))
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        self.app.display_info(
            "Built native artifacts: "
            + ", ".join(str(p.relative_to(root)) for p in artifacts)
        )

    def clean(self, versions: list[str]) -> None:
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        out = Path(self.root) / PACKAGE_DIR / f"_cuda_shim{ext_suffix}"
        if out.exists():
            out.unlink()
        rust_so = Path(self.root) / PACKAGE_DIR / "tesseract_jax_gpu_b.so"
        if rust_so.exists():
            rust_so.unlink()

    def _compile_rust(self, root: Path) -> Path:
        """Build the Rust/PyO3 crate for Strategy B and place its .so in src/.

        The crate ``dlopen``s the CUDA runtime and pulls its HTTP/JSON deps via
        Cargo, so the only build-time requirements are a Rust toolchain and the
        jaxlib headers (located by the crate's build.rs from the same Python).
        """
        crate = root / "rust" / "tesseract_jax_gpu_b"
        if not (crate / "Cargo.toml").is_file():
            raise RuntimeError(f"Rust crate not found: {crate}")

        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError("cargo not found on PATH (install a Rust toolchain)")

        env = os.environ.copy()
        # Point both pyo3 and the crate's build.rs at this interpreter.
        env["PYO3_PYTHON"] = sys.executable
        env["TJGPU_PYTHON"] = sys.executable
        self.app.display_info(f"Building Rust crate (Strategy B) in {crate}")
        subprocess.run([cargo, "build", "--release"], check=True, cwd=crate, env=env)

        built = crate / "target" / "release" / "libtesseract_jax_gpu_b.so"
        if not built.is_file():
            raise RuntimeError(f"cargo did not produce {built}")

        out = root / "src" / "tesseract_jax_gpu_b.so"
        shutil.copy2(built, out)
        return out

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
