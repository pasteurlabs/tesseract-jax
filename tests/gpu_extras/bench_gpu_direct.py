# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: host-callback baseline vs GPU-direct (C++ FFI + cuda_ipc).

Measures median per-call ``apply`` latency in a tight serial loop (the
motivating optimization/MCMC workload) across a sweep of array sizes. The
serial loop is exactly the pattern the ring-1 cuda_ipc lifetime supports.

Run:
    python tests/gpu_extras/bench_gpu_direct.py
"""

from __future__ import annotations

import contextlib
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Load the shared serve helper from *this* repo's tests/conftest.py by file path,
# so the import doesn't depend on cwd / which ``tests`` package is importable.
import importlib.util

_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
_spec = importlib.util.spec_from_file_location("_tjax_tests_conftest", _CONFTEST)
_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conftest)
_serve_gpu_tesseract = _conftest.serve_gpu_tesseract

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract


class _TmpPathFactory:
    """Minimal stand-in for pytest's tmp_path_factory for standalone runs."""

    def mktemp(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}_"))


@contextlib.contextmanager
def serve_gpu_tesseract(output_format: str = "json+base64"):
    """Context-manager wrapper around the shared conftest serve helper."""
    gen = _serve_gpu_tesseract(_TmpPathFactory(), output_format=output_format)
    url = next(gen)
    try:
        yield url
    finally:
        gen.close()

SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
N_ITERS = 40
N_WARMUP = 5


def _sync(out):
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None,
        out,
    )


def _time_loop(fn, a, b, n_iters, n_warmup):
    # Warmup also triggers one-time tracing/compilation for jitted callables, so
    # the timed loop measures steady-state per-call latency (the real workload).
    for _ in range(n_warmup):
        _sync(fn(a, b))
    samples = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(a, b)
        _sync(out)
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.median(samples), min(samples)


def main():
    names = ["cpu(host-callback)", "gpu(FFI cuda_ipc)"]

    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]

    # Both paths are the *same* ``apply_tesseract`` -- the only difference is the
    # device the inputs live on, which selects the platform lowering (CPU host
    # callback vs GPU native FFI). That is the whole point of the fold.
    with serve_gpu_tesseract(output_format="json+base64") as url:
        client = Tesseract.from_url(url)

        header = f"{'size':>12} | " + " | ".join(f"{k:>24}" for k in names)
        print(header)
        print("-" * len(header))

        results = {}
        for n in SIZES:
            a = jnp.arange(n, dtype=jnp.float32)
            b = jnp.ones(n, dtype=jnp.float32)
            a_cpu, b_cpu = jax.device_put((a, b), cpu)
            a_gpu, b_gpu = jax.device_put((a, b), gpu)

            fn = jax.jit(lambda x, y: apply_tesseract(client, {"a": x, "b": y})["c"])

            row = {}
            try:
                row[names[0]] = _time_loop(fn, a_cpu, b_cpu, N_ITERS, N_WARMUP)[0]
            except Exception as e:
                row[names[0]] = None
                print(f"  [{names[0]} @ n={n}] failed: {e}", file=sys.stderr)
            try:
                row[names[1]] = _time_loop(fn, a_gpu, b_gpu, N_ITERS, N_WARMUP)[0]
            except Exception as e:
                row[names[1]] = None
                print(f"  [{names[1]} @ n={n}] failed: {e}", file=sys.stderr)

            results[n] = row
            cells = " | ".join(
                (f"{row[k]:>21.3f}ms" if row[k] is not None else f"{'ERR':>24}")
                for k in names
            )
            print(f"{n:>12} | {cells}")

        print("\nspeedup of gpu(FFI) vs cpu(host-callback):")
        for n in SIZES:
            row = results[n]
            c, g = row.get(names[0]), row.get(names[1])
            if c and g:
                print(f"  n={n:>9}: {c / g:.2f}x")


if __name__ == "__main__":
    main()
