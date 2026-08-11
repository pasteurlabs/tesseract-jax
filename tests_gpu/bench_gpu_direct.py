# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: host-callback baseline vs Strategy A (C++) vs Strategy B (Rust).

Measures median per-call ``apply`` latency in a tight serial loop (the
motivating optimization/MCMC workload) across a sweep of array sizes. The
serial loop is exactly the pattern the ring-1 cuda_ipc lifetime supports.

Run:
    python tests_gpu/bench_gpu_direct.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT / "src", _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract
from tesseract_jax_gpu.frontend import apply_gpu_direct
from tests_gpu.serve_util import serve_gpu_tesseract

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
    from tesseract_jax_gpu.frontend import make_gpu_apply

    names = ["baseline(host-callback)", "A(C++ / py cuda_ipc)", "B(Rust native)"]

    # Serve with the default base64 output so the host-callback baseline works
    # unchanged; the GPU-direct strategies opt into cuda_ipc per call themselves
    # (A via a scoped output-format switch, B via an Accept header).
    with serve_gpu_tesseract(output_format="json+base64") as url:
        client = Tesseract.from_url(url)
        # A fresh client for the baseline so its default transport is unaffected.
        baseline_client = Tesseract.from_url(url)

        header = f"{'size':>12} | " + " | ".join(f"{k:>24}" for k in names)
        print(header)
        print("-" * len(header))

        results = {}
        for n in SIZES:
            a = jnp.arange(n, dtype=jnp.float32)
            b = jnp.ones(n, dtype=jnp.float32)

            # Build reusable, jitted callables per size. This traces/compiles the
            # ffi_call once (the real-workload pattern); the timed loop then
            # measures steady-state dispatch latency, not per-call JAX tracing.
            baseline_fn = jax.jit(
                lambda x, y: apply_tesseract(baseline_client, {"a": x, "b": y})["c"]
            )
            apply_a, close_a = make_gpu_apply(client, {"a": a, "b": b}, impl="A")
            apply_b, close_b = make_gpu_apply(client, {"a": a, "b": b}, impl="B")
            jit_a = jax.jit(lambda x, y: apply_a({"a": x, "b": y})["c"])
            jit_b = jax.jit(lambda x, y: apply_b({"a": x, "b": y})["c"])

            variants = {
                names[0]: baseline_fn,
                names[1]: jit_a,
                names[2]: jit_b,
            }

            row = {}
            for name, fn in variants.items():
                try:
                    med, best = _time_loop(fn, a, b, N_ITERS, N_WARMUP)
                    row[name] = med
                except Exception as e:  # keep going; record the failure
                    row[name] = None
                    print(f"  [{name} @ n={n}] failed: {e}", file=sys.stderr)
            close_a()
            close_b()
            results[n] = row
            cells = " | ".join(
                (f"{row[k]:>21.3f}ms" if row[k] is not None else f"{'ERR':>24}")
                for k in variants
            )
            print(f"{n:>12} | {cells}")

        # Speedups vs baseline.
        print("\nspeedup vs baseline (median latency, higher = faster):")
        for n in SIZES:
            row = results[n]
            base = row.get("baseline(host-callback)")
            if not base:
                continue
            parts = []
            for k in ("A(C++ / py cuda_ipc)", "B(Rust native)"):
                if row.get(k):
                    parts.append(f"{k}: {base / row[k]:.2f}x")
            print(f"  n={n:>9}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
