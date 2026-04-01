# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for tesseract-jax.

Measures execution time for noop and vectoradd_jax Tesseracts:
- Jitted apply (warm cache)
- Dtype casting (float64 -> float32)
- Batching (vmap)
- Reverse-mode AD (vjp)
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import pytest
from conftest import DEFAULT_ARRAY_SIZES, MAX_VMAP_ARRAY_SIZE, create_test_array

from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

# Use vmap_method="auto_experimental" if supported, fall back to default for older versions.
# TODO: Remove once "auto_experimental" is supported on main
_VMAP_KWARGS = (
    {"vmap_method": "auto_experimental"}
    if "vmap_method" in inspect.signature(apply_tesseract).parameters
    else {}
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on --array-sizes."""
    if "array_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--array-sizes", default=None)
        if raw:
            sizes = [int(s.strip()) for s in raw.split(",")]
        else:
            sizes = DEFAULT_ARRAY_SIZES

        ids = [f"{size:,}" for size in sizes]
        metafunc.parametrize("array_size", sizes, ids=ids)


def _make_vectoradd_inputs(a_v, b_v):
    """Create vectoradd_jax inputs from pre-built arrays."""
    return {
        "a": {"v": a_v, "s": jnp.float32(1.0)},
        "b": {"v": b_v, "s": jnp.float32(1.0)},
        "norm_ord": 2,
    }


class TestNoopApi:
    """Benchmarks for noop Tesseract via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, noop_tesseract_api, array_size):
        jax.clear_caches()
        self.tess = noop_tesseract_api
        self.inputs = {"data": create_test_array(array_size)}
        self.inputs_f64 = {"data": create_test_array(array_size, dtype="float64")}
        self.batched_inputs = {
            "data": create_test_array(10 * array_size).reshape(10, array_size)
        }

    def test_noop_api_apply_jit(self, benchmark):
        """Benchmark jitted apply (warm cache) via from_tesseract_api."""
        fn = jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
        fn()
        benchmark(fn)

    def test_noop_api_cast_float64(self, benchmark):
        """Benchmark jitted apply with float64 input (expects float32)."""
        fn = jax.jit(lambda: apply_tesseract(self.tess, self.inputs_f64))
        fn()
        benchmark(fn)

    def test_noop_api_vmap(self, benchmark, array_size):
        """Benchmark vmap (batch_size=10) via from_tesseract_api."""
        if array_size > MAX_VMAP_ARRAY_SIZE:
            pytest.skip(f"array_size {array_size} exceeds vmap limit")
        fn = jax.vmap(
            lambda data: apply_tesseract(self.tess, {"data": data}, **_VMAP_KWARGS)
        )
        benchmark(fn, self.batched_inputs["data"])


class TestVectoraddApi:
    """Benchmarks for vectoradd_jax Tesseract via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, vectoradd_tesseract_api, array_size):
        jax.clear_caches()
        self.tess = vectoradd_tesseract_api
        self.a_v = create_test_array(array_size)
        self.b_v = create_test_array(array_size)

    def test_vectoradd_api_partial_vjp(self, benchmark):
        """Benchmark reverse-mode AD (vjp) of vector_add.result w.r.t. a.v."""
        b_v = self.b_v
        a_v = self.a_v
        cotangent = jnp.ones_like(a_v)

        def fn(a_v):
            out = apply_tesseract(self.tess, _make_vectoradd_inputs(a_v, b_v))
            return out["vector_add"]["result"]

        def do_vjp():
            _, vjp_fn = jax.vjp(fn, a_v)
            return vjp_fn(cotangent)

        benchmark(do_vjp)
