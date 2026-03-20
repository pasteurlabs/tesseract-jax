# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for vectoradd_jax Tesseract via tesseract-jax.

Measures execution time for the vectoradd_jax example Tesseract:
- Eager apply (no JIT)
- Jitted apply (warm cache)
- Compilation (jit + lower + compile)
- Forward-mode AD (jvp)
- Reverse-mode AD (vjp)
- Forward Jacobian (jacfwd)
- Reverse Jacobian (jacrev)
- Batching (vmap)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from conftest import DEFAULT_ARRAY_SIZES, DEFAULT_JAC_ARRAY_SIZES, create_test_array

from tesseract_jax import apply_tesseract


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

    if "jac_array_size" in metafunc.fixturenames:
        ids = [f"{size:,}" for size in DEFAULT_JAC_ARRAY_SIZES]
        metafunc.parametrize("jac_array_size", DEFAULT_JAC_ARRAY_SIZES, ids=ids)


def _make_inputs(a_v, b_v):
    """Create vectoradd_jax inputs from pre-built arrays."""
    return {
        "a": {"v": a_v, "s": jnp.float32(1.0)},
        "b": {"v": b_v, "s": jnp.float32(1.0)},
        "norm_ord": 2,
    }


class TestVectoraddApi:
    """Benchmarks for vectoradd_jax Tesseract via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, vectoradd_tesseract_api, array_size):
        jax.clear_caches()
        self.tess = vectoradd_tesseract_api
        self.a_v = create_test_array(array_size)
        self.b_v = create_test_array(array_size)
        self.inputs = _make_inputs(self.a_v, self.b_v)
        self.tangent = jnp.ones(array_size, dtype=jnp.float32)

    def test_vectoradd_api_apply_eager(self, benchmark):
        """Benchmark eager apply (no JIT) via from_tesseract_api."""
        benchmark(apply_tesseract, self.tess, self.inputs)

    def test_vectoradd_api_apply_jit(self, benchmark):
        """Benchmark jitted apply (warm cache) via from_tesseract_api."""
        fn = jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
        fn()
        benchmark(fn)

    def test_vectoradd_api_compile(self, benchmark):
        """Benchmark jit + lower + compile via from_tesseract_api."""

        def do_compile():
            return (
                jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
                .lower()
                .compile()
            )

        benchmark(do_compile)

    def test_vectoradd_api_jvp(self, benchmark):
        """Benchmark forward-mode AD (jvp) w.r.t. a.v via from_tesseract_api."""
        b_v = self.b_v

        def fn(a_v):
            return apply_tesseract(self.tess, _make_inputs(a_v, b_v))

        benchmark(jax.jvp, fn, (self.a_v,), (self.tangent,))

    def test_vectoradd_api_vjp(self, benchmark):
        """Benchmark reverse-mode AD (vjp) w.r.t. a.v via from_tesseract_api."""
        b_v = self.b_v
        a_v = self.a_v

        def fn(a_v):
            return apply_tesseract(self.tess, _make_inputs(a_v, b_v))

        def do_vjp():
            primals, vjp_fn = jax.vjp(fn, a_v)
            return vjp_fn(primals)

        benchmark(do_vjp)

    def test_vectoradd_api_vmap(self, benchmark):
        """Benchmark vmap (batch_size=10) via from_tesseract_api."""
        a_v_batch = create_test_array(10 * len(self.a_v)).reshape(10, len(self.a_v))
        b_v_batch = create_test_array(10 * len(self.b_v)).reshape(10, len(self.b_v))

        def fn(a_v, b_v):
            return apply_tesseract(self.tess, _make_inputs(a_v, b_v))

        benchmark(jax.vmap(fn), a_v_batch, b_v_batch)


class TestVectoraddApiJacobian:
    """Jacobian benchmarks (smaller sizes) for vectoradd_jax via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, vectoradd_tesseract_api, jac_array_size):
        jax.clear_caches()
        self.tess = vectoradd_tesseract_api
        self.a_v = create_test_array(jac_array_size)
        self.b_v = create_test_array(jac_array_size)

    def test_vectoradd_api_jacrev(self, benchmark):
        """Benchmark reverse Jacobian (jacrev) w.r.t. a.v via from_tesseract_api."""
        b_v = self.b_v

        def fn(a_v):
            return apply_tesseract(self.tess, _make_inputs(a_v, b_v))["vector_add"][
                "result"
            ]

        benchmark(jax.jacrev(fn), self.a_v)

    def test_vectoradd_api_jacfwd(self, benchmark):
        """Benchmark forward Jacobian (jacfwd) w.r.t. a.v via from_tesseract_api."""
        b_v = self.b_v

        def fn(a_v):
            return apply_tesseract(self.tess, _make_inputs(a_v, b_v))["vector_add"][
                "result"
            ]

        benchmark(jax.jacfwd(fn), self.a_v)
