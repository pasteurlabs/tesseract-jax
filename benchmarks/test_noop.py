# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for noop Tesseract via tesseract-jax.

Measures execution time using an identity Tesseract:
- Eager apply (no JIT)
- Jitted apply (warm cache)
- Compilation (jit + lower + compile)
- Batching (vmap)
"""

from __future__ import annotations

import jax
import pytest
from conftest import DEFAULT_ARRAY_SIZES, create_test_array

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


class TestNoopApi:
    """Benchmarks for noop Tesseract via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, noop_tesseract_api, array_size):
        self.tess = noop_tesseract_api
        arr = create_test_array(array_size)
        self.inputs = {"data": arr}
        self.batched_inputs = {
            "data": create_test_array(10 * array_size).reshape(10, array_size)
        }

    def test_noop_api_apply_eager(self, benchmark):
        """Benchmark eager apply (no JIT) via from_tesseract_api."""
        benchmark(apply_tesseract, self.tess, self.inputs)

    def test_noop_api_apply_jit(self, benchmark):
        """Benchmark jitted apply (warm cache) via from_tesseract_api."""
        fn = jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
        fn()
        benchmark(fn)

    def test_noop_api_compile(self, benchmark):
        """Benchmark jit + lower + compile via from_tesseract_api."""

        def do_compile():
            return (
                jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
                .lower()
                .compile()
            )

        benchmark(do_compile)

    def test_noop_api_vmap(self, benchmark):
        """Benchmark vmap (batch_size=10) via from_tesseract_api."""
        fn = jax.vmap(lambda data: apply_tesseract(self.tess, {"data": data}))
        benchmark(fn, self.batched_inputs["data"])


class TestNoopDocker:
    """Benchmarks for noop Tesseract via from_image."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, noop_tesseract_docker, array_size):
        self.tess = noop_tesseract_docker
        self.inputs = {"data": create_test_array(array_size)}

    @pytest.mark.docker
    def test_noop_docker_apply_eager(self, benchmark):
        """Benchmark eager apply (no JIT) via from_image."""
        benchmark(apply_tesseract, self.tess, self.inputs)

    @pytest.mark.docker
    def test_noop_docker_apply_jit(self, benchmark):
        """Benchmark jitted apply (warm cache) via from_image."""
        fn = jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
        fn()
        benchmark(fn)
