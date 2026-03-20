# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for dtype casting (float64 → float32) in tesseract-jax.

This file is separate because jax_enable_x64 is a global setting that
affects all tests in the process.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from conftest import DEFAULT_ARRAY_SIZES, create_test_array

from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)


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


def test_noop_api_cast_float64(benchmark, noop_tesseract_api, array_size):
    """Benchmark jitted apply with float64 input (expects float32)."""
    inputs = {"data": create_test_array(array_size, dtype=jnp.float64)}
    fn = jax.jit(lambda: apply_tesseract(noop_tesseract_api, inputs))
    fn()
    benchmark(fn)


def test_noop_api_cast_int32(benchmark, noop_tesseract_api, array_size):
    """Benchmark jitted apply with int32 input (expects float32)."""
    inputs = {"data": jnp.ones(array_size, dtype=jnp.int32)}
    fn = jax.jit(lambda: apply_tesseract(noop_tesseract_api, inputs))
    fn()
    benchmark(fn)
