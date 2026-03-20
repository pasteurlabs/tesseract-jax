# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for noop Tesseract via Docker (from_image).

Separated from test_noop.py so that Docker image builds don't block
API-based benchmarks.
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


@pytest.mark.docker
class TestNoopDocker:
    """Benchmarks for noop Tesseract via from_image."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, noop_tesseract_docker, array_size):
        self.tess = noop_tesseract_docker
        self.inputs = {"data": create_test_array(array_size)}

    def test_noop_docker_apply_eager(self, benchmark):
        """Benchmark eager apply (no JIT) via from_image."""
        benchmark(apply_tesseract, self.tess, self.inputs)

    def test_noop_docker_apply_jit(self, benchmark):
        """Benchmark jitted apply (warm cache) via from_image."""
        fn = jax.jit(lambda: apply_tesseract(self.tess, self.inputs))
        fn()
        benchmark(fn)
