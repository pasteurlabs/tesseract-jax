# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and CLI options for benchmarks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

# Paths to tesseracts used in benchmarks
NOOP_TESSERACT_PATH = Path(__file__).parent / "tesseract_noop" / "tesseract_api.py"
NOOP_TESSERACT_DIR = NOOP_TESSERACT_PATH.parent

VECTORADD_TESSERACT_PATH = (
    Path(__file__).parent.parent
    / "examples"
    / "simple"
    / "vectoradd_jax"
    / "tesseract_api.py"
)

# Default array sizes when --array-sizes is not specified.
DEFAULT_ARRAY_SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

# Smaller sizes for jacobian benchmarks (jacrev/jacfwd produce NxN matrices).
DEFAULT_JAC_ARRAY_SIZES = [10, 100]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--array-sizes",
        default=None,
        help="Comma-separated array sizes (e.g. '100,10000,1000000')",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "docker: requires Docker")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip Docker-dependent benchmarks when Docker is not available."""
    docker_available = _check_docker()
    if docker_available:
        return

    skip_docker = pytest.mark.skip(reason="Docker not available")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)


def _check_docker() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def array_sizes(request: pytest.FixtureRequest) -> list[int]:
    """Array sizes to benchmark, from --array-sizes or defaults."""
    raw = request.config.getoption("--array-sizes")
    if raw:
        return [int(s.strip()) for s in raw.split(",")]
    return DEFAULT_ARRAY_SIZES


def create_test_array(size, dtype="float32"):
    """Create a random test array of given size."""
    return np.random.default_rng(42).standard_normal(size).astype(dtype)


# --- from_tesseract_api fixtures ---


@pytest.fixture(scope="session")
def noop_tesseract_api(tmp_path_factory):
    """Create a non-containerized noop Tesseract instance."""
    from tesseract_core import Tesseract

    tmpdir = tmp_path_factory.mktemp("noop_api")
    return Tesseract.from_tesseract_api(NOOP_TESSERACT_PATH, output_path=tmpdir)


@pytest.fixture(scope="session")
def vectoradd_tesseract_api(tmp_path_factory):
    """Create a non-containerized vectoradd Tesseract instance."""
    from tesseract_core import Tesseract

    tmpdir = tmp_path_factory.mktemp("vectoradd_api")
    return Tesseract.from_tesseract_api(VECTORADD_TESSERACT_PATH, output_path=tmpdir)


# --- Docker fixtures ---


@pytest.fixture(scope="session")
def noop_tesseract_image():
    """Build the no-op tesseract image once per session."""
    image_name = "benchmark-noop:latest"
    result = subprocess.run(
        ["tesseract", "build", str(NOOP_TESSERACT_DIR)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build noop tesseract: {result.stderr}")
    return image_name


@pytest.fixture(scope="module")
def noop_tesseract_docker(tmp_path_factory, noop_tesseract_image):
    """Create a containerized noop Tesseract instance."""
    from tesseract_core import Tesseract

    tmpdir = tmp_path_factory.mktemp("noop_docker")
    cm = Tesseract.from_image(noop_tesseract_image, output_path=tmpdir)
    tesseract = cm.__enter__()
    tesseract.health()
    yield tesseract
    cm.__exit__(None, None, None)
