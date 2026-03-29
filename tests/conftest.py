# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import socket
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
import pytest
import requests
from tesseract_core import Tesseract

here = Path(__file__).parent

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port():
    """Find a free port to use for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _strip_functions_from_api(source: str, func_names: set[str]) -> str:
    """Return *source* with top-level function definitions in *func_names* removed."""
    tree = ast.parse(source)
    # Collect line ranges (1-indexed) of functions to remove
    remove_ranges: list[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in func_names and node.end_lineno is not None:
                remove_ranges.append((node.lineno, node.end_lineno))

    if not remove_ranges:
        return source

    lines = source.splitlines(keepends=True)
    keep: list[str] = []
    for i, line in enumerate(lines, start=1):
        if not any(start <= i <= end for start, end in remove_ranges):
            keep.append(line)
    return "".join(keep)


def _serve_tesseract(tmp_path_factory, api_path: str | Path, *, name: str):
    """Start a tesseract-runtime server and yield its URL."""
    port = _find_free_port()
    timeout = 10

    output_dir = tmp_path_factory.mktemp(f"tesseract_output_{name}")

    env = os.environ.copy()
    env["TESSERACT_API_PATH"] = str(api_path)
    env["TESSERACT_OUTPUT_PATH"] = str(output_dir)

    process = subprocess.Popen(
        [
            "tesseract-runtime",
            "serve",
            "--host",
            "localhost",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        start_time = time.time()
        while True:
            try:
                requests.get(f"http://localhost:{port}/health")
                break
            except requests.exceptions.ConnectionError as exc:
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Tesseract {name!r} did not start in time"
                    ) from exc
                time.sleep(0.1)

        yield f"http://localhost:{port}"
    finally:
        process.terminate()
        process.communicate()


def _load_tesseract(folder_name: str) -> Tesseract:
    """Load a Tesseract directly from a test API file."""
    return Tesseract.from_tesseract_api(f"tests/{folder_name}/tesseract_api.py")


# ---------------------------------------------------------------------------
# Served fixtures  (session-scoped, start a tesseract-runtime process)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def served_univariate_tesseract_raw(tmp_path_factory):
    yield from _serve_tesseract(
        tmp_path_factory,
        here / "univariate_tesseract" / "tesseract_api.py",
        name="univariate",
    )


@pytest.fixture(scope="session")
def served_nested_tesseract_raw(tmp_path_factory):
    yield from _serve_tesseract(
        tmp_path_factory,
        here / "nested_tesseract" / "tesseract_api.py",
        name="nested",
    )


@pytest.fixture(scope="session")
def served_non_abstract_tesseract(tmp_path_factory):
    yield from _serve_tesseract(
        tmp_path_factory,
        here / "non_abstract_tesseract" / "tesseract_api.py",
        name="non_abstract",
    )


@pytest.fixture(scope="session")
def served_vectoradd_tesseract(tmp_path_factory):
    yield from _serve_tesseract(
        tmp_path_factory,
        here / "vectoradd_tesseract" / "tesseract_api.py",
        name="vectoradd",
    )


@pytest.fixture(scope="session")
def served_pytree_tesseract(tmp_path_factory):
    yield from _serve_tesseract(
        tmp_path_factory,
        here / "pytree_tesseract" / "tesseract_api.py",
        name="pytree",
    )
    
@pytest.fixture(scope="session")
def served_batched_tesseract(tmp_path_factory):
    yield from _serve_tesseract(
        tmp_path_factory,
        here / "batched_tesseract" / "tesseract_api.py",
        name="batched",
    )


# Tesseracts with specific endpoints removed — generated dynamically from
# the base univariate_tesseract so we don't need separate directories.


@pytest.fixture(scope="session")
def served_tesseract_no_jvp(tmp_path_factory):
    source = (here / "univariate_tesseract" / "tesseract_api.py").read_text()
    stripped = _strip_functions_from_api(source, {"jacobian_vector_product"})
    api_file = tmp_path_factory.mktemp("univariate_no_jvp") / "tesseract_api.py"
    api_file.write_text(stripped)
    yield from _serve_tesseract(tmp_path_factory, api_file, name="univariate_no_jvp")


@pytest.fixture(scope="session")
def served_tesseract_no_vjp(tmp_path_factory):
    source = (here / "univariate_tesseract" / "tesseract_api.py").read_text()
    stripped = _strip_functions_from_api(source, {"vector_jacobian_product"})
    api_file = tmp_path_factory.mktemp("univariate_no_vjp") / "tesseract_api.py"
    api_file.write_text(stripped)
    yield from _serve_tesseract(tmp_path_factory, api_file, name="univariate_no_vjp")


# ---------------------------------------------------------------------------
# Direct-load fixtures  (function-scoped, no server needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def pytree_tess() -> Tesseract:
    return _load_tesseract("pytree_tesseract")


@pytest.fixture
def univariate_tess() -> Tesseract:
    return _load_tesseract("univariate_tesseract")


@pytest.fixture
def vectoradd_tess() -> Tesseract:
    return _load_tesseract("vectoradd_tesseract")


@pytest.fixture
def static_input_tess() -> Tesseract:
    return _load_tesseract("static_input_tesseract")


# ---------------------------------------------------------------------------
# Shared test inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def pytree_tess_inputs() -> dict:
    """Provide inputs for pytree_tesseract tests with different shapes."""
    x = np.array([1.0, 2.0, 3.0], dtype="float32")  # shape (3,)
    y = np.array([4.0, 5.0, 6.0, 7.0], dtype="float32")  # shape (4,)
    z = np.array([8.0, 9.0, 10.0, 11.0, 12.0], dtype="float32")  # shape (5,)
    u = np.array([13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype="float32")  # shape (6,)
    v = np.array(
        [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype="float32"
    )  # shape (7,)
    d0 = np.array(
        [26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0], dtype="float32"
    )  # shape (8,)
    d1 = np.array(
        [34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0], dtype="float32"
    )  # shape (9,)
    k = np.array([2.0, 2.0], dtype="float32")  # shape (2,) non-differentiable
    m = np.array(
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype="float32"
    )  # shape (10,) non-differentiable
    z0 = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype="float32"
    )  # shape (11,) non-differentiable
    z1 = np.array(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype="float32"
    )  # shape (12,) non-differentiable

    return {
        "alpha": {
            "x": x,
            "y": y,
        },
        "beta": {"z": z, "gamma": {"u": u, "v": v}},
        "delta": [d0, d1],
        "epsilon": {"k": k, "m": m},
        "zeta": [z0, z1],
    }
