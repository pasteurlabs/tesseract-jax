# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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


def get_tesseract_folders():
    tesseract_folders = [
        "univariate_tesseract",
        "nested_tesseract",
        "non_abstract_tesseract",
        "pytree_tesseract",
        # Add more as needed
    ]
    return tesseract_folders


def find_free_port():
    """Find a free port to use for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def make_tesseract_fixture(folder_name):
    """Factory function to create tesseract fixtures for different folders.

    This fixture serves a Tesseract via `tesseract-runtime` for the specified folder.
    This way, we can test with real Tesseracts without a running Docker daemon.
    """

    @pytest.fixture(scope="session")
    def served_tesseract():
        port = find_free_port()
        timeout = 10

        env = os.environ.copy()
        env["TESSERACT_API_PATH"] = str(here / folder_name / "tesseract_api.py")

        # Start the server as a subprocess
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
                            f"Tesseract for {folder_name} did not start in time"
                        ) from exc
                    time.sleep(0.1)

            yield f"http://localhost:{port}"
        finally:
            process.terminate()
            process.communicate()

    return served_tesseract


served_univariate_tesseract_raw = make_tesseract_fixture("univariate_tesseract")
served_nested_tesseract_raw = make_tesseract_fixture("nested_tesseract")
served_non_abstract_tesseract = make_tesseract_fixture("non_abstract_tesseract")
served_pytree_tesseract = make_tesseract_fixture("pytree_tesseract")


@pytest.fixture
def pytree_tess() -> Tesseract:
    """Load pytree_tesseract directly from the API file."""
    return Tesseract.from_tesseract_api("tests/pytree_tesseract/tesseract_api.py")


@pytest.fixture
def pytree_tess_inputs() -> dict:
    """Provide inputs for pytree_tesseract tests."""
    x = np.array([1.0, 2.0, 3.0], dtype="float32")
    y = np.array([4.0, 5.0, 6.0], dtype="float32")
    z = np.array([7.0, 8.0, 9.0], dtype="float32")
    u = np.array([10.0, 11.0, 12.0], dtype="float32")
    v = np.array([13.0, 14.0, 15.0], dtype="float32")
    d0 = np.array([16.0, 17.0, 18.0], dtype="float32")
    d1 = np.array([19.0, 20.0, 21.0], dtype="float32")
    k = np.array([2.0, 2.0, 2.0], dtype="float32")
    m = np.array([3.0, 3.0, 3.0], dtype="float32")
    z0 = np.array([1.0, 1.0, 1.0], dtype="float32")
    z1 = np.array([2.0, 2.0, 2.0], dtype="float32")

    inputs = {
        "alpha": {
            "x": x,
            "y": y,
        },
        "beta": {"z": z, "gamma": {"u": u, "v": v}},
        "delta": [d0, d1],
        "epsilon": {"k": k, "m": m},
        "zeta": [z0, z1],
    }

    return inputs
