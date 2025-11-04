# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import subprocess
import time
from pathlib import Path

import jax
import pytest
import requests

here = Path(__file__).parent

jax.config.update("jax_enable_x64", True)


def get_tesseract_folders():
    tesseract_folders = [
        "univariate_tesseract",
        "nested_tesseract",
        "non_abstract_tesseract",
        "vectoradd_tesseract",
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
served_vectoradd_tesseract = make_tesseract_fixture("vectoradd_tesseract")
