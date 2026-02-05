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


def make_tesseract_fixture_without_endpoint(source_folder, endpoint_to_remove):
    """Factory function to create tesseract fixtures with a specific endpoint removed.

    This creates a temporary copy of the source tesseract and removes the specified endpoint.
    """

    @pytest.fixture(scope="session")
    def served_tesseract(tmp_path_factory):
        import shutil

        # Create a temporary directory for this modified tesseract
        tmp_dir = tmp_path_factory.mktemp(f"{source_folder}_no_{endpoint_to_remove}")

        # Copy the source tesseract_api.py
        source_api = here / source_folder / "tesseract_api.py"
        dest_api = tmp_dir / "tesseract_api.py"

        # Read the source file
        with open(source_api) as f:
            content = f.read()

        # Remove the specified endpoint function using AST
        import ast

        tree = ast.parse(content)
        new_body = []

        for node in tree.body:
            # Skip the function we want to remove
            if isinstance(node, ast.FunctionDef) and node.name == endpoint_to_remove:
                continue
            new_body.append(node)

        tree.body = new_body
        new_content = ast.unparse(tree)

        # Write the modified content
        with open(dest_api, "w") as f:
            f.write(new_content)

        # Copy other necessary files
        for filename in ["tesseract_config.yaml", "tesseract_requirements.txt"]:
            source_file = here / source_folder / filename
            if source_file.exists():
                shutil.copy(source_file, tmp_dir / filename)

        # Start the server
        port = find_free_port()
        timeout = 10

        env = os.environ.copy()
        env["TESSERACT_API_PATH"] = str(dest_api)

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
                            f"Tesseract without {endpoint_to_remove} did not start in time"
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

# Tesseracts with specific endpoints removed for testing error handling
served_tesseract_no_vjp = make_tesseract_fixture_without_endpoint(
    "univariate_tesseract", "vector_jacobian_product"
)
served_tesseract_no_jvp = make_tesseract_fixture_without_endpoint(
    "univariate_tesseract", "jacobian_vector_product"
)
