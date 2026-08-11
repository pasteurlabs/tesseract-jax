# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a GPU Tesseract as a real subprocess (no Docker) for cuda_ipc tests.

Cross-process CUDA IPC needs the producer (the Tesseract) and the consumer (this
process) to be *separate* processes that share the GPU -- a process cannot open
an IPC handle it exported itself. Running the runtime's FastAPI app via uvicorn
in a subprocess gives exactly that, without the cost of building a container.
Both processes are on the same host and see the same GPU, satisfying the
cuda_ipc requirements.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
GPU_TESSERACT = HERE / "gpu_tesseract" / "tesseract_api.py"


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve_gpu_tesseract(
    api_path: Path = GPU_TESSERACT,
    output_format: str = "json+cuda_ipc",
    python: str = sys.executable,
):
    """Context manager yielding the base URL of a served GPU Tesseract."""
    port = _free_port()
    # Give the server its own scratch output directory (and run it from there)
    # so per-request ``run_<uuid>`` dirs never land in the repo tree. Cleaned up
    # on exit.
    workdir = Path(tempfile.mkdtemp(prefix="tesseract_gpu_serve_"))
    env = os.environ.copy()
    env.update(
        TESSERACT_API_PATH=str(Path(api_path).resolve()),
        TESSERACT_OUTPUT_FORMAT=output_format,
        TESSERACT_OUTPUT_PATH=str(workdir),
        # cuda_ipc output is an experimental opt-in in tesseract-core.
        TESSERACT_ENABLE_EXPERIMENTAL_CUDA_IPC="1",
    )
    proc = subprocess.Popen(
        [
            python,
            "-m",
            "uvicorn",
            "tesseract_core.runtime.app_http:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(workdir, ignore_errors=True)


def _wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"Tesseract server exited early:\n{out}")
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(0.3)
    raise RuntimeError("Tesseract server did not become healthy in time")
