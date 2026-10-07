# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests against a Tesseract served from a Docker container.

The other GPU tests reach Tesseracts in-process and in a subprocess. Here the
Tesseract runs in a container, so arrays cross a container boundary, and a
container that cannot see the GPU stands in for a server that shares no GPU with
the client, such as one on another host. The tests need Docker with the NVIDIA
container runtime and are marked ``gpu``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

pytestmark = pytest.mark.gpu

here = Path(__file__).parent


@pytest.fixture(scope="module")
def gpu_container_image() -> str:
    """Build the container test Tesseract once. Skips without a GPU backend for JAX."""
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU backend for JAX")
    from tesseract_core.sdk.engine import build_tesseract

    # The image name comes from the Tesseract's config, the tag from here
    build_tesseract(here / "gpu_container_tesseract", "test")
    return "tesseract-jax-gpu-container:test"


def _on_gpu(x) -> bool:
    return any(d.platform == "gpu" for d in x.devices())


@pytest.mark.parametrize("client", ["served", "from_url"])
def test_container_round_trip_stays_on_device(gpu_container_image, monkeypatch, client):
    """Arrays cross the container boundary over cuda_ipc, host-copy free.

    The client that served the container requests cuda_ipc, and a ``from_url``
    client of it requests nothing and picks cuda_ipc once it checked that it
    works. The container forbids host copies of GPU arrays and the FFI handler
    rejects host buffers, so passing rules out a host copy on either side.
    """
    from tesseract_jax.gpu_ffi import FFI_TARGET_NAME

    monkeypatch.setenv("TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS", "1")
    with Tesseract.from_image(
        gpu_container_image,
        gpus=["all"],
        gpu_transport="cuda_ipc",
        environment={"TESSERACT_FORBID_DEVICE_HOST_COPY": "1"},
    ) as served:
        tess = served if client == "served" else Tesseract.from_url(served._client.url)
        a = jnp.arange(64, dtype=jnp.float32)
        b = jnp.ones(64, dtype=jnp.float32)
        f = jax.jit(lambda a, b: apply_tesseract(tess, {"a": a, "b": b})["c"])
        assert FFI_TARGET_NAME in f.lower(a, b).as_text()

        c = f(a, b)
        assert _on_gpu(c)
        np.testing.assert_allclose(np.asarray(c), 2 * np.arange(64) + 1.0)
        g = jax.jit(jax.grad(lambda a: f(a, b).sum()))(a)
        assert _on_gpu(g)
        np.testing.assert_allclose(np.asarray(g), np.full(64, 2.0))


def test_container_without_a_shared_gpu_gets_host_copies(gpu_container_image):
    """A container offering cuda_ipc that it cannot use leads to host copies.

    A ``from_url`` client, which requests no transport, falls back to the
    host-callback lowering with a warning and still gets correct results on the
    GPU. The client that requested cuda_ipc gets an error instead of a silent
    host copy.
    """
    from tesseract_jax.gpu_ffi import FFI_TARGET_NAME

    with Tesseract.from_image(
        gpu_container_image,
        gpus=["all"],
        gpu_transport="cuda_ipc",
        environment={"CUDA_VISIBLE_DEVICES": ""},
    ) as served:
        remote = Tesseract.from_url(served._client.url)
        a = jnp.arange(8, dtype=jnp.float32)
        b = jnp.ones(8, dtype=jnp.float32)
        f = jax.jit(lambda a, b: apply_tesseract(remote, {"a": a, "b": b})["c"])
        with pytest.warns(UserWarning, match="copied to the host instead"):
            assert FFI_TARGET_NAME not in f.lower(a, b).as_text()

        c = f(a, b)
        assert _on_gpu(c)
        np.testing.assert_allclose(np.asarray(c), 2 * np.arange(8) + 1.0)
        g = jax.jit(jax.grad(lambda a: f(a, b).sum()))(a)
        np.testing.assert_allclose(np.asarray(g), np.full(8, 2.0))

        with pytest.raises(RuntimeError, match="does not work between"):
            jax.jit(lambda a, b: apply_tesseract(served, {"a": a, "b": b}))(a, b)
