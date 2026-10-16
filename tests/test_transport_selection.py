# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for GPU transport selection on the Jaxeract wrapper.

Which transport a call uses gates the GPU (FFI) lowering, so these check that
only the ``cuda`` lowering asks the Tesseract for one, that the CPU lowering
always requests host outputs, and that a transport the FFI path cannot drive is
rejected. Whether a transport works is tesseract-core's to find out (see
``Tesseract.resolve_gpu_transport``), and the GPU tests in ``test_gpu_direct.py``
exercise the transport end to end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract
from tesseract_jax.tesseract_compat import Jaxeract

here = Path(__file__).parent


def _fake_client(gpu_transport: str = "none") -> MagicMock:
    c = MagicMock()
    c.resolve_gpu_transport.return_value = gpu_transport
    c.openapi_schema = {
        "components": {
            "schemas": {
                "Apply_InputSchema": {"properties": {"a": {}, "b": {}}},
                "Apply_OutputSchema": {"properties": {"c": {}}},
                "ApplyInputSchema": {"differentiable_arrays": ["a"]},
                "ApplyOutputSchema": {"differentiable_arrays": ["c"]},
            }
        }
    }
    c.available_endpoints = ["apply"]
    return c


def test_transport_is_the_one_the_tesseract_resolves_to():
    for gpu_transport in ("none", "cuda_ipc"):
        assert Jaxeract(_fake_client(gpu_transport)).resolve_gpu_transport() == (
            gpu_transport
        )


def test_in_process_transport_is_the_one_it_was_created_with():
    api = here / "vectoradd_tesseract" / "tesseract_api.py"
    assert Jaxeract(Tesseract.from_tesseract_api(api)).resolve_gpu_transport() == (
        "none"
    )
    enabled = Tesseract.from_tesseract_api(api, gpu_transport="cuda_ipc")
    assert Jaxeract(enabled).resolve_gpu_transport() == "cuda_ipc"


def test_unsupported_transport_is_rejected():
    # A transport the FFI path cannot drive must not route into the
    # cuda_ipc-specific GPU lowering.
    with pytest.raises(ValueError, match="tesseract-jax cannot drive"):
        Jaxeract(_fake_client("nixl")).resolve_gpu_transport()


def test_equality_and_hash_key_on_the_tesseract():
    c = _fake_client()
    a, b = Jaxeract(c), Jaxeract(c)
    # Same Tesseract -> interchangeable, so XLA may common them up.
    assert a == b and hash(a) == hash(b)
    assert a != Jaxeract(_fake_client())


@pytest.mark.parametrize("client", ["served", "in_process"])
def test_cuda_ipc_tesseract_runs_on_cpu(request, client):
    """A Tesseract created with cuda_ipc stays usable from CPU-only JAX.

    Only the ``cuda`` lowering acts on the transport, so CPU arrays take the host
    callback instead of failing for lack of a CUDA device.
    """
    if client == "served":
        tess = request.getfixturevalue("served_cuda_ipc_vectoradd_tesseract")
    else:
        tess = Tesseract.from_tesseract_api(
            here / "vectoradd_tesseract" / "tesseract_api.py", gpu_transport="cuda_ipc"
        )

    def f(a, b):
        return apply_tesseract(tess, {"a": a, "b": b})["c"]

    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(8, dtype=jnp.float32)
        b = jnp.ones(8, dtype=jnp.float32)
        c = jax.jit(f)(a, b)
        grad_a = jax.jit(jax.grad(lambda a: f(a, b).sum()))(a)

    np.testing.assert_array_equal(c, np.arange(8) + 1.0)
    np.testing.assert_array_equal(grad_a, np.ones(8))


@pytest.mark.parametrize("client", ["served", "from_url"])
def test_host_lowering_requests_host_outputs(
    served_cuda_ipc_vectoradd_tesseract, monkeypatch, client
):
    """Compiled for CPU, no dispatch asks for GPU arrays by reference.

    Both the client that requests cuda_ipc and a ``from_url`` client ask for
    ``gpu_transport=none``. The host callback moves the arrays through the host
    anyway. Neither checks whether cuda_ipc works, which would need a GPU.
    """
    tess = served_cuda_ipc_vectoradd_tesseract
    if client == "from_url":
        tess = Tesseract.from_url(tess._client.url)
    http = tess._client
    sent: list[tuple[str, str | None]] = []
    send = http._send

    def recording_send(url, method, data, params, headers=None):
        sent.append((url, (headers or {}).get("Accept")))
        return send(url, method, data, params, headers)

    monkeypatch.setattr(http, "_send", recording_send)

    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(8, dtype=jnp.float32)
        b = jnp.ones(8, dtype=jnp.float32)
        c = apply_tesseract(tess, {"a": a, "b": b})["c"]

    np.testing.assert_array_equal(c, np.arange(8) + 1.0)
    apply_accepts = [accept for url, accept in sent if url.endswith("/apply")]
    assert len(apply_accepts) == 1
    assert apply_accepts[0].endswith("; gpu_transport=none")
    assert not any(url.endswith("/check_gpu_transport") for url, _ in sent)
