# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for device-transport selection on the Jaxeract wrapper.

These cover the transport-name plumbing that decides *which* on-device transport
a call uses (and whether it uses one at all), since it gates the GPU (FFI)
lowering and a wrong answer silently sends an unsupported ``Accept`` to the
server. Most stub the client; the rest check that tesseract-core's real clients
behave the way the stubs assume, and that a client with a default transport stays
usable from CPU-only JAX. The GPU-direct request/response encoding is exercised
end-to-end by the GPU tests in ``test_gpu_direct.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract
from tesseract_core.sdk.tesseract import HTTPClient

from tesseract_jax import apply_tesseract, gpu_ffi
from tesseract_jax.tesseract_compat import Jaxeract, _default_gpu_transport

# Never contacted: the tests below only build clients for it, sending no requests.
_UNREACHABLE_URL = "http://127.0.0.1:1"


def _fake_client(supported_gpu_transports: tuple[str, ...] = ()) -> MagicMock:
    c = MagicMock()
    c.supported_gpu_transports = supported_gpu_transports
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


def test_gpu_transport_name_selects_transport():
    # A plain ``from_url`` client advertises no transport, so naming one per call
    # must work without it.
    j = Jaxeract(_fake_client(), gpu_transport="cuda_ipc")
    assert j._gpu_transport == "cuda_ipc"


def test_default_is_host_roundtrip():
    j = Jaxeract(_fake_client())
    assert j._gpu_transport is None


def test_default_uses_client_transport():
    j = Jaxeract(_fake_client(("cuda_ipc",)))
    assert j._gpu_transport == "cuda_ipc"


def test_default_ignores_transports_the_lowering_lacks():
    j = Jaxeract(_fake_client(("nixl",)))
    assert j._gpu_transport is None


def test_default_does_not_depend_on_shim(monkeypatch):
    # Without the shim, a selected transport fails at lowering time (see
    # test_gpu_ffi.py) instead of silently falling back to the host.
    monkeypatch.setattr(gpu_ffi, "is_available", lambda: False)
    j = Jaxeract(_fake_client(("cuda_ipc",)))
    assert j._gpu_transport == "cuda_ipc"


def test_none_forces_host_roundtrip():
    j = Jaxeract(_fake_client(("cuda_ipc",)), gpu_transport="none")
    assert j._gpu_transport is None


def test_unsupported_transport_is_rejected():
    # An unsupported name must not be silently accepted: it would route into the
    # cuda_ipc-specific GPU lowering and send an Accept the server has no backend
    # for.
    with pytest.raises(ValueError, match="Unsupported gpu_transport"):
        Jaxeract(_fake_client(), gpu_transport="nixl")


def test_equality_and_hash_key_on_transport():
    c = _fake_client()
    a = Jaxeract(c, gpu_transport="cuda_ipc")
    b = Jaxeract(c, gpu_transport="cuda_ipc")
    host = Jaxeract(c)
    # Same client + same transport -> interchangeable (so XLA may common them up);
    # different transport -> must not compare equal.
    assert a == b and hash(a) == hash(b)
    assert a != host


def _client_with_http() -> MagicMock:
    c = _fake_client()
    # A real HTTPClient, so the private attributes the context manager writes are
    # the ones tesseract-core reads when encoding a request.
    c._client = HTTPClient(_UNREACHABLE_URL)
    return c


# requests sends ``Accept: */*`` unless the session's default is removed.
@pytest.mark.parametrize("session_accept", ["*/*", None])
def test_gpu_transport_encoding_drives_gpu_transport_and_accept(session_accept):
    # tesseract-core keeps CPU encoding (``_output_format``) and GPU transport
    # (``_gpu_transport``) on separate axes: selecting a device transport must set
    # ``_gpu_transport`` and negotiate the server's GPU output transport via an
    # Accept media-type parameter, without disturbing ``_output_format``.
    c = _client_with_http()
    j = Jaxeract(c, gpu_transport="cuda_ipc")
    http = c._client
    if session_accept is None:
        http._session.headers.pop("Accept", None)
    else:
        http._session.headers["Accept"] = session_accept

    with j.gpu_transport_encoding():
        assert http._gpu_transport == "cuda_ipc"
        assert http._output_format == "json+base64"
        assert (
            http._session.headers["Accept"]
            == "application/json+base64; gpu_transport=cuda_ipc"
        )

    # Fully restored on exit: the shared client must not leak the transport onto
    # host-callback / CPU uses.
    assert http._gpu_transport == "none"
    assert http._session.headers.get("Accept") == session_accept


def test_default_on_real_clients_without_transport(vectoradd_tess):
    # The tests above stub ``supported_gpu_transports``. These are the clients
    # tesseract-core builds without a transport: one reached by URL and an
    # in-process one.
    assert _default_gpu_transport(Tesseract.from_url(_UNREACHABLE_URL)) is None
    assert _default_gpu_transport(vectoradd_tess) is None


def test_default_transport_client_runs_on_cpu(served_cuda_ipc_vectoradd_tesseract):
    """A client served with cuda_ipc stays usable from CPU-only JAX.

    The default selects cuda_ipc for it, but only the ``cuda`` lowering acts on
    the transport, so CPU arrays take the host callback instead of failing for
    lack of a CUDA device.
    """
    tess = served_cuda_ipc_vectoradd_tesseract
    assert _default_gpu_transport(tess) == "cuda_ipc"

    def f(a, b):
        return apply_tesseract(tess, {"a": a, "b": b})["c"]

    with jax.default_device(jax.devices("cpu")[0]):
        a = jnp.arange(8, dtype=jnp.float32)
        b = jnp.ones(8, dtype=jnp.float32)
        c = jax.jit(f)(a, b)
        grad_a = jax.jit(jax.grad(lambda a: f(a, b).sum()))(a)

    np.testing.assert_array_equal(c, np.arange(8) + 1.0)
    np.testing.assert_array_equal(grad_a, np.ones(8))
