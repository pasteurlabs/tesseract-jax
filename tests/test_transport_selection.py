# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for device-transport selection on the Jaxeract wrapper.

These cover the transport-name plumbing that decides *which* on-device transport
a call uses (and whether it uses one at all) -- the request/response encoding is
exercised end-to-end by the GPU tests in ``test_gpu_direct.py``. Only the
selection logic is unit-tested here, since it gates the GPU (FFI) lowering and a
wrong answer silently sends an unsupported ``Accept`` to the server.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tesseract_jax.tesseract_compat import Jaxeract


def _fake_client() -> MagicMock:
    c = MagicMock()
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


def test_cuda_ipc_bool_maps_to_transport_name():
    j = Jaxeract(_fake_client(), cuda_ipc=True)
    assert j._device_transport == "cuda_ipc"
    assert j._cuda_ipc is True


def test_device_transport_name_selects_transport():
    j = Jaxeract(_fake_client(), device_transport="cuda_ipc")
    assert j._device_transport == "cuda_ipc"
    assert j._cuda_ipc is True


def test_default_is_host_roundtrip():
    j = Jaxeract(_fake_client())
    assert j._device_transport is None
    assert j._cuda_ipc is False


def test_unsupported_transport_is_rejected():
    # Regression: an unsupported name must not be silently accepted -- it would
    # route into the cuda_ipc-specific GPU lowering and send an Accept the server
    # has no backend for.
    with pytest.raises(ValueError, match="Unsupported device_transport"):
        Jaxeract(_fake_client(), device_transport="nixl")


def test_conflicting_cuda_ipc_and_transport_rejected():
    with pytest.raises(ValueError, match="not both"):
        Jaxeract(_fake_client(), cuda_ipc=True, device_transport="nixl")


def test_equality_and_hash_key_on_transport():
    c = _fake_client()
    a = Jaxeract(c, device_transport="cuda_ipc")
    b = Jaxeract(c, cuda_ipc=True)
    host = Jaxeract(c)
    # Same underlying client + same transport -> interchangeable (so XLA may
    # common them up); different transport -> must not compare equal.
    assert a == b and hash(a) == hash(b)
    assert a != host


class _FakeSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


class _FakeHTTPClient:
    """Stand-in for tesseract-core's HTTPClient with the attrs the CM touches."""

    def __init__(self) -> None:
        self._gpu_transport = "none"
        self._output_format = "json+base64"
        self._session = _FakeSession()


def _client_with_http() -> MagicMock:
    c = _fake_client()
    c._client = _FakeHTTPClient()
    return c


def test_device_transport_encoding_drives_gpu_transport_and_accept():
    # Regression for the tesseract-core split of CPU encoding (``_output_format``)
    # from GPU transport (``_gpu_transport``): opting a call into cuda_ipc must
    # set ``_gpu_transport`` and negotiate the server's GPU output transport via
    # an Accept media-type parameter -- without disturbing ``_output_format``.
    c = _client_with_http()
    j = Jaxeract(c, cuda_ipc=True)
    http = c._client

    with j.device_transport_encoding():
        assert http._gpu_transport == "cuda_ipc"
        assert http._output_format == "json+base64"
        assert (
            http._session.headers["Accept"]
            == "application/json+base64; gpu_transport=cuda_ipc"
        )

    # Fully restored on exit: the shared client must not leak the transport onto
    # host-callback / CPU uses.
    assert http._gpu_transport == "none"
    assert "Accept" not in http._session.headers


def test_device_transport_encoding_is_noop_without_transport():
    c = _client_with_http()
    j = Jaxeract(c)  # host round-trip
    http = c._client

    with j.device_transport_encoding():
        assert http._gpu_transport == "none"
        assert "Accept" not in http._session.headers
