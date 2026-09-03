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
