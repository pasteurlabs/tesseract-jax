# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU FFI integration for Tesseract-JAX.

On CUDA, the ``tesseract_dispatch`` primitive lowers to a native XLA FFI custom
call instead of a host callback, so array data stays on the GPU (moved by CUDA
IPC handle, not copied through the host). This module owns the native side of
that path:

* registering the compiled FFI handler with XLA,
* a process-global registry mapping an integer ``token`` (passed to the handler
  as an FFI attribute) to the Python dispatch closure for that call, and
* the single callback the handler invokes, which wraps the XLA input device
  pointers as ``__cuda_array_interface__`` views, runs the dispatch, and returns
  the result device arrays for the handler to copy into XLA's output buffers.

The dispatch closure is endpoint-generic: it is exactly the same
``getattr(client, eval_func)(...)`` closure the CPU (host-callback) lowering
builds, so every endpoint the CPU path supports (apply / jvp / vjp / jacobian)
routes through here unchanged. See :mod:`tesseract_jax.primitive`.

Importing this module does not require CUDA; it only touches the native shim
lazily, when the GPU path is actually used, so CPU-only installs are unaffected.
The dispatch itself needs no CUDA array library (CuPy/Torch): input buffers are
wrapped as bare ``__cuda_array_interface__`` views and outputs come back as the
runtime's framework-agnostic ``IpcDeviceArray``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

FFI_TARGET_NAME = "tesseract_jax_dispatch"

_registered = False
_register_lock = threading.Lock()

# token -> dispatch closure. The closure takes the tuple of input arrays
# (__cuda_array_interface__ views) and returns a tuple of output arrays.
_registry: dict[int, Callable[..., tuple]] = {}
_registry_lock = threading.Lock()
_next_token = 0
_callback_installed = False


def is_available() -> bool:
    """Whether the native GPU FFI shim is importable (compiled and loadable)."""
    try:
        from tesseract_jax import _cuda_shim  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "unavailable"
        return False
    return True


def _native():
    from tesseract_jax import _cuda_shim

    return _cuda_shim


def ensure_registered() -> str:
    """Register the FFI target and native callback with XLA (idempotent)."""
    global _registered, _callback_installed
    with _register_lock:
        if _registered:
            return FFI_TARGET_NAME
        import jax

        native = _native()
        if not _callback_installed:
            native.set_dispatch_callback(_native_dispatch)
            _callback_installed = True
        jax.ffi.register_ffi_target(
            FFI_TARGET_NAME, native.handler_capsule(), platform="CUDA"
        )
        _registered = True
    return FFI_TARGET_NAME


def register_dispatch(fn: Callable[..., tuple]) -> int:
    """Register a dispatch closure, returning its integer token."""
    global _next_token
    with _registry_lock:
        token = _next_token
        _next_token += 1
        _registry[token] = fn
    return token


def release_dispatch(token: int) -> None:
    """Drop the dispatch closure registered under ``token`` (idempotent)."""
    with _registry_lock:
        _registry.pop(token, None)


class _DeviceArrayView:
    """Zero-copy, unowned view of a raw device pointer as a CUDA array.

    The Tesseract client's ``cuda_ipc`` encoder consumes an input array purely
    through the ``__cuda_array_interface__`` protocol -- it reads the pointer,
    shape, dtype, and strides, then takes an IPC handle on the backing
    allocation. It never adopts, frees, or runs kernels on the object. So the
    minimal thing we must hand it is an object exposing that one attribute; a
    full array library (CuPy) is not needed, which keeps CUDA-array-library
    dependencies off the GPU-direct input path.

    The view owns nothing: the memory is XLA's input buffer, valid for the
    duration of the dispatch. ``data``'s read-only flag is ``False`` because the
    encoder may read it via an on-GPU copy.
    """

    def __init__(self, ptr: int, typestr: str, shape: tuple[int, ...]) -> None:
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": typestr,
            "data": (ptr, False),
            "strides": None,  # XLA hands us C-contiguous buffers
            "version": 3,
        }

    # ``.shape`` / ``.dtype`` mirror the metadata already carried in the CUDA
    # array interface, so the view answers the same shape/dtype queries a real
    # array does. The transport-agnostic dispatch code reads these off its
    # arguments (e.g. to size a return slot); exposing them keeps that code from
    # having to special-case the GPU path.
    @property
    def shape(self) -> tuple[int, ...]:
        return self.__cuda_array_interface__["shape"]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.__cuda_array_interface__["typestr"])


def _native_dispatch(
    token: int, inputs: Sequence[tuple[int, str, Sequence[int]]]
) -> list[Any]:
    """Invoked by the native FFI handler under the GIL.

    ``inputs`` is a sequence of ``(device_ptr, numpy_typestr, shape)`` for the
    XLA input buffers (still on device). Returns a list of arrays exposing
    ``__cuda_array_interface__`` whose bytes the handler copies into the XLA
    output buffers.

    The native shim marshals each entry across the pybind11 boundary, where
    ``shape`` arrives as a Python ``list`` rather than a ``tuple``; the sequence
    annotations describe that faithfully (and are normalized to tuples below).
    """
    fn = _registry.get(token)
    if fn is None:
        raise RuntimeError(f"tesseract_jax gpu_ffi: unknown dispatch token {token}")
    views = tuple(
        _DeviceArrayView(ptr, typestr, tuple(shape)) for ptr, typestr, shape in inputs
    )
    out = fn(views)
    return list(out)
