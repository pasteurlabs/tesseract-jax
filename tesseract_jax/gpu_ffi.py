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
  pointers as CuPy views, runs the dispatch, and returns the result device
  arrays for the handler to copy into XLA's output buffers.

The dispatch closure is endpoint-generic: it is exactly the same
``getattr(client, eval_func)(...)`` closure the CPU (host-callback) lowering
builds, so every endpoint the CPU path supports (apply / jvp / vjp / jacobian)
routes through here unchanged. See :mod:`tesseract_jax.primitive`.

Importing this module does not require CUDA; it only touches the native shim and
CuPy lazily, when the GPU path is actually used, so CPU-only installs are
unaffected.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

FFI_TARGET_NAME = "tesseract_jax_dispatch"

_registered = False
_register_lock = threading.Lock()

# token -> dispatch closure. The closure takes the tuple of input arrays (CuPy
# views) and returns a tuple of output arrays.
_registry: dict[int, Callable[..., tuple]] = {}
_registry_lock = threading.Lock()
_next_token = 0
_callback_installed = False


def is_available() -> bool:
    """Whether the native GPU FFI shim is importable (compiled and loadable)."""
    try:
        from tesseract_jax import _cuda_shim  # noqa: F401
    except Exception:
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
    with _registry_lock:
        _registry.pop(token, None)


def _cupy_view(ptr: int, typestr: str, shape: tuple[int, ...]):
    """Wrap a raw device pointer as an unowned CuPy view (no copy)."""
    import cupy
    import numpy as np

    dtype = np.dtype(typestr)
    n = int(np.prod(shape)) if shape else 1
    nbytes = n * dtype.itemsize
    mem = cupy.cuda.UnownedMemory(ptr, nbytes, owner=None)
    memptr = cupy.cuda.MemoryPointer(mem, 0)
    return cupy.ndarray(tuple(shape), dtype=dtype, memptr=memptr)


def _native_dispatch(
    token: int, inputs: list[tuple[int, str, tuple[int, ...]]]
) -> list[Any]:
    """Invoked by the native FFI handler under the GIL.

    ``inputs`` is a list of ``(device_ptr, numpy_typestr, shape)`` for the XLA
    input buffers (still on device). Returns a list of arrays exposing
    ``__cuda_array_interface__`` whose bytes the handler copies into the XLA
    output buffers.
    """
    fn = _registry.get(token)
    if fn is None:
        raise RuntimeError(f"tesseract_jax gpu_ffi: unknown dispatch token {token}")
    views = tuple(
        _cupy_view(ptr, typestr, tuple(shape)) for ptr, typestr, shape in inputs
    )
    out = fn(views)
    return list(out)
