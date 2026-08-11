# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-direct Tesseract dispatch for Tesseract-JAX (experimental).

This package provides a native XLA FFI handler that lets a JAX program hand
device buffers to a Tesseract and receive device buffers back, using
tesseract-core's ``cuda_ipc`` array encoding, without a device→host→device
round-trip.

The native handler is strategy-agnostic; the two dispatch strategies under
comparison (A: Python does the ``cuda_ipc`` handshake, B: native code does it)
differ only in the Python callback registered behind the FFI boundary.
"""

from __future__ import annotations

_FFI_TARGET_NAME = "tesseract_jax_cuda_ipc_dispatch"
_registered = False


def _native():
    """Import the compiled shim, or raise a clear error if it isn't built."""
    from tesseract_jax_gpu import _cuda_shim

    return _cuda_shim


def ensure_registered() -> str:
    """Register the FFI target with XLA (idempotent). Returns the target name."""
    global _registered
    if _registered:
        return _FFI_TARGET_NAME

    import jax

    native = _native()
    jax.ffi.register_ffi_target(
        _FFI_TARGET_NAME,
        native.handler_capsule(),
        platform="CUDA",
    )
    _registered = True
    return _FFI_TARGET_NAME


def set_dispatch_callback(fn) -> None:
    """Install the Python callable the native handler invokes per dispatch.

    ``fn`` is called as ``fn(token: int, inputs: list[tuple[ptr, typestr,
    shape]]) -> list[array]`` where each returned array must expose
    ``__cuda_array_interface__`` and match the corresponding XLA output buffer's
    shape/dtype. The native handler copies each result device→device into the
    XLA-owned output buffer.
    """
    _native().set_dispatch_callback(fn)
