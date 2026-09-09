# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GPU FFI registration/priming logic (no GPU required).

These do not exercise a real dispatch (that needs a GPU; see
``test_gpu_direct.py``). They pin the *contract* between the Python side and the
native shim: the shim's ``cuda_rt()`` throws unless its libcudart candidate list
was primed first, so ``ensure_registered()`` must prime it -- with a non-empty
list -- before it registers the FFI target that makes the handler reachable. If
that ordering regresses (or the priming is dropped), the shim would fall back to
guessing a libcudart, the exact mismatch the priming exists to prevent, so these
guard it on the CPU test runner rather than only implicitly on GPU CI.
"""

from __future__ import annotations

import pytest

from tesseract_jax import gpu_ffi


@pytest.fixture
def fresh_registry(monkeypatch):
    """Reset ``gpu_ffi``'s process-global registration latches around a test.

    ``ensure_registered`` is idempotent via module-global flags; a prior import
    or test may have flipped them. Reset them (and restore afterwards) so each
    test observes a clean, first-time registration.
    """
    monkeypatch.setattr(gpu_ffi, "_registered", False)
    monkeypatch.setattr(gpu_ffi, "_callback_installed", False)
    yield


class _RecordingNative:
    """Stand-in for the native ``_cuda_shim`` module that records call order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cudart_candidates: list[str] | None = None

    def set_dispatch_callback(self, cb) -> None:
        self.calls.append("set_dispatch_callback")

    def set_cudart_candidates(self, names) -> None:
        self.calls.append("set_cudart_candidates")
        # Capture what was passed so the test can assert it is non-empty.
        self.cudart_candidates = list(names)

    def handler_capsule(self):
        self.calls.append("handler_capsule")
        return object()  # opaque capsule stand-in


def test_ensure_registered_primes_cudart_before_registering(
    fresh_registry, monkeypatch
):
    """The candidate list is primed, non-empty, and set before registration.

    The native shim reads its libcudart list once on the first dispatch and
    errors if it was never set, so priming must happen before the FFI target is
    registered (which is what makes the handler reachable at all).
    """
    native = _RecordingNative()
    monkeypatch.setattr(gpu_ffi, "_native", lambda: native)

    registered: dict = {}

    def fake_register_ffi_target(name, capsule, platform):
        registered["name"] = name
        registered["platform"] = platform
        # Registration must come only after the shim has been primed.
        assert native.calls == [
            "set_dispatch_callback",
            "set_cudart_candidates",
            "handler_capsule",
        ], native.calls

    import jax

    monkeypatch.setattr(jax.ffi, "register_ffi_target", fake_register_ffi_target)

    target = gpu_ffi.ensure_registered()

    assert target == gpu_ffi.FFI_TARGET_NAME
    assert registered["platform"] == "CUDA"
    # The shim treats an empty list as a hard error; priming with an empty list
    # would defeat the purpose, so the contract is a *non-empty* candidate list.
    assert native.cudart_candidates, "cudart candidates must be primed non-empty"


def test_cudart_candidates_non_empty() -> None:
    """``_cudart_candidates`` always yields at least one entry to try.

    On both branches (tesseract-core's shared discovery, or the bare-soname
    fallback for older core) the list must be non-empty -- an empty list is the
    shim's hard-error condition.
    """
    assert gpu_ffi._cudart_candidates()
