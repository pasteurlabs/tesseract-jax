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

from importlib.metadata import version as _pkg_version

import pytest
from packaging.version import Version

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


@pytest.mark.skipif(
    Version(_pkg_version("tesseract-core")) < Version(gpu_ffi._CUDART_LOADER_MIN_CORE),
    reason="tesseract-core predates the iter_cudart_candidates loader",
)
def test_cudart_candidates_uses_core_discovery_above_floor(monkeypatch):
    """At/above the version floor, delegate to tesseract-core's shared discovery.

    The shim must resolve the same libcudart the codec does; above the floor
    that means calling ``iter_cudart_candidates`` rather than the bare-soname
    fallback.
    """
    monkeypatch.setattr(
        gpu_ffi, "_pkg_version", lambda _name: gpu_ffi._CUDART_LOADER_MIN_CORE
    )
    sentinel = ["/wheel/libcudart.so.13", "libcudart.so"]

    from tesseract_core.runtime.cuda import loader

    monkeypatch.setattr(loader, "iter_cudart_candidates", lambda: iter(sentinel))

    assert gpu_ffi._cudart_candidates() == sentinel


def test_cudart_candidates_falls_back_below_floor(monkeypatch):
    """Below the version floor, use the bare-soname fallback.

    Older tesseract-core lacks ``iter_cudart_candidates``; the helper must not
    import it and must return the static fallback instead.
    """
    monkeypatch.setattr(gpu_ffi, "_pkg_version", lambda _name: "1.12.0")

    assert gpu_ffi._cudart_candidates() == list(gpu_ffi._CUDART_SONAME_FALLBACK)


class _StubClient:
    """Minimal stand-in for a Jaxeract: the GPU lowering only reads _cuda_ipc."""

    _cuda_ipc = True


def test_gpu_lowering_raises_when_cuda_ipc_but_shim_unavailable(monkeypatch):
    """cuda_ipc=True with an unavailable shim is a hard error, not a fallback.

    An explicit cuda_ipc opt-in must not silently degrade to the slow host path;
    the lowering raises before touching ctx/array_args.
    """
    from types import SimpleNamespace

    import typeguard

    from tesseract_jax import primitive

    monkeypatch.setattr(gpu_ffi, "is_available", lambda: False)
    # The guard only reads params.client._cuda_ipc before raising, so a stub
    # suffices; suppress typeguard's runtime check of the DispatchParams
    # annotation (armed for the whole package via --typeguard-packages).
    params = SimpleNamespace(client=_StubClient())

    with (
        typeguard.suppress_type_checks(),
        pytest.raises(RuntimeError, match="cuda_ipc=True was requested"),
    ):
        primitive.tesseract_dispatch_gpu_lowering(object(), params=params)
