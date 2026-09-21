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

import numpy as np
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

    An empty list is the shim's hard-error condition, so tesseract-core's shared
    discovery must always surface at least one candidate to dlopen.
    """
    assert gpu_ffi._cudart_candidates()


def test_cudart_candidates_uses_core_discovery(monkeypatch):
    """Delegate to tesseract-core's shared libcudart discovery.

    The shim must resolve the same libcudart the codec does, so
    ``_cudart_candidates`` returns exactly what ``iter_cudart_candidates`` yields.
    """
    sentinel = ["/wheel/libcudart.so.13", "libcudart.so"]

    from tesseract_core.runtime.cuda import loader

    monkeypatch.setattr(loader, "iter_cudart_candidates", lambda: iter(sentinel))

    assert gpu_ffi._cudart_candidates() == sentinel


class _StubClient:
    """Minimal stand-in for a Jaxeract: the GPU lowering only reads _device_transport."""

    _device_transport = "cuda_ipc"


def test_gpu_lowering_raises_when_transport_but_shim_unavailable(monkeypatch):
    """A device transport with an unavailable shim is a hard error, not a fallback.

    An explicit device_transport opt-in must not silently degrade to the slow host
    path; the lowering raises before touching ctx/array_args.
    """
    from types import SimpleNamespace

    import typeguard

    from tesseract_jax import primitive

    monkeypatch.setattr(gpu_ffi, "is_available", lambda: False)
    # The guard only reads params.client._device_transport before raising, so a
    # stub suffices; suppress typeguard's runtime check of the DispatchParams
    # annotation (armed for the whole package via --typeguard-packages).
    params = SimpleNamespace(client=_StubClient())

    with (
        typeguard.suppress_type_checks(),
        pytest.raises(RuntimeError, match="device_transport='cuda_ipc' was requested"),
    ):
        primitive.tesseract_dispatch_gpu_lowering(object(), params=params)


def test_gpu_lowering_raises_when_transport_but_no_cuda_device(monkeypatch):
    """A device transport with no CUDA device visible to JAX is a hard error.

    The shim can import on a CPU-only host, so an available shim does not imply a
    usable GPU. Selecting a transport and then lowering without a CUDA device is a
    misconfiguration the lowering surfaces rather than silently falling back.
    """
    from types import SimpleNamespace

    import jax
    import typeguard

    from tesseract_jax import primitive

    # Get past the shim-availability guard so the device check is what fires.
    monkeypatch.setattr(gpu_ffi, "is_available", lambda: True)

    def no_cuda(*args, **kwargs):
        raise RuntimeError("Unknown backend: 'cuda'")

    monkeypatch.setattr(jax, "devices", no_cuda)
    params = SimpleNamespace(client=_StubClient())

    with (
        typeguard.suppress_type_checks(),
        pytest.raises(RuntimeError, match="JAX sees no CUDA device"),
    ):
        primitive.tesseract_dispatch_gpu_lowering(object(), params=params)


def test_gpu_lowering_falls_back_to_host_without_transport(monkeypatch):
    """No device transport selected: defer to the host-callback lowering.

    A client that did not opt into a device transport must behave exactly as on
    CPU, so the GPU lowering delegates straight to the host lowering.
    """
    from types import SimpleNamespace

    import typeguard

    from tesseract_jax import primitive

    class _HostClient:
        _device_transport = None

    sentinel = object()
    seen: dict = {}

    def fake_host_lowering(ctx, *array_args, params):
        seen["ctx"] = ctx
        seen["array_args"] = array_args
        return sentinel

    monkeypatch.setattr(primitive, "tesseract_dispatch_lowering", fake_host_lowering)
    params = SimpleNamespace(client=_HostClient())

    with typeguard.suppress_type_checks():
        result = primitive.tesseract_dispatch_gpu_lowering(
            "ctx", "arg0", "arg1", params=params
        )

    assert result is sentinel
    assert seen == {"ctx": "ctx", "array_args": ("arg0", "arg1")}


def test_is_available_false_when_shim_import_fails(monkeypatch):
    """``is_available`` reports False when the shim import fails.

    A CPU-only install has no compiled shim, so callers must be able to probe
    availability cheaply without the import error propagating.
    """
    import builtins
    import sys

    # A cached shim would satisfy the import without hitting the patched importer.
    monkeypatch.delitem(sys.modules, "tesseract_jax._cuda_shim", raising=False)

    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        # ``from tesseract_jax import _cuda_shim`` imports the package with
        # ``_cuda_shim`` in the fromlist, so match on either form.
        if name.endswith("_cuda_shim") or "_cuda_shim" in (fromlist or ()):
            raise ImportError("no compiled shim on this platform")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert gpu_ffi.is_available() is False


def test_register_dispatch_dedupes_on_hashable_key(fresh_registry, monkeypatch):
    """A repeated hashable key reuses its token instead of leaking a fresh one.

    Each registry entry pins the dispatch closure (and the client and session
    behind it) for the process lifetime, so re-lowering the same program point
    must collapse to a single entry.
    """
    monkeypatch.setattr(gpu_ffi, "_registry", {})
    monkeypatch.setattr(gpu_ffi, "_token_by_key", {})
    monkeypatch.setattr(gpu_ffi, "_next_token", 0)

    def fn(views):
        return ()

    first = gpu_ffi.register_dispatch(fn, key="dispatch-a")
    again = gpu_ffi.register_dispatch(fn, key="dispatch-a")
    other = gpu_ffi.register_dispatch(fn, key="dispatch-b")

    assert first == again
    assert other != first
    assert set(gpu_ffi._registry) == {first, other}


def test_register_dispatch_unhashable_key_allocates_fresh(fresh_registry, monkeypatch):
    """An unhashable key skips dedup and always allocates a new token.

    Dedup is a best-effort optimisation keyed on a frozen, value-equal object; an
    unhashable key must degrade to a fresh registration rather than raising.
    """
    monkeypatch.setattr(gpu_ffi, "_registry", {})
    monkeypatch.setattr(gpu_ffi, "_token_by_key", {})
    monkeypatch.setattr(gpu_ffi, "_next_token", 0)

    def fn(views):
        return ()

    first = gpu_ffi.register_dispatch(fn, key=["unhashable"])
    second = gpu_ffi.register_dispatch(fn, key=["unhashable"])

    assert first != second
    assert set(gpu_ffi._registry) == {first, second}
    # Nothing was recorded in the dedup map for an unhashable key.
    assert gpu_ffi._token_by_key == {}


def test_device_array_view_exposes_cuda_array_interface() -> None:
    """``_DeviceArrayView`` presents a raw pointer as a zero-copy CUDA array.

    The cuda_ipc encoder consumes only ``__cuda_array_interface__`` plus the
    ``shape`` and ``dtype`` properties, so those must reflect the pointer,
    C-contiguous layout, and dtype it was built with.
    """
    view = gpu_ffi._DeviceArrayView(0xDEADBEEF, "<f4", (2, 3))

    cai = view.__cuda_array_interface__
    assert cai["shape"] == (2, 3)
    assert cai["typestr"] == "<f4"
    assert cai["data"] == (0xDEADBEEF, False)
    assert cai["strides"] is None  # XLA hands us C-contiguous buffers
    assert cai["version"] == 3
    assert view.shape == (2, 3)
    assert view.dtype == np.dtype("<f4")


def test_native_dispatch_unknown_token_raises(monkeypatch):
    """An unknown token from the native side is a hard error.

    The handler passes the token it was lowered with; a token missing from the
    registry means state was corrupted or cleared out from under a live
    executable, so the callback raises rather than returning garbage buffers.
    """
    monkeypatch.setattr(gpu_ffi, "_registry", {})

    with pytest.raises(RuntimeError, match="unknown dispatch token 123"):
        gpu_ffi._native_dispatch(123, [])


def test_native_dispatch_wraps_inputs_as_views(monkeypatch):
    """The callback wraps each XLA input buffer as a ``_DeviceArrayView``.

    The native shim marshals inputs as ``(ptr, typestr, shape)`` with ``shape`` a
    list; the callback must normalise them into views the dispatch closure reads
    through ``__cuda_array_interface__``, then return the closure's outputs as a
    list for the handler to copy back.
    """
    captured: dict = {}
    out_arrays = (np.zeros(1),)

    def fn(views):
        captured["views"] = views
        return out_arrays

    monkeypatch.setattr(gpu_ffi, "_registry", {7: fn})

    result = gpu_ffi._native_dispatch(7, [(0x1000, "<f8", [2, 2])])

    assert result == list(out_arrays)
    (view,) = captured["views"]
    assert isinstance(view, gpu_ffi._DeviceArrayView)
    assert view.shape == (2, 2)  # list shape normalised to a tuple
    assert view.dtype == np.dtype("<f8")
