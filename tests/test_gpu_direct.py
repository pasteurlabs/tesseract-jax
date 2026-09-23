# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-direct dispatch tests: the native FFI (CUDA) lowering of the primitive.

On GPU, ``apply_tesseract(..., gpu_transport="cuda_ipc")`` lowers
``tesseract_dispatch`` to a native XLA FFI custom call, keeping data on the
device. There is no separate entry point: the ``cuda`` platform lowering routes
through the FFI path when the call selected a device transport, and without one
it falls back to the host-callback lowering (a device->host->device round-trip).
So every test here passes ``gpu_transport="cuda_ipc"``.

These require a real GPU and a served (subprocess) GPU Tesseract, since CUDA IPC
is cross-process and cannot be self-opened. Marked ``gpu``; the
``served_gpu_tesseract`` fixture skips where CUDA / CuPy / a GPU-backed JAX are
unavailable.
"""

from __future__ import annotations

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def _arm_residency_check():
    """Arm the FFI residency check for every GPU-direct test in this module.

    ``TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS`` makes the native handler reject any
    buffer that crosses the FFI boundary in host memory. The shim reads the flag
    once on its first dispatch, so it must be set before this module runs. With
    it armed, every assertion below (apply / grad / jvp / vjp / jacobian) also
    guards against a host copy sneaking back onto a dispatch path.
    """
    os.environ["TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS"] = "1"


def _to_np(x):
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


def _on_gpu(x) -> bool:
    return any(d.platform == "gpu" for d in x.devices())


@contextlib.contextmanager
def _patched_dispatch(wrap):
    """Swap the native dispatch callback for the duration of the block.

    ``wrap(real_dispatch)`` returns the replacement callback. The native shim
    caches the callable it was last given, so the original is restored on exit in
    a finally (monkeypatch cannot unwind a C++ setter). Used by the FFI-boundary
    fault-injection tests below to feed the handler a malformed dispatch result.
    """
    from tesseract_jax import gpu_ffi

    gpu_ffi.ensure_registered()
    native = gpu_ffi._native()
    real_dispatch = gpu_ffi._native_dispatch
    native.set_dispatch_callback(wrap(real_dispatch))
    try:
        yield
    finally:
        native.set_dispatch_callback(real_dispatch)


def _assert_interpreter_alive():
    """A fresh trivial computation still runs after a handled FFI failure."""
    assert float(jnp.arange(3, dtype=jnp.float32).sum()) == 3.0


def _apply_c(served_gpu_tesseract, n):
    """A jitted ``apply`` returning ``c`` for ``n``-element float32 inputs."""
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    f = jax.jit(
        lambda a, b: apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )["c"]
    )
    return lambda: f(a, b).block_until_ready()


@pytest.mark.parametrize("n", [1, 8, 1000, 100_003])
def test_apply_matches_analytic(served_gpu_tesseract, n):
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32) * 3.0
    out = jax.jit(
        lambda a, b: apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )
    )(a, b)
    c = out["c"]
    assert _on_gpu(c)
    np.testing.assert_allclose(
        _to_np(c), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6, atol=0
    )


def test_apply_mixed_dtype_stays_on_device(served_gpu_mixed_dtype_tesseract):
    """A float32-in/float64-out Tesseract round-trips on-device with the right dtype.

    ``gpu_tesseract`` is all-float32, so the shim's per-buffer dtype handling is
    otherwise only exercised for one width. Here apply returns float64, so a
    correct copy must move the wider buffer and preserve the dtype through the
    cuda_ipc return path.
    """
    x = jnp.arange(64, dtype=jnp.float32)
    y = jax.jit(
        lambda x: apply_tesseract(
            served_gpu_mixed_dtype_tesseract, {"x": x}, gpu_transport="cuda_ipc"
        )["y"]
    )(x)

    assert _on_gpu(y)
    assert _to_np(y).dtype == np.float64
    np.testing.assert_allclose(_to_np(y), np.asarray(x) * 2.0, rtol=0, atol=0)


def test_apply_dtype_mismatch_at_ffi_boundary_errors(served_gpu_tesseract):
    """A returned dtype disagreeing with XLA's output buffer must raise, not reinterpret.

    XLA sizes each output buffer from tesseract-jax's declared avals, but the
    Tesseract is not forced to return that dtype (an unconstrained output schema
    is only dtype-validated server-side when it fully pins the dtype). Unlike the
    host path the shim cannot cast, so it compares each result's
    ``__cuda_array_interface__`` dtype against the expected buffer and raises
    rather than copying a narrower source (an out-of-bounds device read) or
    silently reinterpreting a same-itemsize swap.

    tesseract-core validates a pinned output dtype server-side, so a genuinely
    dtype-lying Tesseract is rejected before its bytes reach the shim. To drive
    the shim's own guard we intercept at the dispatch callback and relabel the
    real device buffer's dtype.
    """

    def _wrong_dtype_dispatch(real_dispatch):
        def dispatch(token, inputs):
            out = real_dispatch(token, inputs)
            # Re-expose the first result's device buffer with a dtype that
            # disagrees with the float32 buffer XLA allocated: same pointer/shape.
            cai = dict(out[0].__cuda_array_interface__)
            cai["typestr"] = "<f8"

            class _WrongDtype:
                __cuda_array_interface__ = cai

            return [_WrongDtype(), *out[1:]]

        return dispatch

    with _patched_dispatch(_wrong_dtype_dispatch):
        run = _apply_c(served_gpu_tesseract, 8)
        with pytest.raises(jax.errors.JaxRuntimeError, match=r"expected|returned"):
            run()

    _assert_interpreter_alive()


def test_apply_matches_host_callback(served_gpu_tesseract):
    """The GPU FFI lowering must match the host-callback lowering exactly.

    We force the same computation onto CPU (host callback) and GPU (FFI) and
    compare bit-for-bit.
    """
    a = jnp.linspace(-5, 5, 257, dtype=jnp.float32)
    b = jnp.linspace(10, -10, 257, dtype=jnp.float32)

    gpu = jax.jit(
        lambda a, b: apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )["c"]
    )(a, b)
    with jax.default_device(jax.devices("cpu")[0]):
        a_cpu = jnp.asarray(np.asarray(a))
        b_cpu = jnp.asarray(np.asarray(b))
        cpu = apply_tesseract(served_gpu_tesseract, {"a": a_cpu, "b": b_cpu})["c"]

    assert _on_gpu(gpu)
    np.testing.assert_array_equal(_to_np(gpu), np.asarray(cpu))


def test_grad_through_gpu_ffi(served_gpu_tesseract):
    """Derivatives dispatch generically through the same FFI path (vjp)."""
    a = jnp.arange(512, dtype=jnp.float32)
    b = jnp.ones(512, dtype=jnp.float32)

    def loss(a):
        return apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )["c"].sum()

    g = jax.jit(jax.grad(loss))(a)
    assert _on_gpu(g)
    # d/da sum(a*2 + b) = 2
    np.testing.assert_allclose(_to_np(g), np.full((512,), 2.0), rtol=1e-6)


def test_serial_reuse_ring1(served_gpu_tesseract):
    """Back-to-back serial dispatches: exercises the ring-1 lifetime contract."""
    f = jax.jit(
        lambda a, b: apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )["c"]
    )
    for i in range(20):
        a = jnp.full((512,), float(i), dtype=jnp.float32)
        b = jnp.full((512,), float(2 * i), dtype=jnp.float32)
        out = f(a, b)
        out.block_until_ready()
        np.testing.assert_allclose(
            _to_np(out), np.full((512,), i * 2.0 + 2 * i, dtype=np.float32), rtol=1e-6
        )


def test_materialized_jacobian_through_gpu_ffi(served_gpu_tesseract):
    """A materialized ``jacobian`` endpoint must dispatch through the FFI on device.

    Forcing ``materialize_jacobian=True`` routes ``jacfwd``/``jacrev`` to the
    Tesseract's ``jacobian`` endpoint rather than the batched jvp/vjp path. On
    GPU that endpoint returns dense CuPy arrays, which must stay on-device end to
    end rather than round-tripping through a host cast.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)

    def f(a):
        return apply_tesseract(
            served_gpu_tesseract,
            {"a": a, "b": b},
            materialize_jacobian=True,
            gpu_transport="cuda_ipc",
        )["c"]

    jac = jax.jit(jax.jacrev(f))(a)
    assert _on_gpu(jac)
    # c = a*scale + b with default scale=2, so dc/da = 2*I.
    np.testing.assert_allclose(
        _to_np(jac), np.eye(n, dtype=np.float32) * 2.0, rtol=1e-6
    )


def test_grad_with_nondiff_array_input_through_gpu_ffi(served_gpu_tesseract):
    """A non-differentiable array input must not force a host copy on the vjp path.

    ``mask`` is a traced (non-static) argument, but the gradient is taken only
    wrt ``a`` (``argnums=0``). Since ``mask`` is non-differentiable in the
    schema, JAX still expects a placeholder gradient slot for it, which the vjp
    path must fill without materializing a host array. The real gradient wrt
    ``a`` must still come back correctly on-device.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    mask = jnp.full((n,), 3.0, dtype=jnp.float32)

    def loss(a, mask):
        out = apply_tesseract(
            served_gpu_tesseract,
            {"a": a, "b": b, "mask": mask},
            gpu_transport="cuda_ipc",
        )
        return out["c"].sum()

    g = jax.jit(jax.grad(loss, argnums=0))(a, mask)
    assert _on_gpu(g)
    # c = (a*scale + b)*mask, scale=2 => d/da sum(c) = 2*mask.
    np.testing.assert_allclose(_to_np(g), np.full((n,), 2.0 * 3.0), rtol=1e-6)


def test_jvp_with_nondiff_output_through_gpu_ffi(served_gpu_tesseract):
    """A non-differentiable *output* must not force a host copy on the jvp path.

    ``c_sum`` is a non-differentiable output, so the jvp endpoint returns no
    tangent for it and ``Jaxeract.jacobian_vector_product`` returns ``None`` for
    its slot, which the native handler NaN-fills on-device. The differentiable
    output ``c`` must still get the correct tangent, and the ``c_sum`` tangent
    must come back NaN (its poison placeholder), not garbage.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    ta = jnp.ones(n, dtype=jnp.float32)
    tb = jnp.zeros(n, dtype=jnp.float32)

    def f(a, b):
        return apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )

    primal, tangent = jax.jit(lambda a, b, ta, tb: jax.jvp(f, (a, b), (ta, tb)))(
        a, b, ta, tb
    )
    assert _on_gpu(tangent["c"])
    # The discarded placeholder for c_sum must not corrupt the real results:
    # primal c = a*scale + b (scale=2), and dc = scale*da + db = 2*ta + tb = 2.
    np.testing.assert_allclose(
        _to_np(primal["c"]), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6
    )
    np.testing.assert_allclose(_to_np(tangent["c"]), np.full((n,), 2.0), rtol=1e-6)
    # The non-differentiable c_sum output's tangent is the discarded placeholder:
    # it must come back NaN-filled on-device, matching the host path's poison.
    assert _on_gpu(tangent["c_sum"])
    assert np.all(np.isnan(_to_np(tangent["c_sum"])))


def test_host_pointer_at_ffi_boundary_errors_gracefully(served_gpu_tesseract):
    """A host pointer reaching the FFI boundary must fail cleanly, not crash.

    The residency check (armed for this module by ``_arm_residency_check``) exists
    to catch an accidental host copy on a dispatch return path. For it to be useful
    the failure must surface as an ``ffi::Error`` (a Python ``JaxRuntimeError``),
    not abort the process -- which it would if the native handler released the GIL
    before destroying the Python objects it kept alive across the copy.

    We inject a result whose ``__cuda_array_interface__`` points at *host* memory
    (so the pointer passes the interface check but fails the residency check) and
    assert we get an exception and the interpreter is still alive afterwards.
    """

    class _HostBackedCudaArray:
        """Exposes ``__cuda_array_interface__`` but backed by host memory."""

        def __init__(self, n: int) -> None:
            self._buf = np.full(n, np.nan, dtype=np.float32)
            self.__cuda_array_interface__ = {
                "shape": (n,),
                "typestr": "<f4",
                "data": (self._buf.ctypes.data, False),
                "strides": None,
                "version": 3,
            }

    def _corrupt_dispatch(real_dispatch):
        def dispatch(token, inputs):
            out = real_dispatch(token, inputs)
            n = out[0].__cuda_array_interface__["shape"][0]
            return [_HostBackedCudaArray(n), *out[1:]]

        return dispatch

    with _patched_dispatch(_corrupt_dispatch):
        run = _apply_c(served_gpu_tesseract, 4)
        with pytest.raises(
            jax.errors.JaxRuntimeError, match=r"not device-resident|host"
        ):
            run()

    _assert_interpreter_alive()


def test_result_shape_mismatch_at_ffi_boundary_errors(served_gpu_tesseract):
    """A returned array whose shape disagrees with XLA's output buffer must raise.

    XLA sizes each output buffer from tesseract-jax's declared avals, but the
    Tesseract is not forced to return that shape (an unconstrained output schema
    is only shape-validated server-side when it fully pins the shape). The native
    handler compares each result's ``__cuda_array_interface__`` shape/dtype
    against the expected buffer and raises rather than copying mismatched bytes
    (which would over- or under-read device memory). We inject a truncated result
    to drive that path; the process must survive the clean error.
    """

    def _wrong_shape_dispatch(real_dispatch):
        def dispatch(token, inputs):
            out = real_dispatch(token, inputs)
            # Re-expose the first result's device buffer with a shape one element
            # short of what XLA allocated: same dtype, wrong (smaller) shape.
            cai = dict(out[0].__cuda_array_interface__)
            n = cai["shape"][0]
            cai["shape"] = (n - 1,)

            class _WrongShape:
                __cuda_array_interface__ = cai

            return [_WrongShape(), *out[1:]]

        return dispatch

    with _patched_dispatch(_wrong_shape_dispatch):
        run = _apply_c(served_gpu_tesseract, 8)
        with pytest.raises(jax.errors.JaxRuntimeError, match=r"expected|returned"):
            run()

    _assert_interpreter_alive()


def test_mixed_cpu_and_gpu_tesseracts_in_one_graph(
    served_gpu_tesseract, served_vectoradd_tesseract
):
    """A single jitted graph can mix a GPU-direct and a host-callback dispatch.

    The GPU Tesseract runs via cuda_ipc (device-resident, residency-checked), and
    its output feeds a CPU Tesseract dispatched over the host transport (no
    gpu_transport), which takes the usual device->host->device round-trip. The
    two lower to different custom calls and compose without interfering.
    """
    cpu_tess = Tesseract.from_url(served_vectoradd_tesseract)

    a = jnp.arange(64, dtype=jnp.float32)
    b = jnp.ones(64, dtype=jnp.float32) * 3.0

    def pipeline(a, b):
        gpu_out = apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )["c"]  # a*2 + b, on-device
        return apply_tesseract(cpu_tess, {"a": gpu_out, "b": b})["c"]  # + b, host

    out = jax.jit(pipeline)(a, b)
    expected = (np.asarray(a) * 2.0 + np.asarray(b)) + np.asarray(b)
    np.testing.assert_allclose(_to_np(out), expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("n", [100_000, 10_000_000])
def test_bench_apply_gpu_direct(benchmark, served_gpu_tesseract, n):
    """Timing guard for the GPU-direct ``apply`` path.

    Measures steady-state per-call latency of a jitted ``apply_tesseract`` whose
    inputs live on the GPU, so the ``cuda`` lowering routes through the native FFI
    path. It lives with the other ``gpu``-marked tests because it needs the
    ``served_gpu_tesseract`` fixture and a real device, unlike the CPU-only
    ``benchmarks/`` suite.

    This is a regression signal, not a hard gate: ``pytest-benchmark`` records
    the median so a slowdown shows up in the timing report, but shared-runner
    noise makes a fixed wall-clock threshold too flaky to fail CI on. The
    deterministic guard against a silent host round-trip is the residency check
    armed by ``_arm_residency_check`` above.
    """
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)

    f = jax.jit(
        lambda a, b: apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="cuda_ipc"
        )["c"]
    )
    # Warm up tracing/compilation so the timed loop measures steady-state latency.
    f(a, b).block_until_ready()
    assert _on_gpu(f(a, b))

    benchmark(lambda: f(a, b).block_until_ready())
