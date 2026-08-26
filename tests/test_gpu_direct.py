# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-direct dispatch tests: the native FFI (CUDA) lowering of the primitive.

On GPU, ``apply_tesseract`` lowers ``tesseract_dispatch`` to a native XLA FFI
custom call (cuda_ipc), keeping data on the device. There is no separate entry
point -- the same ``apply_tesseract`` used everywhere routes through the FFI path
because it is the ``cuda`` platform lowering.

These require a real GPU and a served (subprocess) GPU Tesseract, since CUDA IPC
is cross-process and cannot be self-opened. Marked ``gpu``; the
``served_gpu_tesseract`` fixture skips where CUDA / CuPy / a GPU-backed JAX are
unavailable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract

pytestmark = pytest.mark.gpu


def _to_np(x):
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


def _on_gpu(x) -> bool:
    return any(d.platform == "gpu" for d in x.devices())


@pytest.mark.parametrize("n", [1, 8, 1000, 100_003])
def test_apply_matches_analytic(served_gpu_tesseract, n):
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32) * 3.0
    out = jax.jit(lambda a, b: apply_tesseract(served_gpu_tesseract, {"a": a, "b": b}))(
        a, b
    )
    c = out["c"]
    assert _on_gpu(c)
    np.testing.assert_allclose(
        _to_np(c), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6, atol=0
    )


def test_apply_matches_host_callback(served_gpu_tesseract):
    """The GPU FFI lowering must match the host-callback lowering exactly.

    We force the same computation onto CPU (host callback) and GPU (FFI) and
    compare bit-for-bit.
    """
    a = jnp.linspace(-5, 5, 257, dtype=jnp.float32)
    b = jnp.linspace(10, -10, 257, dtype=jnp.float32)

    gpu = jax.jit(
        lambda a, b: apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})["c"]
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
        return apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})["c"].sum()

    g = jax.jit(jax.grad(loss))(a)
    assert _on_gpu(g)
    # d/da sum(a*2 + b) = 2
    np.testing.assert_allclose(_to_np(g), np.full((512,), 2.0), rtol=1e-6)


def test_serial_reuse_ring1(served_gpu_tesseract):
    """Back-to-back serial dispatches: exercises the ring-1 lifetime contract."""
    f = jax.jit(
        lambda a, b: apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})["c"]
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
    GPU that endpoint returns dense CuPy arrays, so a device-correct return path
    must keep them on-device end to end. The current
    ``Jaxeract.jacobian`` builds its per-pair dtype table from
    ``_DeviceArrayView`` inputs (which expose only ``__cuda_array_interface__``)
    and wraps the server's device result in ``np.asarray`` -- the former has no
    ``.dtype``, the latter forces a device->host copy that then re-crosses the
    FFI boundary as a host pointer.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)

    def f(a):
        return apply_tesseract(
            served_gpu_tesseract, {"a": a, "b": b}, materialize_jacobian=True
        )["c"]

    jac = jax.jit(jax.jacrev(f))(a)
    assert _on_gpu(jac)
    # c = a*scale + b with default scale=2, so dc/da = 2*I.
    np.testing.assert_allclose(
        _to_np(jac), np.eye(n, dtype=np.float32) * 2.0, rtol=1e-6
    )


def test_grad_with_nondiff_array_input_through_gpu_ffi(served_gpu_tesseract):
    """A non-differentiable array input must not force a host copy on the vjp path.

    ``mask`` is passed as a *traced* argument (so it is a non-static input) but
    the gradient is taken only wrt ``a`` (``argnums=0``); since ``mask`` is
    non-differentiable in the schema, JAX still expects a (placeholder) gradient
    slot for it. ``Jaxeract.vector_jacobian_product`` fills that slot with
    ``np.full(array_args[i].shape, np.nan, dtype=array_args[i].dtype)`` -- which
    both reads ``.shape``/``.dtype`` off a ``_DeviceArrayView`` (it has neither)
    and, being a host array, cannot cross the GPU FFI return path. The real
    gradient wrt ``a`` must still come back correctly on-device.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    mask = jnp.full((n,), 3.0, dtype=jnp.float32)

    def loss(a, mask):
        out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b, "mask": mask})
        return out["c"].sum()

    g = jax.jit(jax.grad(loss, argnums=0))(a, mask)
    assert _on_gpu(g)
    # c = (a*scale + b)*mask, scale=2 => d/da sum(c) = 2*mask.
    np.testing.assert_allclose(_to_np(g), np.full((n,), 2.0 * 3.0), rtol=1e-6)


def test_jvp_with_nondiff_output_through_gpu_ffi(served_gpu_tesseract):
    """A non-differentiable *output* must not force a host copy on the jvp path.

    ``c_sum`` is a non-differentiable output, so the jvp endpoint returns no
    tangent for it and ``Jaxeract.jacobian_vector_product`` synthesises a
    placeholder for its slot. That placeholder is keyed on an output aval (there
    is no input device buffer to reuse), and previously used ``np.full(...)`` -- a
    host array that cannot cross the GPU FFI return path. Now the dispatch returns
    ``None`` and the native handler NaN-fills the slot on-device. The
    differentiable output ``c`` must still get the correct tangent, and the
    ``c_sum`` tangent must come back NaN (its poison placeholder), not garbage.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    ta = jnp.ones(n, dtype=jnp.float32)
    tb = jnp.zeros(n, dtype=jnp.float32)

    def f(a, b):
        return apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})

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


def test_host_pointer_at_ffi_boundary_errors_gracefully(
    served_gpu_tesseract, monkeypatch
):
    """A host pointer reaching the FFI boundary must fail cleanly, not crash.

    The residency check (``TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS``) exists to
    catch an accidental host copy on a dispatch return path. For it to be useful
    the failure must surface as an ``ffi::Error`` (a Python ``JaxRuntimeError``),
    not abort the process -- which it would if the native handler released the GIL
    before destroying the Python objects it kept alive across the copy.

    We inject a result whose ``__cuda_array_interface__`` points at *host* memory
    (so the pointer passes the interface check but fails the residency check) and
    assert we get an exception and the interpreter is still alive afterwards.
    """
    monkeypatch.setenv("TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS", "1")

    from tesseract_jax import gpu_ffi

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

    # Ensure the real callback is installed, then swap in a wrapper that returns a
    # host-backed result. The native shim caches the callable it was last given,
    # so we restore the original in a finally (monkeypatch can't unwind a C++
    # setter).
    gpu_ffi.ensure_registered()
    real_dispatch = gpu_ffi._native_dispatch
    native = gpu_ffi._native()

    def _corrupt_dispatch(token, inputs):
        out = real_dispatch(token, inputs)
        n = out[0].__cuda_array_interface__["shape"][0]
        return [_HostBackedCudaArray(n), *out[1:]]

    native.set_dispatch_callback(_corrupt_dispatch)
    try:
        a = jnp.arange(4, dtype=jnp.float32)
        b = jnp.ones(4, dtype=jnp.float32)
        f = jax.jit(
            lambda a, b: apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})["c"]
        )
        with pytest.raises(
            jax.errors.JaxRuntimeError, match=r"not device-resident|host"
        ):
            f(a, b).block_until_ready()
    finally:
        native.set_dispatch_callback(real_dispatch)

    # The interpreter survived the failure: a fresh trivial computation still runs.
    assert float(jnp.arange(3, dtype=jnp.float32).sum()) == 3.0
