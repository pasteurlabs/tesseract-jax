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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

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
