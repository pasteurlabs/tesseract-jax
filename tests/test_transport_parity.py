# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-sensitive behaviours that must agree across dispatch transports and producers.

Each test runs on every leg of the ``transport`` fixture (``tests/conftest.py``),
which are the host callback (device->host->device) and the GPU-direct cuda_ipc FFI
path against a Tesseract computing with JAX and one computing with CuPy. The
cuda_ipc legs are ``gpu``-marked, so on the CPU runner only the host leg runs.

Transport-specific machinery (FFI-boundary fault injection, on-device placement
assertions) has no host analogue and lives in ``test_gpu_direct.py``. What belongs
here is behaviour that should be the same on every path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract
from tesseract_jax.testing import to_numpy


def test_apply_matches_analytic(transport):
    a = jnp.arange(64, dtype=jnp.float32)
    b = jnp.ones(64, dtype=jnp.float32) * 3.0
    out = jax.jit(lambda a, b: apply_tesseract(transport, {"a": a, "b": b})["c"])(a, b)
    np.testing.assert_allclose(
        to_numpy(out), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6, atol=0
    )


def test_mask_nondiff_array_input(transport):
    """A non-differentiable, non-static array input flows through both paths."""
    a = jnp.arange(32, dtype=jnp.float32)
    b = jnp.ones(32, dtype=jnp.float32)
    mask = jnp.full((32,), 3.0, dtype=jnp.float32)

    def f(a, b, mask):
        out = apply_tesseract(transport, {"a": a, "b": b, "mask": mask})
        return out["c"]

    out = jax.jit(f)(a, b, mask)
    np.testing.assert_allclose(
        to_numpy(out), (np.asarray(a) * 2.0 + np.asarray(b)) * 3.0, rtol=1e-6
    )


def test_grad_with_nondiff_array_input(transport):
    """The gradient wrt a differentiable input is correct despite a non-diff input.

    ``mask`` is a traced (non-static) but non-differentiable input, so JAX still
    expects a placeholder gradient slot for it. Both transports must fill that
    slot without corrupting the real gradient wrt ``a``.
    """
    n = 16
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    mask = jnp.full((n,), 3.0, dtype=jnp.float32)

    def loss(a, mask):
        return apply_tesseract(transport, {"a": a, "b": b, "mask": mask})["c"].sum()

    g = jax.jit(jax.grad(loss, argnums=0))(a, mask)
    # c = (a*scale + b)*mask, scale=2 => d/da sum(c) = 2*mask.
    np.testing.assert_allclose(to_numpy(g), np.full((n,), 2.0 * 3.0), rtol=1e-6)


def test_jvp_nondiff_output_placeholder_is_nan(transport):
    """The discarded slot for a non-differentiable output comes back NaN, both paths.

    ``c_sum`` is non-differentiable, so the jvp endpoint returns no tangent for it
    and the dispatch synthesises a placeholder. Host and cuda_ipc fill it the same
    way (a NaN poison), and neither may corrupt the real tangent for ``c``.
    """
    n = 16
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    ta = jnp.ones(n, dtype=jnp.float32)
    tb = jnp.zeros(n, dtype=jnp.float32)

    def f(a, b):
        return apply_tesseract(transport, {"a": a, "b": b})

    primal, tangent = jax.jit(lambda a, b, ta, tb: jax.jvp(f, (a, b), (ta, tb)))(
        a, b, ta, tb
    )
    np.testing.assert_allclose(
        to_numpy(primal["c"]), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6
    )
    # dc = scale*da + db = 2*ta + tb = 2.
    np.testing.assert_allclose(to_numpy(tangent["c"]), np.full((n,), 2.0), rtol=1e-6)
    # The non-differentiable c_sum's tangent is the discarded placeholder: NaN.
    assert np.all(np.isnan(to_numpy(tangent["c_sum"])))


@pytest.mark.parametrize("jac", [jax.jacfwd, jax.jacrev], ids=["fwd", "bwd"])
def test_materialized_jacobian_modes(transport, jac):
    """Both jacobian modes agree across transports.

    ``materialize_jacobian=True`` routes to the Tesseract's ``jacobian`` endpoint.
    ``jacfwd`` (fwd) and ``jacrev`` (bwd) take different dtype rules in the return
    cast (output vs input dtype), so both are exercised on both transports here --
    the fwd leg is otherwise untested on GPU.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)

    def f(a):
        out = apply_tesseract(transport, {"a": a, "b": b}, materialize_jacobian=True)
        return out["c"]

    out = jax.jit(jac(f))(a)
    # c = a*scale + b, scale=2 => dc/da = 2*I.
    np.testing.assert_allclose(
        to_numpy(out), np.eye(n, dtype=np.float32) * 2.0, rtol=1e-6
    )


def test_vmap_batching(transport):
    """A batched (vmapped) dispatch matches the per-row analytic result, both paths."""
    batch, n = 4, 16
    a = jnp.arange(batch * n, dtype=jnp.float32).reshape(batch, n)
    b = jnp.ones((batch, n), dtype=jnp.float32) * 3.0

    def f(a, b):
        out = apply_tesseract(transport, {"a": a, "b": b}, vmap_method="sequential")
        return out["c"]

    out = jax.jit(jax.vmap(f))(a, b)
    np.testing.assert_allclose(
        to_numpy(out), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6, atol=0
    )


def test_input_cast_to_schema_dtype(transport):
    """Inputs whose dtype differs from their ``Float32`` fields are cast, on both paths.

    tesseract-core casts these when it decodes a host array but rejects a device
    array sent over cuda_ipc, so tesseract-jax casts them before the dispatch.
    ``mask`` is an optional field, whose schema wraps the array in ``anyOf``.
    """
    n = 8
    a = jnp.arange(n, dtype=jnp.int32)
    b = jnp.ones(n, dtype=jnp.float16)
    mask = jnp.full((n,), 3.0, dtype=jnp.float64)

    def f(a, b, mask):
        return apply_tesseract(transport, {"a": a, "b": b, "mask": mask})["c"]

    out = jax.jit(f)(a, b, mask)
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(
        to_numpy(out), (np.arange(n) * 2.0 + 1.0) * 3.0, rtol=1e-6
    )


@pytest.mark.parametrize("use_jit", [True, False], ids=["jit", "eager"])
def test_grad_wrt_float64_inputs(transport, use_jit):
    """Gradients wrt float64 inputs to ``Float32`` fields come back in float64.

    JAX differentiates through the cast to the schema dtype, so the Tesseract
    sees float32 cotangents and the caller gets its own dtype back.
    """
    n = 8
    a = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
    b = jnp.ones(n, dtype=jnp.float64)

    def loss(a, b):
        return apply_tesseract(transport, {"a": a, "b": b})["c"].sum()

    grad = jax.grad(loss, argnums=(0, 1))
    if use_jit:
        grad = jax.jit(grad)
    ga, gb = grad(a, b)
    assert ga.dtype == gb.dtype == jnp.float64
    # c = a*scale + b, scale=2.
    np.testing.assert_allclose(to_numpy(ga), np.full((n,), 2.0))
    np.testing.assert_allclose(to_numpy(gb), np.ones(n))


def test_jvp_wrt_float64_inputs(transport):
    """float64 primals and tangents for ``Float32`` fields are cast on the way in."""
    n = 8
    a = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
    b = jnp.ones(n, dtype=jnp.float64)

    def f(a, b):
        return apply_tesseract(transport, {"a": a, "b": b})["c"]

    primal, tangent = jax.jit(lambda a, b: jax.jvp(f, (a, b), (a, b)))(a, b)
    assert primal.dtype == tangent.dtype == jnp.float32
    # dc = scale*da + db.
    np.testing.assert_allclose(to_numpy(tangent), np.asarray(a) * 2.0 + 1.0, rtol=1e-6)


@pytest.mark.parametrize("materialize_jacobian", [True, False])
@pytest.mark.parametrize("jac", [jax.jacfwd, jax.jacrev], ids=["fwd", "bwd"])
def test_jacobian_wrt_float64_input(transport, jac, materialize_jacobian):
    """Jacobians wrt a float64 input to a ``Float32`` field take JAX's dtypes.

    ``jacfwd`` gives the output's float32 and ``jacrev`` the caller's float64.
    The Tesseract itself only sees float32, so the materialized ``jacobian``
    endpoint's result needs no cast on either transport.
    """
    n = 6
    a = jnp.arange(n, dtype=jnp.float64)
    b = jnp.ones(n, dtype=jnp.float32)

    def f(a):
        return apply_tesseract(
            transport,
            {"a": a, "b": b},
            vmap_method="sequential",
            materialize_jacobian=materialize_jacobian,
        )["c"]

    out = jax.jit(jac(f))(a)
    assert out.dtype == (jnp.float32 if jac is jax.jacfwd else jnp.float64)
    np.testing.assert_allclose(to_numpy(out), np.eye(n) * 2.0, rtol=1e-6)
