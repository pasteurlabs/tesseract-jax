# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-sensitive behaviours that must agree across dispatch transports.

Each test here runs on both dispatch lowerings via the ``transport`` fixture
(``tests/conftest.py``): the host callback (device->host->device) and the
GPU-direct cuda_ipc FFI path. The fixture yields ``(client, apply_kwargs)`` and
serves the array-agnostic ``transport_tesseract`` with numpy or cupy compute to
match, so one body pins that the two paths compute -- and, crucially, handle their
edge cases -- identically.

The cuda_ipc parameter is ``gpu``-marked and skips without a GPU, so on the CPU
runner only the host leg executes; the GPU CI job runs both.

Transport-specific machinery (residency checks, FFI-boundary fault injection,
on-device placement assertions) has no host analogue and lives in
``test_gpu_direct.py`` instead. What belongs here is behaviour that should be
*the same* on both paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract


def _to_np(x):
    """Host copy of a result array, whether it is a cupy or a jax/numpy array."""
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


def _apply(transport, inputs, **kwargs):
    client, apply_kwargs = transport
    return apply_tesseract(client, inputs, **apply_kwargs, **kwargs)


def test_apply_matches_analytic(transport):
    a = jnp.arange(64, dtype=jnp.float32)
    b = jnp.ones(64, dtype=jnp.float32) * 3.0
    out = jax.jit(lambda a, b: _apply(transport, {"a": a, "b": b})["c"])(a, b)
    np.testing.assert_allclose(
        _to_np(out), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6, atol=0
    )


def test_mask_nondiff_array_input(transport):
    """A non-differentiable, non-static array input flows through both paths."""
    a = jnp.arange(32, dtype=jnp.float32)
    b = jnp.ones(32, dtype=jnp.float32)
    mask = jnp.full((32,), 3.0, dtype=jnp.float32)
    out = jax.jit(
        lambda a, b, mask: _apply(transport, {"a": a, "b": b, "mask": mask})["c"]
    )(a, b, mask)
    np.testing.assert_allclose(
        _to_np(out), (np.asarray(a) * 2.0 + np.asarray(b)) * 3.0, rtol=1e-6
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
        return _apply(transport, {"a": a, "b": b, "mask": mask})["c"].sum()

    g = jax.jit(jax.grad(loss, argnums=0))(a, mask)
    # c = (a*scale + b)*mask, scale=2 => d/da sum(c) = 2*mask.
    np.testing.assert_allclose(_to_np(g), np.full((n,), 2.0 * 3.0), rtol=1e-6)


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
        return _apply(transport, {"a": a, "b": b})

    primal, tangent = jax.jit(lambda a, b, ta, tb: jax.jvp(f, (a, b), (ta, tb)))(
        a, b, ta, tb
    )
    np.testing.assert_allclose(
        _to_np(primal["c"]), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6
    )
    # dc = scale*da + db = 2*ta + tb = 2.
    np.testing.assert_allclose(_to_np(tangent["c"]), np.full((n,), 2.0), rtol=1e-6)
    # The non-differentiable c_sum's tangent is the discarded placeholder: NaN.
    assert np.all(np.isnan(_to_np(tangent["c_sum"])))


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
        return _apply(transport, {"a": a, "b": b}, materialize_jacobian=True)["c"]

    out = jax.jit(jac(f))(a)
    # c = a*scale + b, scale=2 => dc/da = 2*I.
    np.testing.assert_allclose(
        _to_np(out), np.eye(n, dtype=np.float32) * 2.0, rtol=1e-6
    )


def test_vmap_batching(transport):
    """A batched (vmapped) dispatch matches the per-row analytic result, both paths."""
    batch, n = 4, 16
    a = jnp.arange(batch * n, dtype=jnp.float32).reshape(batch, n)
    b = jnp.ones((batch, n), dtype=jnp.float32) * 3.0

    def f(a, b):
        return _apply(transport, {"a": a, "b": b}, vmap_method="sequential")["c"]

    out = jax.jit(jax.vmap(f))(a, b)
    np.testing.assert_allclose(
        _to_np(out), np.asarray(a) * 2.0 + np.asarray(b), rtol=1e-6, atol=0
    )
