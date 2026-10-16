# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common subexpression elimination of Tesseract calls.

A Tesseract endpoint is a pure function of its inputs, so XLA is free to fold
repeated identical calls into one request -- as it already does for LAPACK
custom calls. Two things are needed: the callback must be lowered as pure, and
the bind params (including the Jaxeract client) must compare equal.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract, primitive
from tesseract_jax.tesseract_compat import Jaxeract


def _spy_endpoints(tess, monkeypatch):
    """Wrap apply / jacobian / jvp / vjp endpoints with counters."""
    counts = {"apply": 0, "jacobian": 0, "jvp": 0, "vjp": 0}
    orig_apply = tess.apply
    orig_jac = tess.jacobian
    orig_jvp = tess.jacobian_vector_product
    orig_vjp = tess.vector_jacobian_product

    def wa(*a, **kw):
        counts["apply"] += 1
        return orig_apply(*a, **kw)

    def wj(*a, **kw):
        counts["jacobian"] += 1
        return orig_jac(*a, **kw)

    def wjvp(*a, **kw):
        counts["jvp"] += 1
        return orig_jvp(*a, **kw)

    def wvjp(*a, **kw):
        counts["vjp"] += 1
        return orig_vjp(*a, **kw)

    monkeypatch.setattr(tess, "apply", wa)
    monkeypatch.setattr(tess, "jacobian", wj)
    monkeypatch.setattr(tess, "jacobian_vector_product", wjvp)
    monkeypatch.setattr(tess, "vector_jacobian_product", wvjp)
    return counts


def test_chunked_jacobian_calls_endpoint_once(vectoradd_tess, monkeypatch):
    """Chunking the identity matrix must not multiply the ``jacobian`` requests.

    Splitting the eye-vmap into chunks is a way to cap peak memory when the
    Tesseract is one component of a larger function. The Jacobian does not depend
    on the tangents, so every chunk issues an identical request and only one of
    them needs to reach the Tesseract.
    """
    n = 6
    a = jnp.arange(n, dtype="float32") + 1.0
    b = jnp.full((n,), 0.5, dtype="float32")

    def f(a):
        # tanh stands in for the other, memory-hungry components that motivate
        # chunking in the first place.
        return jnp.tanh(apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"])

    expected = jax.jacfwd(f)(a)
    _primal, tangent_fn = jax.linearize(f, a)
    batched = jax.vmap(tangent_fn)
    eye = jnp.eye(n, dtype="float32")

    # Spy only over the chunked computation, so the reference above is not counted.
    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jit(lambda e: jnp.concatenate([batched(c) for c in jnp.split(e, 3)]))(eye)

    np.testing.assert_allclose(M, expected, atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_identical_calls_are_commoned_up(vectoradd_tess, monkeypatch):
    """Two identical ``apply_tesseract`` calls in one trace issue one request.

    Both operands are passed as arguments rather than closed over: a concrete
    closed-over array becomes a static arg wrapped in ``_Hashable``, which
    compares by identity and so defeats CSE for unrelated reasons.
    """
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    @jax.jit
    def twice(a, b):
        c1 = apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]
        c2 = apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]
        return c1 + c2

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    out = twice(a, b)

    np.testing.assert_allclose(out, 2.0 * (a + b), atol=1e-6)
    assert counts["apply"] == 1


def test_distinct_calls_are_not_commoned_up(vectoradd_tess, monkeypatch):
    """Calls that differ in their inputs must stay separate requests."""
    a1 = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    a2 = jnp.array([100.0, 200.0, 300.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    @jax.jit
    def two_different(a1, a2, b):
        c1 = apply_tesseract(vectoradd_tess, dict(a=a1, b=b))["c"]
        c2 = apply_tesseract(vectoradd_tess, dict(a=a2, b=b))["c"]
        return c1, c2

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    c1, c2 = two_different(a1, a2, b)

    np.testing.assert_allclose(c1, a1 + b, atol=1e-6)
    np.testing.assert_allclose(c2, a2 + b, atol=1e-6)
    assert counts["apply"] == 2


def test_jaxeract_wrappers_compare_equal(vectoradd_tess):
    """Distinct wrappers around one Tesseract are equal, so bind params match."""
    assert Jaxeract(vectoradd_tess) == Jaxeract(vectoradd_tess)
    assert hash(Jaxeract(vectoradd_tess)) == hash(Jaxeract(vectoradd_tess))
    assert Jaxeract(vectoradd_tess) != object()


def test_jaxeract_views_are_distinct(served_cuda_ipc_vectoradd_tesseract):
    """A view with its own encoding lowers separately from the Tesseract it views.

    Each lowering resolves the GPU transport of its own Tesseract, so a view
    requesting ``"none"`` must not be commoned up with one that uses cuda_ipc.
    """
    tess = served_cuda_ipc_vectoradd_tesseract
    host = Jaxeract(tess.with_encoding(gpu_transport="none"))
    assert Jaxeract(tess) != host


def test_jaxeract_views_with_one_encoding_compare_equal(
    served_cuda_ipc_vectoradd_tesseract,
):
    """Two views requesting the same encoding wrap interchangeable Tesseracts.

    ``with_encoding`` returns a new object every time, so a caller that builds a
    view per call would otherwise lower (and on GPU, register) every call anew.
    """
    tess = served_cuda_ipc_vectoradd_tesseract
    view1 = Jaxeract(tess.with_encoding(gpu_transport="none"))
    view2 = Jaxeract(tess.with_encoding(gpu_transport="none"))
    assert view1 == view2
    assert hash(view1) == hash(view2)


def test_jaxeract_separate_tesseracts_are_distinct(vectoradd_tess):
    """Tesseracts loaded separately stay distinct, even from the same API."""
    other = Tesseract.from_tesseract_api("tests/vectoradd_tesseract/tesseract_api.py")
    assert Jaxeract(vectoradd_tess) != Jaxeract(other)


def test_fresh_view_per_call_lowers_once(vectoradd_tess, monkeypatch):
    """Eager calls through a fresh ``with_encoding`` view reuse one compilation."""
    lowerings = []
    build = primitive._build_dispatch_closure

    def counting_build(*args, **kwargs):
        lowerings.append(args)
        return build(*args, **kwargs)

    monkeypatch.setattr(primitive, "_build_dispatch_closure", counting_build)

    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")
    for _ in range(3):
        view = vectoradd_tess.with_encoding(gpu_transport="none")
        out = apply_tesseract(view, dict(a=a, b=b))["c"]
        np.testing.assert_allclose(out, a + b, atol=1e-6)

    assert len(lowerings) == 1
