# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spike tests for the JVP/VJP-batching shortcut that calls the ``jacobian`` endpoint.

When a JVP / VJP eqn is vmapped with primals unbatched and (co)tangents batched
— the pattern produced by ``jax.jacfwd`` / ``jax.jacrev``,
``lineax.materialize(FunctionLinearOperator)``, and optimistix's eye-vmap —
the batching rule materializes the Jacobian once via the ``jacobian`` endpoint
and contracts it against the batched (co)tangents instead of calling the
``jvp`` / ``vjp`` endpoint per batch element.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract
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


def test_jacfwd_uses_jacobian_endpoint(vectoradd_tess, monkeypatch):
    """``jax.jacfwd`` hits the ``jacobian`` endpoint exactly once, never ``jvp``."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jacfwd(f)(a)

    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_jacrev_uses_jacobian_endpoint(vectoradd_tess, monkeypatch):
    """``jax.jacrev`` hits the ``jacobian`` endpoint exactly once, never ``vjp``."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jacrev(f)(a)

    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["vjp"] == 0


def test_jacfwd_chained_tesseracts(vectoradd_tess, monkeypatch):
    """``jacfwd`` on ``tess(tess(a))`` uses 2 jacobian calls (one per Tesseract)."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b1 = jnp.array([0.5, 0.5, 0.5], dtype="float32")
    b2 = jnp.array([0.1, 0.2, 0.3], dtype="float32")

    def f(a):
        c1 = apply_tesseract(vectoradd_tess, dict(a=a, b=b1))["c"]
        c2 = apply_tesseract(vectoradd_tess, dict(a=c1, b=b2))["c"]
        return c2

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jacfwd(f)(a)

    # df/da = I @ I = I.
    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 2
    assert counts["jvp"] == 0


def test_jacfwd_through_jit(vectoradd_tess, monkeypatch):
    """``jacfwd`` of a ``jax.jit``-wrapped Tesseract call still uses the shortcut."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    @jax.jit
    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jacfwd(f)(a)

    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_jacfwd_of_linearized_uses_jacobian_endpoint(vectoradd_tess, monkeypatch):
    """``jacfwd`` of a ``jax.linearize`` tangent function uses the shortcut too.

    The spy covers the ``linearize`` call as well, so the counts are the whole cost
    of the pattern: one ``apply`` (the forward evaluation ``linearize`` itself
    needs, and differentiating the tangent function adds none) plus one
    ``jacobian`` (the batching shortcut), and no ``jvp`` at all.

    That last one is the interesting part. The JVP rule for a derivative endpoint
    emits two binds -- the undifferentiated output and the endpoint re-applied at
    the new tangent -- and ``jacfwd`` discards the former. Under ``jit`` DCE drops
    it, so this costs exactly what ``jacfwd(f)`` costs. Un-jitted, DCE does not
    fire and the discarded bind is paid for; hence the ``jit`` here.
    """
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    _primal_out, tangent_fn = jax.linearize(f, a)
    M = jax.jit(jax.jacfwd(tangent_fn))(a)

    # tangent_fn is linear with f's Jacobian, so this is df/da = I.
    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["apply"] == 1
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_jacfwd_residual_pattern(vectoradd_tess, monkeypatch):
    """``f(a) = a + tess(a)`` — residual pattern. df/da = 2*I."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return a + apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jacfwd(f)(a)

    np.testing.assert_allclose(M, 2.0 * np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_explicit_vmap_of_jvp_uses_jacobian(vectoradd_tess, monkeypatch):
    """``vmap(jvp_fn)(eye)`` (the optimistix / lineax materialize pattern) is intercepted."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    _primal, jvp_fn = jax.linearize(f, a)
    eye = jnp.eye(3, dtype="float32")

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.vmap(jvp_fn)(eye)

    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_explicit_vmap_of_vjp_uses_jacobian(vectoradd_tess, monkeypatch):
    """``vmap(vjp_fn)(eye)`` is intercepted too."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    _primal, vjp_fn = jax.vjp(f, a)
    eye = jnp.eye(3, dtype="float32")

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    (M,) = jax.vmap(vjp_fn)(eye)

    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 1
    assert counts["vjp"] == 0


def test_jvp_single_tangent_still_uses_jvp_endpoint(vectoradd_tess, monkeypatch):
    """A plain ``jax.jvp`` (no vmap) still calls the ``jvp`` endpoint."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")
    tangent = jnp.array([1.0, 0.0, 0.0], dtype="float32")

    def f(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    _, jvp_out = jax.jvp(f, (a,), (tangent,))

    np.testing.assert_allclose(jvp_out, tangent, atol=1e-6)
    assert counts["jacobian"] == 0
    assert counts["jvp"] == 1


def test_materialize_jacobian_false_forces_sequential(vectoradd_tess, monkeypatch):
    """``materialize_jacobian=False`` skips the shortcut even when the endpoint exists."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f(a):
        return apply_tesseract(
            vectoradd_tess,
            dict(a=a, b=b),
            vmap_method="sequential",
            materialize_jacobian=False,
        )["c"]

    counts = _spy_endpoints(vectoradd_tess, monkeypatch)
    M = jax.jacfwd(f)(a)

    np.testing.assert_allclose(M, np.eye(3, dtype="float32"), atol=1e-6)
    assert counts["jacobian"] == 0
    assert counts["jvp"] == 3  # N=3 eye-vmap columns


def test_materialize_jacobian_true_errors_without_endpoint(vectoradd_tess, monkeypatch):
    """``materialize_jacobian=True`` raises if the Tesseract has no ``jacobian`` endpoint."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    # Hide the ``jacobian`` endpoint from the Jaxeract wrapper.
    from tesseract_jax.tesseract_compat import Jaxeract

    orig_init = Jaxeract.__init__

    def patched_init(self, tess):
        orig_init(self, tess)
        self.available_methods = [m for m in self.available_methods if m != "jacobian"]

    monkeypatch.setattr(Jaxeract, "__init__", patched_init)

    def f(a):
        return apply_tesseract(
            vectoradd_tess, dict(a=a, b=b), materialize_jacobian=True
        )["c"]

    with pytest.raises(RuntimeError, match="materialize_jacobian=True"):
        jax.jacfwd(f)(a)


def test_scalar_jacobian_fallback(univariate_tess, monkeypatch):
    """Scalar (1x1) case still routes through the jacobian endpoint via vmap."""
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    counts = _spy_endpoints(univariate_tess, monkeypatch)
    g = jax.jacfwd(f)(x)

    # df/dx = -2(1-x) - 4*100*x*(y-x^2). At x=1, y=2: -2(0) - 400*1*1 = -400.
    np.testing.assert_allclose(g, -400.0, rtol=1e-5)
    assert counts["jacobian"] == 1
    assert counts["jvp"] == 0


def test_batched_vjp_dtype_compatible_with_scan(mixed_dtype_tess):
    """``vmap(vjp_fn)`` must return input dtype; otherwise ``lax.scan`` errors.

    JAX's VJP returns gradients in the *input* dtype. If the shortcut produces a
    wider dtype, downstream consumers that strictly type-check — like
    ``jax.lax.scan``'s carry — raise ``TypeError`` rather than silently casting.
    This is a natural pattern (e.g. an iterative algorithm that uses the
    Jacobian inside its loop body).
    """
    x = jnp.array([1.0, 2.0, 3.0], dtype="float32")

    def f(x):
        return apply_tesseract(mixed_dtype_tess, dict(x=x))["y"]

    def body(carry, _):
        _, vjp_fn = jax.vjp(f, carry)
        (g,) = jax.vmap(vjp_fn)(jnp.eye(3, dtype="float64"))
        return g.sum(axis=0), None

    final, _ = jax.lax.scan(body, x, None, length=2)
    assert final.dtype == x.dtype


def test_batched_jvp_dtype_matches_jax_convention(mixed_dtype_tess):
    """``vmap(jvp_fn)`` must return output-dtype results (JAX convention)."""
    x = jnp.array([1.0, 2.0, 3.0], dtype="float32")

    def f_tess(x):
        return apply_tesseract(mixed_dtype_tess, dict(x=x))["y"]

    def f_jax(x):
        return x.astype(jnp.float64) * 2.0

    _, jvp_jax = jax.linearize(f_jax, x)
    _, jvp_tess = jax.linearize(f_tess, x)
    eye_x = jnp.eye(3, dtype="float32")
    out_jax = jax.vmap(jvp_jax)(eye_x)
    out_tess = jax.vmap(jvp_tess)(eye_x)
    assert out_tess.dtype == out_jax.dtype, (
        f"vmap(jvp_fn): tess {out_tess.dtype} vs jax {out_jax.dtype}"
    )


def test_jacfwd_partial_diff_restricts_jac_inputs(univariate_tess, monkeypatch):
    """``jacfwd`` wrt one of several diff inputs requests only that column."""
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x):
        # `y` is also schema-differentiable but JAX won't carry a tangent for it.
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    captured: dict[str, Any] = {}
    orig = univariate_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        captured["jac_inputs"] = list(jac_inputs)
        captured["jac_outputs"] = list(jac_outputs)
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(univariate_tess, "jacobian", spy)
    g = jax.jacfwd(f)(x)

    np.testing.assert_allclose(g, -400.0, rtol=1e-5)
    assert captured["jac_inputs"] == ["x"], (
        f"expected only 'x' to be requested, got {captured['jac_inputs']}"
    )


def test_jacrev_partial_diff_restricts_jac_inputs(univariate_tess, monkeypatch):
    """``jacrev`` wrt one of several diff inputs requests only that column."""
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    captured: dict[str, Any] = {}
    orig = univariate_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        captured["jac_inputs"] = list(jac_inputs)
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(univariate_tess, "jacobian", spy)
    g = jax.jacrev(f)(x)

    np.testing.assert_allclose(g, -400.0, rtol=1e-5)
    assert captured["jac_inputs"] == ["x"]


def test_jacfwd_of_tangent_fn_restricts_jac_inputs(univariate_tess, monkeypatch):
    """``jacfwd`` of a linearized function wrt one argument narrows the request too.

    The tangent function's other argument gets a symbolic-zero tangent, which is
    instantiated to dense zeros before it can cross a bind. Its Jacobian column
    would then be fetched only to be multiplied by those zeros, so the JVP rule
    recomputes ``has_tangent`` for the tangent bind rather than inheriting it.
    """
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x, y):
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    _primal, tangent_fn = jax.linearize(f, x, y)
    expected = jax.jacfwd(f, argnums=0)(x, y)  # before the spy, so it is not captured

    captured: dict[str, Any] = {}
    orig = univariate_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        captured["jac_inputs"] = list(jac_inputs)
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(univariate_tess, "jacobian", spy)
    g = jax.jacfwd(tangent_fn, argnums=0)(x, y)

    np.testing.assert_allclose(g, expected, rtol=1e-5)
    assert captured["jac_inputs"] == ["x"], (
        f"expected only 'x' to be requested, got {captured['jac_inputs']}"
    )


@pytest.mark.parametrize("use_jit", [True, False])
def test_matches_jacrev_on_pure_jax(vectoradd_tess, use_jit, monkeypatch):
    """The shortcut's numerical result matches a pure-JAX implementation."""
    a = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    b = jnp.array([0.5, 0.5, 0.5], dtype="float32")

    def f_tess(a):
        return apply_tesseract(vectoradd_tess, dict(a=a, b=b))["c"]

    def f_jax(a):
        return a + b  # the Tesseract's apply rule, in JAX

    if use_jit:
        f_tess = jax.jit(f_tess)
        f_jax = jax.jit(f_jax)

    M_tess = jax.jacfwd(f_tess)(a)
    M_jax = jax.jacfwd(f_jax)(a)
    np.testing.assert_allclose(M_tess, M_jax, rtol=1e-5)


# ---------------------------------------------------------------------------
# Common subexpression elimination of Tesseract calls
#
# A Tesseract endpoint is a pure function of its inputs, so XLA is free to fold
# repeated identical calls into one request -- as it already does for LAPACK
# custom calls. Two things are needed: the callback must be lowered as pure, and
# the bind params (including the Jaxeract client) must compare equal.
# ---------------------------------------------------------------------------


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
