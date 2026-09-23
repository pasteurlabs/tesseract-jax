# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for jax.linearize and jax.linear_transpose with tesseract-jax.

Tests use the (nonlinear) Rosenbrock tesseract unless they need several
differentiable inputs: a linear tesseract makes the Jacobian the identity, which
renders most assertions here tautological.

Every test is parametrised over ``use_jit`` to check that an eager call and one
staged out under ``jit`` agree.
"""

import jax
import numpy as np
import pytest

from tesseract_jax import apply_tesseract

# ---------------------------------------------------------------------------
# jax.linearize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_jit", [True, False])
def test_linearize_univariate(univariate_tess, use_jit):
    """jax.linearize on the Rosenbrock tesseract, check tangent_fn matches JVP."""
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    _primal_out, tangent_fn = jax.linearize(f, x)

    # Compare tangent_fn with jax.jvp
    t = np.array(1.0, dtype="float64")
    tangent_out = tangent_fn(t)

    _, jvp_out = jax.jvp(f, (x,), (t,))
    np.testing.assert_allclose(tangent_out, jvp_out, rtol=1e-5)

    # Linearity check
    t2 = np.array(3.0, dtype="float64")
    np.testing.assert_allclose(
        tangent_fn(t + t2), tangent_fn(t) + tangent_fn(t2), rtol=1e-5
    )


# ---------------------------------------------------------------------------
# jax.linear_transpose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_jit", [True, False])
def test_linear_transpose_univariate(univariate_tess, use_jit):
    """linear_transpose of the linearized tangent function matches VJP."""
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    _primal_out, tangent_fn = jax.linearize(f, x)

    transpose_fn = jax.linear_transpose(tangent_fn, x)

    ct = np.array(1.0, dtype="float64")
    (transposed,) = transpose_fn(ct)

    # Compare with VJP
    _, vjp_fn = jax.vjp(f, x)
    (vjp_result,) = vjp_fn(ct)
    np.testing.assert_allclose(transposed, vjp_result, rtol=1e-5)


# ---------------------------------------------------------------------------
# UndefinedPrimal guard in the transpose rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_jit", [True, False])
def test_linear_transpose_on_raw_jvp_without_closure(univariate_tess, use_jit):
    """linear_transpose on jvp_fun(primals, tangents) is rejected.

    When primals are NOT closed over, JAX leaves them as UndefinedPrimal
    residuals when the transpose rule fires — transposing with respect to a
    primal is not linear, so there is nothing sensible to dispatch. Tracing
    succeeds; calling the transposed function raises a guidance ValueError.
    """
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    def jvp_fun(primals, tangents):
        return jax.jvp(f, (primals,), (tangents,))[1]

    transpose_fn = jax.linear_transpose(jvp_fun, x, x)
    ct = np.array(1.0, dtype="float64")
    with pytest.raises(ValueError, match=r"(?i)UndefinedPrimal"):
        transpose_fn(ct)


@pytest.mark.parametrize("use_jit", [True, False])
def test_linear_transpose_on_raw_jvp_with_closure(univariate_tess, use_jit):
    """The workaround the guard's error message recommends must actually work.

    Closing over the primals leaves the function linear in its tangent argument
    alone, so the transpose is well defined and agrees with the VJP.
    """
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    primals = (x,)
    transpose_fn = jax.linear_transpose(lambda t: jax.jvp(f, primals, (t,))[1], x)

    ct = np.array(1.0, dtype="float64")
    (transposed,) = transpose_fn(ct)

    _, vjp_fn = jax.vjp(f, x)
    (vjp_result,) = vjp_fn(ct)
    np.testing.assert_allclose(transposed, vjp_result, rtol=1e-5)


# ---------------------------------------------------------------------------
# Differentiating a derivative endpoint
#
# J(x)·v is linear in v, so jax.jvp / jax.jacfwd of a jax.linearize tangent
# function is well defined and reduces to the same endpoint at a new tangent.
# Differentiating with respect to x is not, and must stay refused.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_jit", [True, False])
def test_jvp_of_tangent_fn(univariate_tess, use_jit):
    """jax.jvp of a linearized function: both outputs are the endpoint re-applied."""
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")
    t = np.array(3.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    _primal_out, tangent_fn = jax.linearize(f, x)
    primal_out, tangent_out = jax.jvp(tangent_fn, (x,), (t,))

    # tangent_fn is linear, so its JVP is itself evaluated at the new tangent.
    np.testing.assert_allclose(primal_out, tangent_fn(x), rtol=1e-5)
    np.testing.assert_allclose(tangent_out, tangent_fn(t), rtol=1e-5)


@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("use_jit", [True, False])
def test_jacobian_of_tangent_fn(univariate_tess, use_jit, mode):
    """jacfwd/jacrev of a linearized function recovers f's Jacobian.

    This is the pattern optimistix uses.
    """
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    jac = jax.jacfwd if mode == "fwd" else jax.jacrev
    _primal_out, tangent_fn = jax.linearize(f, x)

    # d/dv [J(x)·v] == J(x)
    np.testing.assert_allclose(jac(tangent_fn)(x), jac(f)(x), rtol=1e-5)


@pytest.mark.parametrize("use_jit", [True, False])
def test_jvp_of_vjp_fn(univariate_tess, use_jit):
    """The VJP endpoint is likewise linear, in its cotangent slots."""
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")
    ct = np.array(1.0, dtype="float64")
    dct = np.array(3.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    _primal_out, vjp_fn = jax.vjp(f, x)
    primal_out, tangent_out = jax.jvp(vjp_fn, (ct,), (dct,))

    np.testing.assert_allclose(primal_out[0], vjp_fn(ct)[0], rtol=1e-5)
    np.testing.assert_allclose(tangent_out[0], vjp_fn(dct)[0], rtol=1e-5)


@pytest.mark.parametrize("use_jit", [True, False])
def test_transpose_of_vjp_fn(univariate_tess, use_jit):
    """The VJP endpoint's transpose wrt its cotangent slot is the JVP endpoint.

    Regression test for #266: reverse mode through vjp_fn's cotangent (grad,
    linear_transpose) used to hit a bare AssertionError, even though forward
    mode (test_jvp_of_vjp_fn) already worked.
    """
    x = np.array(1.5, dtype="float64")
    y = np.array(0.5, dtype="float64")
    ct = np.array(1.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    _primal_out, vjp_fn = jax.vjp(f, x)
    g = lambda ct: vjp_fn(ct)[0]

    _, expected = jax.jvp(f, (x,), (1.0,))

    grad_out = jax.grad(g)(ct)
    np.testing.assert_allclose(grad_out, expected, rtol=1e-5)

    (transposed,) = jax.linear_transpose(g, ct)(ct)
    np.testing.assert_allclose(transposed, expected, rtol=1e-5)


@pytest.mark.parametrize("use_jit", [True, False])
def test_transpose_of_vjp_fn_with_undifferentiated_array_primals(
    univariate_tess, use_jit
):
    """Canary for a VJP-arity assumption in the transpose rule's ``else`` branch.

    ``a``/``b`` are passed as real arrays (not left at their schema defaults),
    so ``n_primals`` (4: x, y, a, b) is strictly larger than the number of
    differentiated primals (1: x). Today the VJP bind's own output is
    ``n_primals``-long regardless of ``has_tangent``, which is what
    ``tesseract_dispatch_transpose_rule``'s ``else`` branch assumes when it
    zips the incoming cotangent against ``_flat_inputs`` (strict=True).

    PR #263 shrinks that output to just the differentiated primals. If it
    lands without updating the transpose rule's ``else`` branch to match,
    this test should start failing loudly (a ``strict=True`` zip
    length-mismatch, not a silently wrong gradient) -- that failure is the
    signal that the `else` branch needs a coordinated update for the new
    VJP arity contract, the same way the rest of the file was updated in
    that PR.
    """
    x = np.array(1.5, dtype="float64")
    y, a, b = (np.array(v, dtype="float64") for v in (0.5, 1.0, 100.0))

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y, a=a, b=b))[
            "result"
        ]

    if use_jit:
        f = jax.jit(f)

    _primal_out, vjp_fn = jax.vjp(f, x)
    g = lambda ct: vjp_fn(ct)[0]
    ct = np.array(1.0, dtype="float64")

    _, expected = jax.jvp(f, (x,), (1.0,))

    grad_out = jax.grad(g)(ct)
    np.testing.assert_allclose(grad_out, expected, rtol=1e-5)


@pytest.mark.parametrize("argnums", [0, 1])
def test_jacfwd_of_tangent_fn_partial_argnums(pytree_tess, pytree_tess_inputs, argnums):
    """Differentiating only some arguments leaves the rest symbolically zero.

    Uses a multi-differentiable-input tesseract so that some tangent slots arrive
    as ad.Zero and have to be instantiated before they can cross a bind.
    """
    inp = pytree_tess_inputs

    def f(alpha, delta):
        return apply_tesseract(
            pytree_tess, inputs={**inp, "alpha": alpha, "delta": delta}
        )["result"]

    alpha, delta = inp["alpha"], inp["delta"]
    _primal_out, tangent_fn = jax.linearize(f, alpha, delta)

    expected = jax.jacfwd(f, argnums=argnums)(alpha, delta)
    got = jax.jacfwd(tangent_fn, argnums=argnums)(alpha, delta)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5), expected, got
    )


@pytest.mark.parametrize("use_jit", [True, False])
def test_second_derivative_wrt_primals_is_refused(univariate_tess, use_jit):
    """A Hessian needs a second derivative, which Tesseract does not expose."""
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    with pytest.raises(RuntimeError, match=r"(?i)primal inputs"):
        jax.jacfwd(jax.grad(f))(x)
