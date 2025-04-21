# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax.typing import ArrayLike
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract


def _assert_pytree_isequal(a, b, rtol=None, atol=None):
    """Check if two PyTrees are equal."""
    a_flat, a_structure = jax.tree.flatten_with_path(a)
    b_flat, b_structure = jax.tree.flatten_with_path(b)

    if a_structure != b_structure:
        raise AssertionError(
            f"PyTree structures are different:\n{a_structure}\n{b_structure}"
        )

    if rtol is not None or atol is not None:
        array_compare = lambda x, y: np.testing.assert_allclose(
            x, y, rtol=rtol, atol=atol
        )
    else:
        array_compare = lambda x, y: np.testing.assert_array_equal(x, y)

    failures = []
    for (a_path, a_elem), (b_path, b_elem) in zip(a_flat, b_flat, strict=True):
        assert a_path == b_path, f"Unexpected path mismatch: {a_path} != {b_path}"
        try:
            if isinstance(a_elem, ArrayLike) or isinstance(b_elem, ArrayLike):
                array_compare(a_elem, b_elem)
            else:
                assert a_elem == b_elem, f"Values are different: {a_elem} != {b_elem}"
        except AssertionError as e:
            failures.append(a_path, str(e))

    if failures:
        msg = "\n".join(f"Path: {path}, Error: {error}" for path, error in failures)
        raise AssertionError(f"PyTree elements are different:\n{msg}")


def rosenbrock_impl(x, y, a=1.0, b=100.0):
    """JAX-traceable version of the Rosenbrock function used by univariate_tesseract."""
    return (a - x) ** 2 + b * (y - x**2) ** 2


@pytest.mark.parametrize("use_jit", [True, False])
def test_univariate_tesseract_apply(served_univariate_tesseract_raw, use_jit):
    rosenbrock_tess = Tesseract(served_univariate_tesseract_raw)
    x, y = np.array(0.0), np.array(0.0)

    def f(x, y):
        return apply_tesseract(rosenbrock_tess, inputs=dict(x=x, y=y))

    rosenbrock_raw = rosenbrock_impl
    if use_jit:
        f = jax.jit(f)
        rosenbrock_raw = jax.jit(rosenbrock_raw)

    # Test against Tesseract client
    result = f(x, y)
    result_ref = rosenbrock_tess.apply(dict(x=x, y=y))
    _assert_pytree_isequal(result, result_ref)

    # Test against direct implementation
    result_raw = rosenbrock_raw(x, y)
    np.testing.assert_array_equal(result["result"], result_raw)


@pytest.mark.parametrize("use_jit", [True, False])
def test_univariate_tesseract_jvp(served_univariate_tesseract_raw, use_jit):
    rosenbrock_tess = Tesseract(served_univariate_tesseract_raw)

    # make things callable without keyword args
    def f(x, y):
        return apply_tesseract(rosenbrock_tess, inputs=dict(x=x, y=y))

    rosenbrock_raw = rosenbrock_impl
    if use_jit:
        f = jax.jit(f)
        rosenbrock_raw = jax.jit(rosenbrock_raw)

    x, y = np.array(0.0), np.array(0.0)
    dx, dy = np.array(1.0), np.array(0.0)
    (primal, jvp) = jax.jvp(f, (x, y), (dx, dy))

    # Test against Tesseract client
    primal_ref = rosenbrock_tess.apply(dict(x=x, y=y))
    _assert_pytree_isequal(primal, primal_ref)

    jvp_ref = rosenbrock_tess.jacobian_vector_product(
        inputs=dict(x=x, y=y),
        jvp_inputs=["x", "y"],
        jvp_outputs=["result"],
        tangent_vector=dict(x=dx, y=dy),
    )
    _assert_pytree_isequal(jvp, jvp_ref)

    # Test against direct implementation
    _, jvp_raw = jax.jvp(rosenbrock_raw, (x, y), (dx, dy))
    np.testing.assert_array_equal(jvp["result"], jvp_raw)


@pytest.mark.parametrize("use_jit", [True, False])
def test_univariate_tesseract_vjp(served_univariate_tesseract_raw, use_jit):
    rosenbrock_tess = Tesseract(served_univariate_tesseract_raw)

    def f(x, y):
        return apply_tesseract(rosenbrock_tess, inputs=dict(x=x, y=y))

    rosenbrock_raw = rosenbrock_impl
    if use_jit:
        f = jax.jit(f)
        rosenbrock_raw = jax.jit(rosenbrock_raw)

    x, y = np.array(0.0), np.array(0.0)
    (primal, f_vjp) = jax.vjp(f, x, y)

    if use_jit:
        f_vjp = jax.jit(f_vjp)

    vjp = f_vjp(primal)

    # Test against Tesseract client
    primal_ref = rosenbrock_tess.apply(dict(x=x, y=y))
    _assert_pytree_isequal(primal, primal_ref)

    vjp_ref = rosenbrock_tess.vector_jacobian_product(
        inputs=dict(x=x, y=y),
        vjp_inputs=["x", "y"],
        vjp_outputs=["result"],
        cotangent_vector=primal_ref,
    )
    # JAX vjp returns a flat tuple, so unflatten it to match the Tesseract output (dict with keys vjp_inputs)
    vjp = {"x": vjp[0], "y": vjp[1]}
    _assert_pytree_isequal(vjp, vjp_ref)

    # Test against direct implementation
    primal_raw, f_vjp_raw = jax.vjp(rosenbrock_raw, x, y)
    if use_jit:
        f_vjp_raw = jax.jit(f_vjp_raw)
    vjp_raw = f_vjp_raw(primal_raw)
    vjp_raw = {"x": vjp_raw[0], "y": vjp_raw[1]}
    _assert_pytree_isequal(vjp, vjp_raw)


@pytest.mark.parametrize("use_jit", [True, False])
def test_nested_tesseract_apply(served_nested_tesseract_raw, use_jit):
    nested_tess = Tesseract(served_nested_tesseract_raw)
    a, b = np.array(1.0, dtype="float32"), np.array(2.0, dtype="float32")
    v, w = (
        np.array([1.0, 2.0, 3.0], dtype="float32"),
        np.array([5.0, 7.0, 9.0], dtype="float32"),
    )

    def f(a, v, s, i):
        return apply_tesseract(
            nested_tess,
            inputs={
                "scalars": {"a": a, "b": b},
                "vectors": {"v": v, "w": w},
                "other_stuff": {"s": s, "i": i, "f": 2.718},
            },
        )

    if use_jit:
        f = jax.jit(f, static_argnames=["s", "i"])

    result = f(a, v, "hello", 3)
    result_ref = nested_tess.apply(
        inputs={
            "scalars": {"a": a, "b": b},
            "vectors": {"v": v, "w": w},
            "other_stuff": {"s": "hello", "i": 3, "f": 2.718},
        }
    )
    _assert_pytree_isequal(result, result_ref)


@pytest.mark.parametrize("use_jit", [True, False])
def test_nested_tesseract_jvp(served_nested_tesseract_raw, use_jit):
    nested_tess = Tesseract(served_nested_tesseract_raw)
    a, b = np.array(1.0, dtype="float32"), np.array(2.0, dtype="float32")
    v, w = (
        np.array([1.0, 2.0, 3.0], dtype="float32"),
        np.array([5.0, 7.0, 9.0], dtype="float32"),
    )

    def f(a, v):
        return apply_tesseract(
            nested_tess,
            inputs=dict(
                scalars={"a": a, "b": b},
                vectors={"v": v, "w": w},
                other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
            ),
        )

    if use_jit:
        f = jax.jit(f)

    (primal, jvp) = jax.jvp(f, (a, v), (a, v))

    primal_ref = nested_tess.apply(
        inputs=dict(
            scalars={"a": a, "b": b},
            vectors={"v": v, "w": w},
            other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
        )
    )
    _assert_pytree_isequal(primal, primal_ref)

    jvp_ref = nested_tess.jacobian_vector_product(
        inputs=dict(
            scalars={"a": a, "b": b},
            vectors={"v": v, "w": w},
            other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
        ),
        jvp_inputs=["scalars.a", "vectors.v"],
        jvp_outputs=["scalars.a", "vectors.v"],
        tangent_vector={"scalars.a": a, "vectors.v": v},
    )
    # JAX returns a nested dict, so we need to flatten it to match the Tesseract output (dict with keys jvp_outputs)
    jvp = {"scalars.a": jvp["scalars"]["a"], "vectors.v": jvp["vectors"]["v"]}
    _assert_pytree_isequal(jvp, jvp_ref)


@pytest.mark.parametrize("use_jit", [True, False])
def test_nested_tesseract_vjp(served_nested_tesseract_raw, use_jit):
    nested_tess = Tesseract(served_nested_tesseract_raw)

    a, b = np.array(1.0, dtype="float32"), np.array(2.0, dtype="float32")
    v, w = (
        np.array([1.0, 2.0, 3.0], dtype="float32"),
        np.array([5.0, 7.0, 9.0], dtype="float32"),
    )

    def f(a, v):
        return apply_tesseract(
            nested_tess,
            inputs=dict(
                scalars={"a": a, "b": b},
                vectors={"v": v, "w": w},
                other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
            ),
        )

    if use_jit:
        f = jax.jit(f)

    (primal, f_vjp) = jax.vjp(f, a, v)

    if use_jit:
        f_vjp = jax.jit(f_vjp)

    vjp = f_vjp(primal)

    primal_ref = nested_tess.apply(
        inputs=dict(
            scalars={"a": a, "b": b},
            vectors={"v": v, "w": w},
            other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
        )
    )
    _assert_pytree_isequal(primal, primal_ref)

    vjp_ref = nested_tess.vector_jacobian_product(
        inputs=dict(
            scalars={"a": a, "b": b},
            vectors={"v": v, "w": w},
            other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
        ),
        vjp_inputs=["scalars.a", "vectors.v"],
        vjp_outputs=["scalars.a", "vectors.v"],
        cotangent_vector={
            "scalars.a": primal_ref["scalars"]["a"],
            "vectors.v": primal_ref["vectors"]["v"],
        },
    )
    # JAX vjp returns a flat tuple, so unflatten it to match the Tesseract output (dict with keys vjp_inputs)
    vjp = {"scalars.a": vjp[0], "vectors.v": vjp[1]}
    _assert_pytree_isequal(vjp, vjp_ref)


@pytest.mark.parametrize("use_jit", [True, False])
def test_partial_differentiation(served_univariate_tesseract_raw, use_jit):
    """Test that differentiation works correctly in cases where some inputs are constants."""
    rosenbrock_tess = Tesseract(served_univariate_tesseract_raw)
    x, y = np.array(0.0), np.array(0.0)

    def f(y):
        return apply_tesseract(rosenbrock_tess, inputs=dict(x=x, y=y))["result"]

    if use_jit:
        f = jax.jit(f)

    # Test forward application
    result = f(y)
    result_ref = rosenbrock_tess.apply(dict(x=x, y=y))["result"]
    _assert_pytree_isequal(result, result_ref)

    # Test gradient
    grad = jax.grad(f)(y)
    grad_ref = rosenbrock_tess.vector_jacobian_product(
        inputs=dict(x=x, y=y),
        vjp_inputs=["y"],
        vjp_outputs=["result"],
        cotangent_vector=dict(result=1.0),
    )["y"]
    _assert_pytree_isequal(grad, grad_ref)


def test_tesseract_as_jax_pytree(served_univariate_tesseract_raw):
    """Test that Tesseract can be used as a JAX PyTree."""
    tess = Tesseract(served_univariate_tesseract_raw)

    @jax.jit
    def f(x, y, tess):
        return apply_tesseract(tess, inputs=dict(x=x, y=y))["result"]

    x, y = np.array(0.0), np.array(0.0)
    result = f(x, y, tess)
    result_ref = rosenbrock_impl(x, y)
    _assert_pytree_isequal(result, result_ref)
