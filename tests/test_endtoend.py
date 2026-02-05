# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import jax
import jax.numpy as jnp
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
            failures.append((a_path, str(e)))

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
@pytest.mark.parametrize("jac_direction", ["fwd", "rev"])
def test_univariate_tesseract_jacobian(
    served_univariate_tesseract_raw, use_jit, jac_direction
):
    rosenbrock_tess = Tesseract(served_univariate_tesseract_raw)

    # make things callable without keyword args
    def f(x, y):
        return apply_tesseract(rosenbrock_tess, inputs=dict(x=x, y=y))["result"]

    if jac_direction == "fwd":
        f = jax.jacfwd(f, argnums=(0, 1))
        rosenbrock_raw = jax.jacfwd(rosenbrock_impl, argnums=(0, 1))
    else:
        f = jax.jacrev(f, argnums=(0, 1))
        rosenbrock_raw = jax.jacrev(rosenbrock_impl, argnums=(0, 1))

    if use_jit:
        f = jax.jit(f)
        rosenbrock_raw = jax.jit(rosenbrock_raw)

    x, y = np.array(0.0), np.array(0.0)
    jac = f(x, y)

    # Test against Tesseract client
    jac_ref = rosenbrock_tess.jacobian(
        inputs=dict(x=x, y=y), jac_inputs=["x", "y"], jac_outputs=["result"]
    )

    # Convert from nested dict to nested tuple
    jac_ref = tuple((jac_ref["result"]["x"], jac_ref["result"]["y"]))
    _assert_pytree_isequal(jac, jac_ref)

    # Test against direct implementation
    jac_raw = rosenbrock_raw(x, y)
    _assert_pytree_isequal(jac, jac_raw)


@pytest.mark.parametrize("use_jit", [True, False])
def test_univariate_tesseract_vmap(served_univariate_tesseract_raw, use_jit):
    rosenbrock_tess = Tesseract(served_univariate_tesseract_raw)

    # make things callable without keyword args
    def f(x, y):
        return apply_tesseract(rosenbrock_tess, inputs=dict(x=x, y=y))["result"]

    # add one batch dimension
    for axes in [(0, 0), (0, None), (None, 0)]:
        x = np.arange(3) if axes[0] is not None else np.array(0.0)
        y = np.arange(3) if axes[1] is not None else np.array(0.0)
        f_vmapped = jax.vmap(f, in_axes=axes)
        raw_vmapped = jax.vmap(rosenbrock_impl, in_axes=axes)

        if use_jit:
            f_vmapped = jax.jit(f_vmapped)
            raw_vmapped = jax.jit(raw_vmapped)

        result = f_vmapped(x, y)
        result_raw = raw_vmapped(x, y)

        _assert_pytree_isequal(result, result_raw)

        # add an additional batch dimension
        for extra_dim in [0, 1, -1]:
            if axes[0] is not None:
                x = np.arange(6).reshape(2, 3)
            if axes[1] is not None:
                y = np.arange(6).reshape(2, 3)

            additional_axes = tuple(
                extra_dim if ax is not None else None for ax in axes
            )

            for out_axis in [0, 1, -1]:
                f_vmappedtwice = jax.vmap(
                    f_vmapped, in_axes=additional_axes, out_axes=out_axis
                )
                raw_vmappedtwice = jax.vmap(
                    raw_vmapped, in_axes=additional_axes, out_axes=out_axis
                )

                if use_jit:
                    f_vmappedtwice = jax.jit(f_vmappedtwice)
                    raw_vmappedtwice = jax.jit(raw_vmappedtwice)

                result = f_vmappedtwice(x, y)
                result_raw = raw_vmappedtwice(x, y)

                _assert_pytree_isequal(result, result_raw)


@pytest.mark.parametrize("use_jit", [True, False])
def test_nested_tesseract_apply(served_nested_tesseract_raw, use_jit):
    nested_tess = Tesseract.from_tesseract_api(
        "tests/nested_tesseract/tesseract_api.py"
    )
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
@pytest.mark.parametrize("jac_direction", ["fwd", "rev"])
def test_nested_tesseract_jacobian(served_nested_tesseract_raw, use_jit, jac_direction):
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

    if jac_direction == "fwd":
        f = jax.jacfwd(f, argnums=(0, 1))
    else:
        f = jax.jacrev(f, argnums=(0, 1))

    if use_jit:
        f = jax.jit(f)

    jac = f(a, v)

    jac_ref = nested_tess.jacobian(
        inputs=dict(
            scalars={"a": a, "b": b},
            vectors={"v": v, "w": w},
            other_stuff={"s": "hey!", "i": 1234, "f": 2.718},
        ),
        jac_inputs=["scalars.a", "vectors.v"],
        jac_outputs=["scalars.a", "vectors.v"],
    )
    # JAX returns a 2-layered nested dict containing tuples of arrays
    # we need to flatten it to match the Tesseract output (2 layered nested dict of arrays)
    jac = {
        "scalars.a": {
            "scalars.a": jac["scalars"]["a"][0],
            "vectors.v": jac["scalars"]["a"][1],
        },
        "vectors.v": {
            "scalars.a": jac["vectors"]["v"][0],
            "vectors.v": jac["vectors"]["v"][1],
        },
    }
    _assert_pytree_isequal(jac, jac_ref)


@pytest.mark.parametrize("use_jit", [True, False])
def test_nested_tesseract_vmap(served_nested_tesseract_raw, use_jit):
    nested_tess = Tesseract(served_nested_tesseract_raw)
    b = np.array(2.0, dtype="float32")
    w = np.array([5.0, 7.0, 9.0], dtype="float32")

    def f(a, v, s, i):
        return apply_tesseract(
            nested_tess,
            inputs={
                "scalars": {"a": a, "b": b},
                "vectors": {"v": v, "w": w},
                "other_stuff": {"s": s, "i": i, "f": 2.718},
            },
        )

    def f_raw(a, v, s, i):
        return {
            "scalars": {"a": a * 10 + b, "b": b},
            "vectors": {"v": v * 10 + w, "w": w},
        }

    # add one batch dimension
    for a_axis in [None, 0]:
        for v_axis in [None, -1, 0, 1]:
            if a_axis is None and v_axis is None:
                continue
            if a_axis == 0:
                a = np.arange(4, dtype="float32")
            else:
                a = np.array(0.0, dtype="float32")
            if v_axis == 0:
                v = np.arange(12, dtype="float32").reshape((4, 3))
            elif v_axis in [-1, 1]:
                v = np.arange(12, dtype="float32").reshape((3, 4))
            else:
                v = np.arange(3, dtype="float32")

            mapped_in_axes = (a_axis, v_axis, None, None)

            for mapped_out_axes in [-1, 0, 1] if v_axis else [0]:
                if v_axis:
                    mapped_out_axes = {
                        "scalars": {"a": 0, "b": 0},
                        "vectors": {"v": mapped_out_axes, "w": 0},
                    }
                f_vmapped = jax.vmap(
                    f, in_axes=mapped_in_axes, out_axes=mapped_out_axes
                )
                raw_vmapped = jax.vmap(
                    f_raw, in_axes=mapped_in_axes, out_axes=mapped_out_axes
                )

                if use_jit:
                    f_vmapped = jax.jit(f_vmapped, static_argnames=["s", "i"])
                    raw_vmapped = jax.jit(raw_vmapped, static_argnames=["s", "i"])

                result = f_vmapped(a, v, "hello", 3)
                result_raw = raw_vmapped(a, v, "hello", 3)

                _assert_pytree_isequal(result, result_raw)


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


@pytest.mark.parametrize("use_jit", [True, False])
def test_non_abstract_tesseract_apply(served_non_abstract_tesseract, use_jit):
    non_abstract_tess = Tesseract(served_non_abstract_tesseract)
    a = np.array([0.0, 1.0, 2.0], dtype="float32")

    def f(a):
        return apply_tesseract(non_abstract_tess, inputs=dict(a=a))

    if use_jit:
        f = jax.jit(f)

        # make sure value error is raised if input shape is incorrect
        with pytest.raises(ValueError):
            f(a)

    else:
        # Test against Tesseract client
        result = f(a)
        result_ref = non_abstract_tess.apply(dict(a=a))
        _assert_pytree_isequal(result, result_ref)


@pytest.mark.parametrize("use_jit", [True, False])
def test_tesseract_loss(served_vectoradd_tesseract, use_jit):
    vectoradd_tess = Tesseract(served_vectoradd_tesseract)
    a = np.array([1.0, 2.0, 3.0], dtype="float32")

    # b = jax.lax.stop_gradient(b)

    def loss_fn(a):
        b = np.array([4.0, 5.0, 6.0], dtype="float32")

        vectoradd_fn_a: jax.Callable = lambda a: apply_tesseract(
            vectoradd_tess,
            inputs=dict(
                a=a,
                b=b,
            ),
        )

        c = vectoradd_fn_a(a)["c"]
        c = jax.lax.stop_gradient(c)

        vectoradd_fn_b: jax.Callable = lambda a: apply_tesseract(
            vectoradd_tess,
            inputs=dict(
                a=a,
                b=c,
            ),
        )

        outputs = vectoradd_fn_b(a)

        return jnp.sum((outputs["c"]) ** 2)

    if use_jit:
        loss_fn = jax.jit(loss_fn)

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    assert value_and_grad_fn(a) is not None


def test_non_abstract_tesseract_vjp(served_non_abstract_tesseract):
    non_abstract_tess = Tesseract(served_non_abstract_tesseract)

    a = np.array([1.0, 2.0, 3.0], dtype="float32")

    def f(a):
        return apply_tesseract(
            non_abstract_tess,
            inputs=dict(
                a=a,
            ),
        )

    with pytest.raises(ValueError):
        jax.vjp(f, a)
