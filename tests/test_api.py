# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract


def pytree_apply_impl(inputs: dict) -> dict:
    """JAX-traceable version of pytree_tesseract apply function."""
    x = inputs["alpha"]["x"]  # shape (3,)
    y = inputs["alpha"]["y"]  # shape (4,)
    z = inputs["beta"]["z"]  # shape (5,)
    u = inputs["beta"]["gamma"]["u"]  # shape (6,)
    v = inputs["beta"]["gamma"]["v"]  # shape (7,)
    d0 = inputs["delta"][0]  # shape (8,)
    d1 = inputs["delta"][1]  # shape (9,)
    k = inputs["epsilon"]["k"]  # shape (2,)
    m = inputs["epsilon"]["m"]  # shape (10,)
    z0 = inputs["zeta"][0]  # shape (11,)
    z1 = inputs["zeta"][1]  # shape (12,)

    # Complex operations with non-element-wise ops for non-trivial Jacobians
    # Since inputs have different shapes, we need to broadcast/slice appropriately

    # Element-wise terms (broadcast to shape 3)
    term1 = x * y[:3]
    # Dot product - couples all elements (use first 5 elements of v)
    term2 = jnp.dot(z, v[:5])
    # Reductions - each output depends on all input elements
    term3 = u.sum()
    # Mixed reduction and element-wise (broadcast to shape 3)
    term4 = d0[:3] * d1.mean()
    # Non-linear with reduction (broadcast to shape 3)
    term5 = jnp.exp(jnp.clip(k.sum() * 0.1, -5, 5)) * m[:3]

    result = term1 + term2 + term3 + term4 + term5 + z0[:3] + z1[:3]

    # Dictionary outputs with various coupling
    result_dict = {
        "a": x + y[:3] + z.mean(),  # shape (3,), reduction couples z to outputs
        "b": z
        + u[:5]
        + jnp.outer(x[:1], y[:1]).sum(),  # shape (5,), outer product coupling
    }

    # List outputs with reductions and cross-terms
    result_list = [
        d0[:7] + v + u.mean(),  # shape (7,), reduction couples all u elements
        d1[:6] + u + jnp.sum(d0[:6] * v[:6]),  # shape (6,), dot-product-like coupling
    ]

    return {
        "metadata": k + m[:2],  # shape (2,)
        "result": result,  # shape (3,)
        "result_dict": result_dict,
        "result_list": result_list,
    }


# Parametrization for testing different subsets of differentiable inputs
# This includes subsets of non pydantic model dicts
# and lists
DIFFABLE_PATHS_PARAMS = pytest.mark.parametrize(
    "diffable_paths",
    [
        ["alpha.x"],
        ["beta.gamma.v"],
        ["delta.0"],
        ["alpha.x", "alpha.y"],
        ["delta.0", "delta.1"],
        [
            "alpha.x",
            "alpha.y",
            "beta.z",
            "beta.gamma.u",
            "beta.gamma.v",
            "delta.0",
            "delta.1",
        ],
    ],
    ids=["single_x", "single_v", "single_list", "pair_xy", "pair_list", "all_inputs"],
)


def path_to_str(path):
    """Convert JAX tree path to string like 'alpha.x' or 'delta.0'."""
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(key.key)
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)


def filter_by_paths(inputs, paths_to_keep):
    """Filter pytree keeping only specified paths."""
    # mark entries not in paths_to_keep as None
    paths_set = set(paths_to_keep)

    def filter_fn(path, x):
        path_str = path_to_str(path)
        return x if path_str in paths_set else None

    filtered = jax.tree_util.tree_map_with_path(filter_fn, inputs)

    # Recursively remove None values from the tree
    def remove_nones(tree):
        if isinstance(tree, dict):
            result = {}
            for k, v in tree.items():
                cleaned = remove_nones(v)
                if cleaned is not None:
                    result[k] = cleaned
            return result if result else None

        elif isinstance(tree, (list, tuple)):
            result = []
            for v in tree:
                cleaned = remove_nones(v)
                if cleaned is not None:
                    result.append(cleaned)
            return type(tree)(result) if result else None
        return tree

    return remove_nones(filtered) or {}


def split_by_paths(inputs, diffable_paths):
    """Split inputs into diffable and non-diffable parts based on path list."""
    # Get all paths from inputs
    all_paths = set()
    jax.tree_util.tree_map_with_path(
        lambda path, x: all_paths.add(path_to_str(path)), inputs
    )

    diffable_paths_set = set(diffable_paths)
    non_diffable_paths = all_paths - diffable_paths_set

    diffable = filter_by_paths(inputs, diffable_paths)
    non_diffable = filter_by_paths(inputs, non_diffable_paths)
    return diffable, non_diffable


def merge_dicts(d1, d2):
    """Merge two dicts recursively. Assumes disjoint leaf keys.

    Lists are merged by concatenation, so ordering is only preserved when
    d1 contains the lower-indexed elements and d2 the higher-indexed ones.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1 if d1 is not None else d2

    result = {}
    all_keys = set(d1.keys()) | set(d2.keys())

    for key in all_keys:
        if (
            key in d1
            and key in d2
            and isinstance(d1[key], list)
            and isinstance(d2[key], list)
        ):
            result[key] = d1[key] + d2[key]
        elif key in d1 and key in d2:
            result[key] = merge_dicts(d1[key], d2[key])
        elif key in d1:
            result[key] = d1[key]
        else:
            result[key] = d2[key]

    return result


@pytest.mark.parametrize("use_jit", [True, False])
def test_pytree_tesseract_primal(pytree_tess, pytree_tess_inputs, use_jit):
    """Test the primal (apply) endpoint of pytree_tesseract."""

    def f(inputs):
        return apply_tesseract(pytree_tess, inputs=inputs)

    if use_jit:
        f = jax.jit(f)

    _ = f(pytree_tess_inputs)


@pytest.mark.parametrize("use_jit", [True, False])
@DIFFABLE_PATHS_PARAMS
def test_pytree_tesseract_jvp(pytree_tess, pytree_tess_inputs, use_jit, diffable_paths):
    """Test the JVP endpoint of pytree_tesseract."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        pytree_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(pytree_tess, inputs=inputs)

    def f_raw(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return pytree_apply_impl(inputs)

    if use_jit:
        f = jax.jit(f)
        f_raw = jax.jit(f_raw)

    primal, jvp = jax.jvp(f, (diffable_inputs,), (diffable_inputs,))
    primal_raw, jvp_raw = jax.jvp(f_raw, (diffable_inputs,), (diffable_inputs,))

    # Verify results match raw implementation
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
        primal,
        primal_raw,
    )
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
        jvp,
        jvp_raw,
    )


@pytest.mark.parametrize("use_jit", [True, False])
@DIFFABLE_PATHS_PARAMS
def test_pytree_tesseract_vjp(pytree_tess, pytree_tess_inputs, use_jit, diffable_paths):
    """Test the VJP endpoint of pytree_tesseract."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        pytree_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(pytree_tess, inputs=inputs)

    def f_raw(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return pytree_apply_impl(inputs)

    if use_jit:
        f = jax.jit(f)
        f_raw = jax.jit(f_raw)

    (primal, f_vjp) = jax.vjp(f, diffable_inputs)
    (primal_raw, f_vjp_raw) = jax.vjp(f_raw, diffable_inputs)

    if use_jit:
        f_vjp = jax.jit(f_vjp)
        f_vjp_raw = jax.jit(f_vjp_raw)

    vjp = f_vjp(primal)
    vjp_raw = f_vjp_raw(primal_raw)

    # Verify results match raw implementation
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
        primal,
        primal_raw,
    )
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
        vjp,
        vjp_raw,
    )


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("jac_direction", ["fwd", "rev"])
@DIFFABLE_PATHS_PARAMS
def test_pytree_tesseract_jacobian(
    pytree_tess, pytree_tess_inputs, use_jit, jac_direction, diffable_paths
):
    """Test the Jacobian endpoint of pytree_tesseract using jacfwd and jacrev."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        pytree_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(pytree_tess, inputs=inputs)

    def f_raw(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return pytree_apply_impl(inputs)

    if jac_direction == "fwd":
        f_jac = jax.jacfwd(f)
        f_jac_raw = jax.jacfwd(f_raw)
    else:
        f_jac = jax.jacrev(f)
        f_jac_raw = jax.jacrev(f_raw)

    if use_jit:
        f_jac = jax.jit(f_jac)
        f_jac_raw = jax.jit(f_jac_raw)

    jac = f_jac(diffable_inputs)
    jac_raw = f_jac_raw(diffable_inputs)

    # Verify results match raw implementation
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
        jac,
        jac_raw,
    )


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@DIFFABLE_PATHS_PARAMS
def test_pytree_tesseract_loss(
    pytree_tess, pytree_tess_inputs, use_jit, mode, diffable_paths
):
    """Test that non-differentiable inputs in closure don't get included in VJP/JVP for pytree."""
    # Split inputs into differentiable and non-differentiable based on test parameters
    diffable_inputs, non_diffable_inputs = split_by_paths(
        pytree_tess_inputs, diffable_paths
    )

    # Also include the non-differentiable epsilon and zeta inputs as function arguments
    # to trigger the tracer issue
    k = pytree_tess_inputs["epsilon"]["k"]  # non-differentiable per schema
    m = pytree_tess_inputs["epsilon"]["m"]  # non-differentiable per schema
    z0 = pytree_tess_inputs["zeta"][0]  # non-differentiable per schema
    z1 = pytree_tess_inputs["zeta"][1]  # non-differentiable per schema

    def loss_fn(diffable_inputs, k, m, z0, z1):
        # Lambda closes over non-differentiable inputs, making them tracers
        pytree_fn: jax.Callable = lambda diffable_inputs: apply_tesseract(
            pytree_tess,
            inputs=merge_dicts(
                diffable_inputs,
                merge_dicts(
                    non_diffable_inputs, {"epsilon": {"k": k, "m": m}, "zeta": [z0, z1]}
                ),
            ),
        )

        result = pytree_fn(diffable_inputs)["result"]
        return jnp.sum(result**2)

    if use_jit:
        loss_fn = jax.jit(loss_fn)

    if mode == "fwd":
        # Forward mode: JVP
        # Note: k, m, z0, z1 get passed as tracers to apply_tesseract.
        # The schema says they are non-differentiable, so they should not be included in jvp_inputs.
        primal, tangent = jax.jvp(
            loss_fn, (diffable_inputs, k, m, z0, z1), (diffable_inputs, k, m, z0, z1)
        )
        assert primal is not None
        assert tangent is not None
    else:
        # Reverse mode: VJP via grad
        # Note: k, m, z0, z1 get passed as tracers to apply_tesseract.
        # The schema says they are non-differentiable, so they should not be included in vjp_inputs.
        # Only differentiate w.r.t. diffable_inputs (argnums=0)
        value_and_grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        assert value_and_grad_fn(diffable_inputs, k, m, z0, z1) is not None


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_tesseract_loss(vectoradd_tess, use_jit, mode):
    """Test that non-differentiable inputs in closure don't get included in VJP/JVP."""
    a = np.array([1.0, 2.0, 3.0], dtype="float32")
    b = np.array([4.0, 5.0, 6.0], dtype="float32")

    def loss_fn(a, b):
        vectoradd_fn_a: jax.Callable = lambda a: apply_tesseract(
            vectoradd_tess,
            inputs=dict(
                a=a,
                b=b,
            ),
        )

        c = vectoradd_fn_a(a)["c"]

        return jnp.sum((c) ** 2)

    if use_jit:
        loss_fn = jax.jit(loss_fn)

    if mode == "fwd":
        # Forward mode: JVP
        # Note: 'b' gets passed as a tracer to apply_tesseract.
        # The schema says 'b' is non-differentiable, so it should not be included in jvp_inputs.
        primal, tangent = jax.jvp(loss_fn, (a, b), (a, b))
        assert primal is not None
        assert tangent is not None
    else:
        # Reverse mode: VJP via grad
        # Note: This should only differentiate w.r.t. the first argument 'a',
        # but 'b' gets passed as a tracer to apply_tesseract.
        # The schema says 'b' is non-differentiable, so it should not be included in vjp_inputs.
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        assert value_and_grad_fn(a, b) is not None


def rosenbrock(x: float, y: float, a: float = 1.0, b: float = 100.0) -> float:
    return (a - x) ** 2 + b * (y - x**2) ** 2


def test_tesseract_loss_univariate(univariate_tess):
    x = np.array(1.0, dtype="float64")
    y = np.array(2.0, dtype="float64")

    # First verify forward pass
    result = apply_tesseract(univariate_tess, inputs=dict(x=x, y=y))["result"]
    expected = rosenbrock(x, y, a=1.0, b=100.0)
    assert np.allclose(result, expected), (
        f"Forward pass mismatch: {result} vs {expected}"
    )

    def loss_fn(x, y):
        univariate_fn_x: jax.Callable = lambda x: apply_tesseract(
            univariate_tess,
            inputs=dict(
                x=x,
                y=y,
            ),
        )

        c = univariate_fn_x(x)["result"]

        return jnp.sum((c) ** 2)

    def loss_fn_raw(x, y):
        c = rosenbrock(x, y, a=1.0, b=100.0)
        return jnp.sum((c) ** 2)

    loss_fn = jax.jit(loss_fn)

    grad_fn = jax.grad(loss_fn)

    grad = grad_fn(x, y)

    assert grad is not None

    grad_fn_raw = jax.grad(loss_fn_raw)

    grad_raw = grad_fn_raw(x, y)

    assert np.allclose(grad, grad_raw)
