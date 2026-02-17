# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract


def pytree_apply_impl(inputs: dict) -> dict:
    """JAX-traceable version of pytree_tesseract apply function."""
    x = inputs["alpha"]["x"]
    y = inputs["alpha"]["y"]
    z = inputs["beta"]["z"]
    u = inputs["beta"]["gamma"]["u"]
    v = inputs["beta"]["gamma"]["v"]
    d0 = inputs["delta"][0]
    d1 = inputs["delta"][1]
    k = inputs["epsilon"]["k"]
    m = inputs["epsilon"]["m"]
    z0 = inputs["zeta"][0]
    z1 = inputs["zeta"][1]

    # Complex operations with non-element-wise ops for non-trivial Jacobians
    # Element-wise terms
    term1 = x * y
    # Dot product - couples all elements
    term2 = jnp.dot(z, v)
    # Reductions - each output depends on all input elements
    term3 = u.sum()
    # Mixed reduction and element-wise
    term4 = d0 * d1.mean()
    # Non-linear with reduction
    term5 = jnp.exp(jnp.clip(k.sum() * 0.1, -5, 5)) * m

    result = term1 + term2 + term3 + term4 + term5 + z0 + z1

    # Dictionary outputs with various coupling
    result_dict = {
        "a": x + y + z.mean(),  # reduction couples z to outputs
        "b": z + u + jnp.outer(x[:1], y[:1]).sum(),  # outer product coupling
    }

    # List outputs with reductions and cross-terms
    result_list = [
        d0 + v + u.mean(),  # reduction couples all u elements
        d1 + u + jnp.sum(d0 * v),  # dot-product-like coupling
    ]

    return {
        "metadata": k + m,
        "result": result,
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
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b), primal, primal_raw)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b), jvp, jvp_raw)


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
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b), primal, primal_raw)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b), vjp, vjp_raw)


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
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b), jac, jac_raw)


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_tesseract_loss(vectoradd_tess, use_jit, mode):
    """Test nested tesseract calls with stop_gradient in both forward and reverse mode."""
    a = np.array([1.0, 2.0, 3.0], dtype="float32")

    def loss_fn(a):
        b = np.array([4.0, 5.0, 6.0], dtype="float32")

        c = apply_tesseract(vectoradd_tess, inputs=dict(a=a, b=b))["c"]
        c = jax.lax.stop_gradient(c)

        outputs = apply_tesseract(vectoradd_tess, inputs=dict(a=a, b=c))

        return jnp.sum((outputs["c"]) ** 2)

    if use_jit:
        loss_fn = jax.jit(loss_fn)

    if mode == "fwd":
        # Forward mode: JVP
        primal, tangent = jax.jvp(loss_fn, (a,), (a,))
        assert primal is not None
        assert tangent is not None
    else:
        # Reverse mode: VJP via grad
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        assert value_and_grad_fn(a) is not None


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
