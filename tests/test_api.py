# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest

from tesseract_jax import apply_tesseract

# Parametrization for testing different subsets of differentiable inputs
# This inlcudes subsets of non pydantic model dicts
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
    """Merge two dicts recursively. Assumes disjoint leaf keys."""
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
        return apply_tesseract(pytree_tess, inputs=inputs)["result"]

    if use_jit:
        f = jax.jit(f)

    _ = jax.jvp(f, (diffable_inputs,), (diffable_inputs,))


@pytest.mark.parametrize("use_jit", [True, False])
@DIFFABLE_PATHS_PARAMS
def test_pytree_tesseract_vjp(pytree_tess, pytree_tess_inputs, use_jit, diffable_paths):
    """Test the VJP endpoint of pytree_tesseract."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        pytree_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(pytree_tess, inputs=inputs)["result"]

    if use_jit:
        f = jax.jit(f)

    (primal, f_vjp) = jax.vjp(f, diffable_inputs)

    if use_jit:
        f_vjp = jax.jit(f_vjp)

    _ = f_vjp(primal)


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
        return apply_tesseract(pytree_tess, inputs=inputs)["result"]

    if jac_direction == "fwd":
        f_jac = jax.jacfwd(f)
    else:
        f_jac = jax.jacrev(f)

    if use_jit:
        f_jac = jax.jit(f_jac)

    _ = f_jac(diffable_inputs)
