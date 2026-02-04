# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest

from tesseract_jax import apply_tesseract


def path_to_str(path):
    """Convert JAX tree path to string like 'alpha.x'."""
    return ".".join(key.key if hasattr(key, "key") else str(key) for key in path)


def filter_by_paths(inputs, paths_to_keep):
    """Filter pytree keeping only specified paths."""
    paths_set = set(paths_to_keep)

    def filter_fn(path, x):
        path_str = path_to_str(path)
        return x if path_str in paths_set else None

    filtered = jax.tree_util.tree_map_with_path(filter_fn, inputs)

    # Remove None values from the tree
    def remove_nones(tree):
        if isinstance(tree, dict):
            result = {k: remove_nones(v) for k, v in tree.items() if v is not None}
            return result if result else None
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
        if key in d1 and key in d2:
            result[key] = merge_dicts(d1[key], d2[key])
        elif key in d1:
            result[key] = d1[key]
        else:
            result[key] = d2[key]

    return result


@pytest.mark.parametrize("use_jit", [True, False])
def test_dict_tesseract_primal(dict_tess, dict_tess_inputs, use_jit):
    """Test the primal (apply) endpoint of dict_tesseract."""

    def f(inputs):
        return apply_tesseract(dict_tess, inputs=inputs)

    if use_jit:
        f = jax.jit(f)

    _ = f(dict_tess_inputs)


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize(
    "diffable_paths",
    [
        ["alpha.x"],
        ["beta.gamma.v"],
        ["alpha.x", "alpha.y"],
        ["alpha.x", "alpha.y", "beta.z", "beta.gamma.u", "beta.gamma.v"],
    ],
    ids=["single_x", "single_v", "pair_xy", "all_inputs"],
)
def test_dict_tesseract_jvp(dict_tess, dict_tess_inputs, use_jit, diffable_paths):
    """Test the JVP endpoint of dict_tesseract."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        dict_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(dict_tess, inputs=inputs)["result"]

    if use_jit:
        f = jax.jit(f)

    _ = jax.jvp(f, (diffable_inputs,), (diffable_inputs,))


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize(
    "diffable_paths",
    [
        ["alpha.x"],
        ["beta.gamma.v"],
        ["alpha.x", "alpha.y"],
        ["alpha.x", "alpha.y", "beta.z", "beta.gamma.u", "beta.gamma.v"],
    ],
    ids=["single_x", "single_v", "pair_xy", "all_inputs"],
)
def test_dict_tesseract_vjp(dict_tess, dict_tess_inputs, use_jit, diffable_paths):
    """Test the VJP endpoint of dict_tesseract."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        dict_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(dict_tess, inputs=inputs)["result"]

    if use_jit:
        f = jax.jit(f)

    (primal, f_vjp) = jax.vjp(f, diffable_inputs)

    if use_jit:
        f_vjp = jax.jit(f_vjp)

    _ = f_vjp(primal)


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("jac_direction", ["fwd", "rev"])
@pytest.mark.parametrize(
    "diffable_paths",
    [
        ["alpha.x"],
        ["beta.gamma.v"],
        ["alpha.x", "alpha.y"],
        ["alpha.x", "alpha.y", "beta.z", "beta.gamma.u", "beta.gamma.v"],
    ],
    ids=["single_x", "single_v", "pair_xy", "all_inputs"],
)
def test_dict_tesseract_jacobian(
    dict_tess, dict_tess_inputs, use_jit, jac_direction, diffable_paths
):
    """Test the Jacobian endpoint of dict_tesseract using jacfwd and jacrev."""
    diffable_inputs, non_diffable_inputs = split_by_paths(
        dict_tess_inputs, diffable_paths
    )

    def f(diffable_inputs):
        inputs = merge_dicts(diffable_inputs, non_diffable_inputs)
        return apply_tesseract(dict_tess, inputs=inputs)["result"]

    if jac_direction == "fwd":
        f_jac = jax.jacfwd(f)
    else:
        f_jac = jax.jacrev(f)

    if use_jit:
        f_jac = jax.jit(f_jac)

    _ = f_jac(diffable_inputs)
