# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dict_tesseract API endpoints: primal (apply), vjp, and jacobian."""

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
            failures.append((a_path, str(e)))

    if failures:
        msg = "\n".join(f"Path: {path}, Error: {error}" for path, error in failures)
        raise AssertionError(f"PyTree elements are different:\n{msg}")


def dict_tess_impl(x, y):
    """JAX-traceable reference implementation of dict_tesseract (element-wise product)."""
    return x * y


@pytest.fixture
def dict_tess():
    """Load dict_tesseract directly from the API file."""
    return Tesseract.from_tesseract_api("tests/dict_tesseract/tesseract_api.py")


# =============================================================================
# Primal (Apply) Tests
# =============================================================================


@pytest.mark.parametrize("use_jit", [True, False])
def test_dict_tesseract_primal(dict_tess, use_jit):
    """Test the primal (apply) endpoint of dict_tesseract."""
    x = np.array([1.0, 2.0, 3.0], dtype="float32")
    y = np.array([4.0, 5.0, 6.0], dtype="float32")

    def f(x, y):
        return apply_tesseract(dict_tess, inputs={"parameters": {"x": x, "y": y}})

    raw_impl = dict_tess_impl
    if use_jit:
        f = jax.jit(f)
        raw_impl = jax.jit(raw_impl)

    # Test against Tesseract client
    result = f(x, y)
    result_ref = dict_tess.apply({"parameters": {"x": x, "y": y}})
    _assert_pytree_isequal(result, result_ref)

    # Test against direct implementation
    result_raw = raw_impl(x, y)
    np.testing.assert_array_equal(result["result"], result_raw)


# =============================================================================
# VJP (Vector-Jacobian Product) Tests
# =============================================================================


@pytest.mark.parametrize("use_jit", [True, False])
def test_dict_tesseract_vjp(dict_tess, use_jit):
    """Test the VJP endpoint of dict_tesseract."""
    x = np.array([1.0, 2.0, 3.0], dtype="float32")
    y = np.array([4.0, 5.0, 6.0], dtype="float32")

    def f(x, y):
        return apply_tesseract(dict_tess, inputs={"parameters": {"x": x, "y": y}})

    raw_impl = dict_tess_impl
    if use_jit:
        f = jax.jit(f)
        raw_impl = jax.jit(raw_impl)

    (primal, f_vjp) = jax.vjp(f, x, y)

    if use_jit:
        f_vjp = jax.jit(f_vjp)

    vjp = f_vjp(primal)

    # Test primal against Tesseract client
    primal_ref = dict_tess.apply({"parameters": {"x": x, "y": y}})
    _assert_pytree_isequal(primal, primal_ref)

    # Test against direct implementation
    primal_raw, f_vjp_raw = jax.vjp(raw_impl, x, y)
    if use_jit:
        f_vjp_raw = jax.jit(f_vjp_raw)
    vjp_raw = f_vjp_raw(primal_raw)
    # JAX vjp returns a flat tuple
    np.testing.assert_array_equal(vjp[0], vjp_raw[0])
    np.testing.assert_array_equal(vjp[1], vjp_raw[1])


# =============================================================================
# Jacobian Tests
# =============================================================================


@pytest.mark.parametrize("use_jit", [True, False])
@pytest.mark.parametrize("jac_direction", ["fwd", "rev"])
def test_dict_tesseract_jacobian(dict_tess, use_jit, jac_direction):
    """Test the Jacobian endpoint of dict_tesseract using jacfwd and jacrev."""
    x = np.array([1.0, 2.0, 3.0], dtype="float32")
    y = np.array([4.0, 5.0, 6.0], dtype="float32")

    def f(x, y):
        return apply_tesseract(dict_tess, inputs={"parameters": {"x": x, "y": y}})[
            "result"
        ]

    if jac_direction == "fwd":
        f_jac = jax.jacfwd(f, argnums=(0, 1))
        raw_jac = jax.jacfwd(dict_tess_impl, argnums=(0, 1))
    else:
        f_jac = jax.jacrev(f, argnums=(0, 1))
        raw_jac = jax.jacrev(dict_tess_impl, argnums=(0, 1))

    if use_jit:
        f_jac = jax.jit(f_jac)
        raw_jac = jax.jit(raw_jac)

    jac = f_jac(x, y)

    # Test against direct implementation
    jac_raw = raw_jac(x, y)
    _assert_pytree_isequal(jac, jac_raw)
