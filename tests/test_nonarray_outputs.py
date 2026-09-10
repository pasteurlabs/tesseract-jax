# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for an OutputSchema that mixes arrays with a str and a bool.

A JAX primitive can only return arrays, so non-array leaves are taken from
`abstract_eval`, carried as static primitive parameters, and put back into the
output pytree after the bind. These tests check that the arrays are untouched:
the static leaves must not shift, drop or reorder anything the gradient path
depends on.

Before this was supported, all of these raised
`TypeError: string indices must be integers`.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract
from tesseract_jax.primitive import CHECK_STATIC_OUTPUTS_ENV_VAR
from tesseract_jax.tree_util import _leaves_differ

X = jnp.arange(3, dtype="float64")


def test_static_leaves_come_back_beside_the_arrays(nonarray_output_tess):
    out = apply_tesseract(nonarray_output_tess, dict(x=X))

    assert set(out) == {"y", "backend", "converged"}
    np.testing.assert_allclose(out["y"], 2.0 * X)
    assert out["backend"] == "reference"
    assert out["converged"] is True


def test_a_static_leaf_is_a_python_value_inside_a_trace(nonarray_output_tess):
    """A static leaf is an ordinary Python object inside a jit trace.

    It can be branched on while tracing, but cannot be returned from the jitted
    function, since JAX has no type for a str output. The pattern is to consume
    the leaf inside the trace and return the arrays.
    """
    seen = {}

    @jax.jit
    def f(x):
        out = apply_tesseract(nonarray_output_tess, dict(x=x))
        seen["backend"] = out["backend"]
        # A Python bool, so this really is a trace-time branch.
        scale = 1.0 if out["converged"] else 0.0
        return out["y"] * scale

    np.testing.assert_allclose(f(X), 2.0 * X)
    assert seen["backend"] == "reference"


def test_the_opaque_subscript_error_is_gone(nonarray_output_tess):
    """Regression test for the original TypeError."""
    out = apply_tesseract(nonarray_output_tess, dict(x=X))
    np.testing.assert_allclose(out["y"], 2.0 * X)


@pytest.mark.parametrize("use_jit", [True, False])
def test_the_gradient_is_unaffected_by_a_static_leaf(nonarray_output_tess, use_jit):
    """A static leaf must not disturb the cotangents.

    The statics sit either side of `y` in the schema, so a bookkeeping error
    shows up here as a shifted or missing cotangent.
    """

    def loss(x):
        return jnp.sum(apply_tesseract(nonarray_output_tess, dict(x=x))["y"] ** 2)

    if use_jit:
        loss = jax.jit(loss)

    # y = 2x, so d/dx sum(y^2) = 8x.
    np.testing.assert_allclose(jax.grad(loss)(X), 8.0 * X, rtol=1e-6)


def test_forward_mode_is_unaffected_by_a_static_leaf(nonarray_output_tess):
    def f(x):
        return apply_tesseract(nonarray_output_tess, dict(x=x))["y"]

    primal, tangent = jax.jvp(f, (X,), (jnp.ones_like(X),))
    np.testing.assert_allclose(primal, 2.0 * X)
    np.testing.assert_allclose(tangent, 2.0 * jnp.ones_like(X))


def test_vmap_is_unaffected_by_a_static_leaf(nonarray_output_tess):
    xs = jnp.stack([X, X + 1.0])

    def f(x):
        return apply_tesseract(
            nonarray_output_tess, dict(x=x), vmap_method="sequential"
        )["y"]

    np.testing.assert_allclose(jax.vmap(f)(xs), 2.0 * xs)


def test_no_warning_when_apply_agrees_with_abstract_eval(drifting_static_tess):
    """The check must stay quiet when the two endpoints agree."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = apply_tesseract(drifting_static_tess, dict(x=X))

    assert out["backend"] == "reference"


def test_a_static_leaf_that_apply_disagrees_with_is_warned_about(drifting_static_tess):
    """A drifting static leaf from `apply` is warned about, not dropped silently.

    The warning names the leaf, the value `apply` returned, and the value the
    caller is holding.
    """
    with pytest.warns(UserWarning, match="backend") as record:
        out = apply_tesseract(drifting_static_tess, dict(x=-X))

    assert out["backend"] == "reference"
    message = str(record[0].message)
    assert "'fallback'" in message
    assert "'reference'" in message


def test_the_warning_survives_jit(drifting_static_tess):
    """The check fires under jit too.

    It runs inside the callback that `apply` is dispatched from, not at trace
    time, so jit does not suppress it.
    """
    seen = {}

    @jax.jit
    def f(x):
        out = apply_tesseract(drifting_static_tess, dict(x=x))
        seen["backend"] = out["backend"]
        return out["y"]

    with pytest.warns(UserWarning, match="backend"):
        y = f(-X)

    np.testing.assert_allclose(y, -2.0 * X)
    assert seen["backend"] == "reference"


def test_the_drift_warning_does_not_disturb_the_gradient(drifting_static_tess):
    def loss(x):
        return jnp.sum(apply_tesseract(drifting_static_tess, dict(x=x))["y"] ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = jax.grad(loss)(-X)

    np.testing.assert_allclose(grad, -8.0 * X, rtol=1e-6)


def test_the_check_can_be_turned_off_per_call(drifting_static_tess):
    """`check_static_outputs=False` skips the comparison for one call.

    The response is then flattened without keypaths, and the value the caller
    gets is unchanged.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = apply_tesseract(
            drifting_static_tess, dict(x=-X), check_static_outputs=False
        )

    assert out["backend"] == "reference"
    np.testing.assert_allclose(out["y"], -2.0 * X)


def test_turning_the_check_off_leaves_the_gradient_alone(drifting_static_tess):
    def loss(x):
        return jnp.sum(
            apply_tesseract(
                drifting_static_tess, dict(x=x), check_static_outputs=False
            )["y"]
            ** 2
        )

    grad = jax.grad(loss)(-X)

    np.testing.assert_allclose(grad, -8.0 * X, rtol=1e-6)


def test_turning_the_check_off_for_one_call_leaves_the_next_one_alone(
    drifting_static_tess,
):
    """The kwarg applies to one call only, not to later ones."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_tesseract(drifting_static_tess, dict(x=-X), check_static_outputs=False)

    with pytest.warns(UserWarning, match="backend"):
        apply_tesseract(drifting_static_tess, dict(x=-X))


@pytest.mark.parametrize("value", ["0", "false", "no", "OFF", " 0 "])
def test_the_env_var_turns_the_check_off(drifting_static_tess, monkeypatch, value):
    monkeypatch.setenv(CHECK_STATIC_OUTPUTS_ENV_VAR, value)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = apply_tesseract(drifting_static_tess, dict(x=-X))

    assert out["backend"] == "reference"


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_env_var_can_also_be_true(drifting_static_tess, monkeypatch, value):
    monkeypatch.setenv(CHECK_STATIC_OUTPUTS_ENV_VAR, value)

    with pytest.warns(UserWarning, match="backend"):
        apply_tesseract(drifting_static_tess, dict(x=-X))


def test_the_kwarg_beats_the_env_var(drifting_static_tess, monkeypatch):
    monkeypatch.setenv(CHECK_STATIC_OUTPUTS_ENV_VAR, "0")

    with pytest.warns(UserWarning, match="backend"):
        apply_tesseract(drifting_static_tess, dict(x=-X), check_static_outputs=True)


def test_a_non_boolean_env_var_is_an_error(drifting_static_tess, monkeypatch):
    monkeypatch.setenv(CHECK_STATIC_OUTPUTS_ENV_VAR, "maybe")

    with pytest.raises(ValueError, match=CHECK_STATIC_OUTPUTS_ENV_VAR):
        apply_tesseract(drifting_static_tess, dict(x=X))


def test_leaves_that_do_not_compare_to_a_bool_fall_back_to_identity():
    """Ensure that leaves which do not compare to a bool (arrays) fall back to identity."""
    a = np.zeros(3)

    assert not _leaves_differ(a, a)
    assert _leaves_differ(a, np.zeros(3))
