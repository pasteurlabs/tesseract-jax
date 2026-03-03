# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract, save_intermediates, sow


def maybe_jit(fn, use_jit):
    """Optionally wrap *fn* in jax.jit."""
    return jax.jit(fn) if use_jit else fn


jit_parametrize = pytest.mark.parametrize(
    "use_jit", [False, True], ids=["nojit", "jit"]
)


class TestSowIdentity:
    """sow acts as pure identity outside save_intermediates."""

    @jit_parametrize
    def test_scalar(self, use_jit):
        def fn(x):
            return sow(x, "test")

        result = maybe_jit(fn, use_jit)(jnp.array(1.0))
        np.testing.assert_array_equal(result, 1.0)

    @jit_parametrize
    def test_array(self, use_jit):
        x = jnp.array([1.0, 2.0, 3.0])

        def fn(x):
            return sow(x, "test")

        result = maybe_jit(fn, use_jit)(x)
        np.testing.assert_array_equal(result, x)

    @jit_parametrize
    def test_pytree(self, use_jit):
        def fn(x):
            tree = {"a": x, "b": x * 2}
            return sow(tree, "test")

        result = maybe_jit(fn, use_jit)(jnp.array([1.0, 2.0]))
        np.testing.assert_array_equal(result["a"], jnp.array([1.0, 2.0]))
        np.testing.assert_array_equal(result["b"], jnp.array([2.0, 4.0]))

    @jit_parametrize
    def test_grad_passthrough(self, use_jit):
        """Sow doesn't affect gradients when used without save_intermediates."""

        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y.sum()

        x = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(maybe_jit(fn, use_jit))(x)
        np.testing.assert_array_equal(grad, 2 * x)


class TestSowForward:
    """save_intermediates captures primal values in forward-only mode."""

    @jit_parametrize
    def test_scalar(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y

        result, intermediates = save_intermediates(maybe_jit(fn, use_jit))(
            jnp.array(3.0)
        )
        np.testing.assert_array_equal(result, 9.0)
        np.testing.assert_array_equal(intermediates["squared"]["primal"], 9.0)

    @jit_parametrize
    def test_array(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y.sum()

        x = jnp.array([1.0, 2.0, 3.0])
        result, intermediates = save_intermediates(maybe_jit(fn, use_jit))(x)
        np.testing.assert_array_equal(result, 14.0)
        np.testing.assert_array_equal(intermediates["squared"]["primal"], x**2)

    @jit_parametrize
    def test_pytree(self, use_jit):
        def fn(x):
            result = {"a": x, "b": x**2}
            result = sow(result, "intermediate")
            return result["a"].sum() + result["b"].sum()

        x = jnp.array([1.0, 2.0])
        result, intermediates = save_intermediates(maybe_jit(fn, use_jit))(x)
        np.testing.assert_array_equal(result, 8.0)
        assert isinstance(intermediates["intermediate"]["primal"], dict)
        np.testing.assert_array_equal(intermediates["intermediate"]["primal"]["a"], x)
        np.testing.assert_array_equal(
            intermediates["intermediate"]["primal"]["b"], x**2
        )

    @jit_parametrize
    def test_nested_dict(self, use_jit):
        def fn(x):
            result = {"outer": {"a": x, "b": x**2}}
            result = sow(result, "step")
            return result["outer"]["a"].sum() + result["outer"]["b"].sum()

        x = jnp.array([1.0, 2.0])
        result, intermediates = save_intermediates(maybe_jit(fn, use_jit))(x)
        np.testing.assert_array_equal(result, 8.0)
        np.testing.assert_array_equal(intermediates["step"]["primal"]["outer"]["a"], x)
        np.testing.assert_array_equal(
            intermediates["step"]["primal"]["outer"]["b"], x**2
        )

    @jit_parametrize
    def test_multiple_sows(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "step1")
            z = y + 1
            z = sow(z, "step2")
            return z.sum()

        x = jnp.array([1.0, 2.0, 3.0])
        result, intermediates = save_intermediates(maybe_jit(fn, use_jit))(x)
        np.testing.assert_array_equal(result, 17.0)
        np.testing.assert_array_equal(intermediates["step1"]["primal"], x**2)
        np.testing.assert_array_equal(intermediates["step2"]["primal"], x**2 + 1)

    @jit_parametrize
    def test_no_sow_returns_empty(self, use_jit):
        def fn(x):
            return (x**2).sum()

        result, intermediates = save_intermediates(maybe_jit(fn, use_jit))(
            jnp.array([1.0, 2.0])
        )
        np.testing.assert_array_equal(result, 5.0)
        assert intermediates == {}


class TestSowGrad:
    """save_intermediates captures primals and cotangents with jax.grad."""

    @jit_parametrize
    def test_basic_grad(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y.sum()

        x = jnp.array([1.0, 2.0, 3.0])
        grad, intermediates = save_intermediates(jax.grad(maybe_jit(fn, use_jit)))(x)
        np.testing.assert_array_equal(grad, 2 * x)
        np.testing.assert_array_equal(intermediates["squared"]["primal"], x**2)
        # cotangent of y when loss = sum(y) is all ones
        np.testing.assert_array_equal(
            intermediates["squared"]["cotangent"], jnp.ones(3)
        )

    @jit_parametrize
    def test_grad_pytree(self, use_jit):
        def fn(x):
            result = {"a": x, "b": x**2}
            result = sow(result, "step")
            return result["a"].sum() + result["b"].sum()

        x = jnp.array([1.0, 2.0])
        grad, intermediates = save_intermediates(jax.grad(maybe_jit(fn, use_jit)))(x)
        np.testing.assert_array_equal(grad, 1 + 2 * x)
        np.testing.assert_array_equal(intermediates["step"]["primal"]["a"], x)
        np.testing.assert_array_equal(intermediates["step"]["primal"]["b"], x**2)
        np.testing.assert_array_equal(
            intermediates["step"]["cotangent"]["a"], jnp.ones(2)
        )
        np.testing.assert_array_equal(
            intermediates["step"]["cotangent"]["b"], jnp.ones(2)
        )

    @jit_parametrize
    def test_grad_multiple_sows(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "step1")
            z = y + 1
            z = sow(z, "step2")
            return z.sum()

        x = jnp.array([1.0, 2.0, 3.0])
        grad, intermediates = save_intermediates(jax.grad(maybe_jit(fn, use_jit)))(x)
        np.testing.assert_array_equal(grad, 2 * x)
        assert "step1" in intermediates
        assert "step2" in intermediates
        assert "primal" in intermediates["step1"]
        assert "cotangent" in intermediates["step1"]
        assert "primal" in intermediates["step2"]
        assert "cotangent" in intermediates["step2"]


class TestSowVjp:
    """save_intermediates captures primals and cotangents with jax.vjp."""

    @jit_parametrize
    def test_explicit_vjp(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y.sum()

        def with_vjp(x):
            primals, f_vjp = jax.vjp(maybe_jit(fn, use_jit), x)
            grad = f_vjp(jnp.ones_like(primals))
            return grad[0]

        x = jnp.array([1.0, 2.0, 3.0])
        result, intermediates = save_intermediates(with_vjp)(x)
        np.testing.assert_array_equal(result, 2 * x)
        assert "squared" in intermediates
        np.testing.assert_array_equal(intermediates["squared"]["primal"], x**2)
        np.testing.assert_array_equal(
            intermediates["squared"]["cotangent"], jnp.ones(3)
        )


class TestSowJvp:
    """save_intermediates captures primals and tangents with jax.jvp."""

    @jit_parametrize
    def test_jvp(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y.sum()

        def with_jvp(x):
            primals, _ = jax.jvp(maybe_jit(fn, use_jit), (x,), (jnp.ones_like(x),))
            return primals

        x = jnp.array([1.0, 2.0, 3.0])
        result, intermediates = save_intermediates(with_jvp)(x)
        np.testing.assert_array_equal(result, 14.0)
        np.testing.assert_array_equal(intermediates["squared"]["primal"], x**2)
        # tangent of x^2 at x with dx=1 is 2*x
        np.testing.assert_array_equal(intermediates["squared"]["tangent"], 2 * x)


class TestSowJacobian:
    """save_intermediates captures intermediates with jax.jacobian."""

    @jit_parametrize
    def test_jacobian(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared")
            return y

        x = jnp.array([1.0, 2.0])
        jac, intermediates = save_intermediates(jax.jacobian(maybe_jit(fn, use_jit)))(x)
        expected_jac = jnp.diag(2 * x)
        np.testing.assert_array_equal(jac, expected_jac)
        assert "squared" in intermediates


class TestSowDuplicateName:
    """Duplicate sow names raise ValueError."""

    @jit_parametrize
    def test_duplicate_raises(self, use_jit):
        def fn(x):
            y = sow(x, "val")
            z = sow(y**2, "val")
            return z.sum()

        with pytest.raises(ValueError, match="Duplicate sow name"):
            save_intermediates(maybe_jit(fn, use_jit))(jnp.array([1.0, 2.0]))


class TestSowTag:
    """save_intermediates respects the tag parameter."""

    @jit_parametrize
    def test_different_tags(self, use_jit):
        def fn(x):
            y = x**2
            y = sow(y, "squared", tag="debug")
            z = y + 1
            z = sow(z, "plus_one", tag="intermediates")
            return z.sum()

        fn_ = maybe_jit(fn, use_jit)

        # Only capture 'debug' tag
        _, intermediates = save_intermediates(fn_, tag="debug")(jnp.array([1.0, 2.0]))
        assert "squared" in intermediates
        assert "plus_one" not in intermediates

        # Only capture 'intermediates' tag
        _, intermediates = save_intermediates(fn_, tag="intermediates")(
            jnp.array([1.0, 2.0])
        )
        assert "squared" not in intermediates
        assert "plus_one" in intermediates


class TestSowIntegration:
    """Integration tests with apply_tesseract."""

    def test_pipeline_grad(self, served_univariate_tesseract_raw):
        tess = Tesseract(served_univariate_tesseract_raw)

        def pipeline(x, y):
            res = apply_tesseract(tess, {"x": x, "y": y})
            res = sow(res, "tess_output")
            return res["result"]

        x = jnp.array(1.0)
        y = jnp.array(2.0)

        # Forward only
        result, intermediates = save_intermediates(pipeline)(x, y)
        assert "tess_output" in intermediates
        assert "primal" in intermediates["tess_output"]
        np.testing.assert_array_equal(
            intermediates["tess_output"]["primal"]["result"], result
        )

        # With grad
        _, grad_ints = save_intermediates(jax.grad(pipeline))(x, y)
        assert "tess_output" in grad_ints
        assert "primal" in grad_ints["tess_output"]
        assert "cotangent" in grad_ints["tess_output"]

    def test_pipeline_forward_only(self, served_univariate_tesseract_raw):
        tess = Tesseract(served_univariate_tesseract_raw)

        def pipeline(x, y):
            res = apply_tesseract(tess, {"x": x, "y": y})
            res = sow(res, "output")
            return res["result"].sum()

        x = jnp.array(1.0)
        y = jnp.array(2.0)
        _, intermediates = save_intermediates(pipeline)(x, y)
        assert "output" in intermediates
        assert "primal" in intermediates["output"]
