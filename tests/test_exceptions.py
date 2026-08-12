# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failures must stay legible, including from inside a jitted callback.

A Tesseract can fail in four distinguishable ways, and only the first is
knowable at trace time:

1. abstract (shape / dtype) validation -- caught before anything is dispatched
2. value-based *input* validation      -- only knowable once values exist
3. value-based *output* validation     -- the endpoint returned something invalid
4. an exception raised inside the endpoint

Cases 2-4 surface from inside the lowered callback, which is the interesting
part: the original exception and message must still reach the caller rather than
being swallowed or crashing the process. All four behave identically whether the
callback is lowered as side-effecting or pure.

Exception *types* are normalised twice on the way out, so these tests pin what a
caller actually sees rather than what the endpoint raised:

* tesseract-core wraps whatever the endpoint raised in ``RuntimeError``
* under ``jit``, anything raised inside the callback arrives as
  ``jax.errors.JaxRuntimeError``

Case 1 escapes both, because it fails while tracing rather than in the callback.
The *message* survives intact either way.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from tesseract_jax import apply_tesseract

V = jnp.zeros(3, dtype="float64")


def _expected_error(use_jit):
    """What a caller sees when the endpoint raises."""
    return jax.errors.JaxRuntimeError if use_jit else RuntimeError


def _count_apply(tess, monkeypatch):
    """Count apply endpoint invocations."""
    calls = {"n": 0}
    orig = tess.apply

    def spy(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(tess, "apply", spy)
    return calls


def test_abstract_validation_fails_before_dispatch(validating_tess, monkeypatch):
    """A shape mismatch is caught at trace time and never reaches the endpoint."""
    calls = _count_apply(validating_tess, monkeypatch)

    @jax.jit
    def f(v):
        return apply_tesseract(validating_tess, dict(x=1.0, v=v))["result"]

    with pytest.raises(ValidationError, match=r"Expected shape: \(3,\)"):
        f(jnp.zeros(4, dtype="float64"))

    assert calls["n"] == 0, "endpoint should never be reached"


@pytest.mark.parametrize("use_jit", [True, False])
def test_value_based_input_validation_reaches_caller(validating_tess, use_jit):
    """`x > 0` depends on the value, so it can only fail at run time."""

    def f(x):
        return apply_tesseract(validating_tess, dict(x=x, v=V))["result"]

    if use_jit:
        f = jax.jit(f)

    with pytest.raises(
        _expected_error(use_jit), match=r"x must be strictly positive, got -1\.0"
    ):
        jax.block_until_ready(f(jnp.array(-1.0, dtype="float64")))


@pytest.mark.parametrize("use_jit", [True, False])
def test_value_based_output_validation_reaches_caller(validating_tess, use_jit):
    """The endpoint returns valid Python that violates OutputSchema."""

    def f(x):
        return apply_tesseract(validating_tess, dict(x=x, v=V, mode="bad_output"))[
            "result"
        ]

    if use_jit:
        f = jax.jit(f)

    with pytest.raises(
        _expected_error(use_jit), match=r"(?s)OutputSchema.*non-numeric"
    ):
        jax.block_until_ready(f(jnp.array(4.0, dtype="float64")))


@pytest.mark.parametrize("use_jit", [True, False])
def test_exception_inside_endpoint_reaches_caller(validating_tess, use_jit):
    """An exception raised in the endpoint body arrives with its message."""

    def f(x):
        return apply_tesseract(validating_tess, dict(x=x, v=V, mode="raise"))["result"]

    if use_jit:
        f = jax.jit(f)

    with pytest.raises(
        _expected_error(use_jit), match=r"deliberate failure inside the apply endpoint"
    ):
        jax.block_until_ready(f(jnp.array(4.0, dtype="float64")))


def test_still_usable_after_a_failure(validating_tess):
    """A failure is an ordinary Python exception and does not wedge the runtime.

    Catching it needs nothing special: a plain ``try``/``except`` on
    ``JaxRuntimeError`` works, the message is on the exception, and the very next
    call succeeds.
    """

    @jax.jit
    def f(x):
        return apply_tesseract(validating_tess, dict(x=x, v=V))["result"]

    bad = jnp.array(-1.0, dtype="float64")
    good = jnp.array(4.0, dtype="float64")

    # plain try/except, as a caller would write it
    caught = None
    try:
        jax.block_until_ready(f(bad))
    except jax.errors.JaxRuntimeError as exc:
        caught = exc
    assert caught is not None, "the failure must propagate to the caller"
    assert "x must be strictly positive, got -1.0" in str(caught)

    # ...and the runtime is unharmed
    np.testing.assert_allclose(f(good), 2.0, rtol=1e-6)

    # the same failure is equally catchable via pytest.raises, and recovering a
    # second time still works
    with pytest.raises(
        jax.errors.JaxRuntimeError, match=r"x must be strictly positive"
    ):
        jax.block_until_ready(f(bad))
    np.testing.assert_allclose(f(good), 2.0, rtol=1e-6)


def test_guarded_call_is_elided_on_a_constant_predicate(validating_tess, monkeypatch):
    """A guard XLA can fold spares the endpoint an invalid call.

    Since the callback is lowered as pure, XLA may drop a branch whose predicate
    it can evaluate at compile time -- so a caller who guards against invalid
    input gets their guard honoured. With a traced predicate the guard cannot be
    folded and the endpoint is still called, which is the pre-existing behaviour.
    """
    x = jnp.array(-1.0, dtype="float64")  # invalid for this Tesseract
    calls = _count_apply(validating_tess, monkeypatch)

    @jax.jit
    def traced_predicate(x):
        return jnp.where(
            x > 0, apply_tesseract(validating_tess, dict(x=x, v=V))["result"], 0.0
        )

    with pytest.raises(
        jax.errors.JaxRuntimeError, match=r"x must be strictly positive"
    ):
        jax.block_until_ready(traced_predicate(x))
    assert calls["n"] == 1, "a traced guard cannot be folded, so the call happens"

    calls["n"] = 0

    @jax.jit
    def constant_predicate():
        # `x` is closed over as a concrete value, so `x > 0` is a compile-time
        # constant and the branch is dead.
        return jnp.where(
            x > 0, apply_tesseract(validating_tess, dict(x=x, v=V))["result"], 0.0
        )

    np.testing.assert_allclose(constant_predicate(), 0.0, rtol=1e-6)
    assert calls["n"] == 0, "a foldable guard should spare the endpoint entirely"
