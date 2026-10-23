# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract, check_gradients, primitive

here = Path(__file__).parent


@pytest.fixture(scope="module")
def bug_tess() -> Tesseract:
    """Squares its input, with a VJP made wrong on purpose by ``bug``."""
    return Tesseract.from_tesseract_api(here / "vjp_bug_tesseract" / "tesseract_api.py")


def square(tess, x, bug="none", vmap_method=None, noise=0.0):
    inputs = {"x": x, "bug": bug, "noise": noise}
    return apply_tesseract(tess, inputs, vmap_method=vmap_method)["y"]


def rosenbrock(tess, p):
    return apply_tesseract(tess, {"x": p["x"], "y": p["y"], "a": 1.0, "b": 100.0})[
        "result"
    ]


PARAMS = {"x": jnp.array(0.3), "y": jnp.array(0.5)}


def test_correct_pipeline_passes(univariate_tess, bug_tess):
    @jax.jit
    def loss_fn(p):
        r = rosenbrock(univariate_tess, p)
        return jnp.sum(square(bug_tess, jnp.stack([r, p["x"]])))

    # A leaf the loss ignores has a zero gradient, which is not a problem.
    report = check_gradients(loss_fn, {**PARAMS, "unused": jnp.array(1.0)})
    assert report.passed, report
    assert len(report.calls) == 2
    assert report.calls[0].tesseract is univariate_tess
    assert report.calls[1].tesseract is bug_tess
    assert not any(c.problems for c in report.calls)
    assert report.verdict == "Gradients agree."


@pytest.mark.parametrize("use_jit", [True, False], ids=["jit", "eager"])
def test_wrong_vjp_is_named(univariate_tess, bug_tess, use_jit):
    def loss_fn(p):
        r = rosenbrock(univariate_tess, p)
        return jnp.sum(square(bug_tess, jnp.stack([r, p["x"]]), bug="scale"))

    report = check_gradients(jax.jit(loss_fn) if use_jit else loss_fn, PARAMS)
    assert not report.pipeline.passed
    assert [c.passed for c in report.calls] == [True, False]
    assert report.verdict.startswith(report.calls[1].name)
    assert "Start there" in report.verdict


@pytest.mark.parametrize(
    "bug, noise, problems",
    [
        # A NaN or inf gradient is wrong however noisy the finite difference is.
        ("nan", 0.0, ("NaN in gradient of ['x']",)),
        ("nan", 1e-2, ("NaN in gradient of ['x']",)),
        ("inf", 0.0, ("inf in gradient of ['x']",)),
        ("inf", 1e-2, ("inf in gradient of ['x']",)),
        ("zero", 0.0, ()),
        # apply returns zeros, so the finite difference is exactly zero.
        ("detached", 0.0, ()),
    ],
)
def test_bad_vjp_flagged(bug_tess, bug, noise, problems):
    report = check_gradients(
        lambda x: jnp.sum(square(bug_tess, x, bug=bug, noise=noise)),
        jnp.array([0.5, 1.5]),
    )
    assert not report.pipeline.passed
    (call,) = report.calls
    assert not call.passed
    assert call.problems == problems
    assert report.verdict.endswith("Start there.")


def test_wrong_vjp_named_when_finite_difference_does_not_settle(bug_tess):
    report = check_gradients(
        lambda x: jnp.sum(square(bug_tess, x, bug="scale", noise=1e-2)),
        jnp.array([0.5, 1.5]),
    )
    (call,) = report.calls
    assert call.problems == ("finite difference too uncertain",)
    assert report.verdict.startswith(call.name)
    assert "Start there. Its finite difference is too uncertain" in report.verdict


def test_raising_vjp_is_reported_on_its_row(univariate_tess, bug_tess):
    # The first call's VJP raises; the second call is still checked.
    def loss_fn(p):
        x = jnp.sum(square(bug_tess, p["x"], bug="raise"))
        return rosenbrock(univariate_tess, {"x": x, "y": p["y"]})

    report = check_gradients(loss_fn, PARAMS)
    first, second = report.calls
    assert "adjoint solve failed" in first.error
    assert second.passed
    assert "ERROR" in str(report)
    assert report.verdict.startswith(f"{first.name} raised")


def test_vjp_ignoring_cotangent_fails(bug_tess):
    # The loss is a plain sum, so its cotangent is all ones and the bug does
    # not reach it; only the random cotangent of the call's own check does.
    report = check_gradients(
        lambda x: jnp.sum(square(bug_tess, x, bug="ones")), jnp.asarray([0.5, 1.5])
    )
    assert report.pipeline.passed
    (call,) = report.calls
    assert not call.passed
    assert "try another seed" in report.verdict


def test_constant_inputs_held_fixed(univariate_tess, monkeypatch):
    seen = []
    apply = univariate_tess.apply

    def spy(inputs):
        seen.append(float(inputs["y"]))
        return apply(inputs)

    monkeypatch.setattr(univariate_tess, "apply", spy)
    report = check_gradients(
        lambda x: rosenbrock(univariate_tess, {"x": x, "y": 0.5}), jnp.array(0.3)
    )
    assert report.passed, report
    assert set(seen) == {0.5}


def _half_stop_gradient(y):
    return 0.5 * y + 0.5 * jax.lax.stop_gradient(y)


def _scaled_identity(scale):
    @jax.custom_vjp
    def identity(y):
        return y

    identity.defvjp(lambda y: (y, None), lambda _, g: (scale * g,))
    return identity


@pytest.mark.parametrize(
    "glue, seed",
    [
        (_half_stop_gradient, 0),
        (_scaled_identity(1.5), 0),
        (_scaled_identity(1.05), 16),
    ],
    ids=["stop_gradient", "custom_vjp", "custom_vjp_5pct"],
)
def test_blames_code_between_tesseracts(univariate_tess, bug_tess, glue, seed):
    # At seed 16 the curvature of the loss makes the finite difference at the
    # coarsest step about 5% high, so it alone would match the wrong gradient.
    def loss_fn(p):
        r = rosenbrock(univariate_tess, p)
        return jnp.sum(square(bug_tess, glue(r)))

    report = check_gradients(loss_fn, PARAMS, seed=seed)
    assert not report.pipeline.passed
    assert len(report.calls) == 2
    assert all(c.passed for c in report.calls)
    assert "JAX code between the Tesseracts" in report.verdict


def test_near_a_minimum_blames_nothing(bug_tess):
    # Correct, but the gradient is tiny next to the curvature of the loss, so
    # the finite difference keeps shrinking with the step.
    def loss_fn(x):
        y = square(bug_tess, x)
        return jnp.sum((y - 1) ** 2) + jnp.sum((y - 3) ** 2)

    report = check_gradients(loss_fn, jnp.full(3, np.sqrt(2) + 1e-8))
    assert not report.pipeline.passed
    assert report.pipeline.problems == ("finite difference too uncertain",)
    assert "cannot tell" in report.verdict
    assert "Start there" not in report.verdict
    assert "JAX code between" not in report.verdict


def test_glue_not_blamed_when_a_tesseract_is_nearly_as_wrong(univariate_tess, bug_tess):
    # At rtol=0.06 the Tesseract's 5% error passes, but it is half of what the
    # pipeline is off by, so the code between them cannot be blamed for sure.
    def loss_fn(p):
        r = rosenbrock(univariate_tess, p)
        return jnp.sum(square(bug_tess, _scaled_identity(1.05)(r), bug="scale"))

    report = check_gradients(loss_fn, PARAMS, rtol=0.06)
    assert not report.pipeline.passed
    assert all(c.passed for c in report.calls)
    assert "cannot tell" in report.verdict
    assert report.calls[1].name in report.verdict


@pytest.mark.parametrize(
    "loss_fn",
    [
        # The loss is 1e8 plus changes far below the rounding of float32.
        lambda t, x: apply_tesseract(t, {"a": x, "z": x})["result"] + 1e8,
        # The Tesseract's output is: z is a constant much larger than a.
        lambda t, x: apply_tesseract(t, {"a": x, "z": jnp.full_like(x, 1e4)})["result"],
    ],
    ids=["pipeline", "call"],
)
def test_float32_rounding_blames_nothing(static_input_tess, loss_fn):
    x = jnp.array([0.5, 1.0, 1.5], jnp.float32)
    for seed in range(3):
        report = check_gradients(lambda v: loss_fn(static_input_tess, v), x, seed=seed)
        assert not report.passed
        assert "JAX code between" not in report.verdict
        assert "may still be right" in report.verdict or "cannot tell" in report.verdict


@pytest.mark.parametrize("noise, seed", [(1e-4, 5), (3e-4, 8)])
def test_noise_blames_nothing(bug_tess, noise, seed):
    # Noise of about rtol at the finer steps can make neighbouring finite
    # differences agree with each other, or drift together, away from the
    # correct gradient.
    report = check_gradients(
        lambda x: jnp.sum(jnp.sin(square(bug_tess, x, noise=noise))),
        jnp.array([0.5, 1.5, 0.7, 1.1]),
        seed=seed,
    )
    assert "JAX code between" not in report.verdict
    if "Start there" in report.verdict:
        assert "may still be right" in report.verdict


@pytest.mark.parametrize(
    "loss_fn, x",
    [
        # Inputs near 1e-4: a step is only small if it is relative to them.
        (lambda t, x: jnp.sum(square(t, square(t, x))), [1e-4, 2e-4, 3e-4]),
        # A large constant: fine steps lose to rounding, coarse ones do not.
        (lambda t, x: jnp.sum(square(t, x)) + 1e11, [0.1, 0.2, 0.3]),
    ],
    ids=["small_inputs", "large_constant"],
)
def test_step_size(bug_tess, loss_fn, x):
    report = check_gradients(lambda v: loss_fn(bug_tess, v), jnp.array(x))
    assert report.passed, report


def test_nan_step_not_kept(bug_tess):
    # The coarse step leaves the domain of sqrt.
    report = check_gradients(
        lambda x: jnp.sum(square(bug_tess, jnp.sqrt(x))),
        jnp.array([1.0, 2.0, 3.0]),
        steps=(1e3, 1e-2, 1e-3, 1e-4),
    )
    assert np.isnan(report.pipeline.finite_differences[0])
    assert report.pipeline.step == 1e-3
    assert report.passed, report


@pytest.mark.parametrize(
    "steps",
    [
        (1e-1, 1e-2),
        (0.0, 1e-1, 1e-2),
        (-1e-3, 1e-1, 1e-2),
        (np.inf, 1, 0.1),
        (np.nan,) * 3,
    ],
)
def test_rejects_bad_steps(steps):
    with pytest.raises(ValueError, match="steps must be"):
        check_gradients(jnp.sum, jnp.ones(2), steps=steps)


def test_rejects_zero_max_calls_per_site():
    with pytest.raises(ValueError, match="max_calls_per_site"):
        check_gradients(jnp.sum, jnp.ones(2), max_calls_per_site=0)


def test_seed_sets_every_direction(univariate_tess, bug_tess):
    def loss_fn(p):
        r = rosenbrock(univariate_tess, p)
        return jnp.sum(square(bug_tess, jnp.stack([r, p["x"]])))

    def projections(seed):
        report = check_gradients(loss_fn, PARAMS, seed=seed)
        return [c.grad_dot_d for c in (report.pipeline, *report.calls)]

    first = projections(3)
    assert projections(3) == first
    assert all(a != b for a, b in zip(first, projections(4), strict=True))


def test_scan_checks_first_call_and_says_so(bug_tess):
    # Every value is below 1 at the first iteration only, so only the later
    # calls hit the bug.
    @jax.jit
    def loss_fn(x):
        step = lambda c, _: (square(bug_tess, c, bug="late") + 1, None)
        return jnp.sum(jax.lax.scan(step, x, length=4)[0])

    x = jnp.array([0.1, 0.2, 0.3])
    report = check_gradients(loss_fn, x)
    (call,) = report.calls
    assert call.n_calls == 4
    assert call.passed
    assert not report.pipeline.passed
    assert "max_calls_per_site=None" in report.verdict
    assert "The problem is in the JAX code" not in report.verdict

    report = check_gradients(loss_fn, x, max_calls_per_site=None)
    assert len(report.calls) == 4
    assert [c.call for c in report.calls] == [1, 2, 3, 4]
    np.testing.assert_allclose(report.calls[1].inputs["x"], x**2 + 1)
    assert "(call 2)" in report.verdict


X = np.array([0.1, 0.2, 0.3])


def plain(tess, x):
    return jnp.sum(square(tess, x))


def while_loop(tess, x):
    def body(carry):
        v, i = carry
        return square(tess, v) + 1, i + 1

    v, _ = jax.lax.while_loop(lambda c: c[1] < 3, body, (x, 0))
    return jnp.sum(v)


def cond(tess, x):
    return jnp.sum(jax.lax.cond(x[0] < 1, lambda v: square(tess, v), jnp.sin, x))


def vmap(tess, x, vmap_method):
    batched = jax.vmap(lambda r: square(tess, r, vmap_method=vmap_method))
    return jnp.sum(batched(jnp.stack([x, 2 * x])))


def checkpoint(tess, x):
    # The backward pass needs the output, so it runs the apply again.
    return jnp.sum(jax.checkpoint(lambda v: square(tess, v) ** 2)(x))


@pytest.mark.parametrize(
    "pipeline, n_calls, first_x",
    [
        pytest.param(plain, 1, X, id="jit"),
        pytest.param(while_loop, 3, X, id="while_loop"),
        pytest.param(cond, 1, X, id="cond"),
        pytest.param(partial(vmap, vmap_method="sequential"), 2, X, id="vmap_seq"),
        pytest.param(
            partial(vmap, vmap_method="expand_dims"),
            1,
            np.stack([X, 2 * X]),
            id="vmap_expand_dims",
        ),
        pytest.param(checkpoint, 1, X, id="checkpoint"),
    ],
)
def test_records_under_transformations(bug_tess, pipeline, n_calls, first_x):
    report = check_gradients(jax.jit(lambda x: pipeline(bug_tess, x)), jnp.asarray(X))
    (call,) = report.calls
    assert call.n_calls == n_calls
    np.testing.assert_array_equal(call.inputs["x"], first_x)
    assert call.passed


def test_cond_branch_not_taken_records_nothing(bug_tess):
    @jax.jit
    def loss_fn(x):
        return jnp.sum(
            jax.lax.cond(x[0] > 1, lambda v: square(bug_tess, v), jnp.sin, x)
        )

    report = check_gradients(loss_fn, jnp.asarray(X))
    assert report.calls == ()
    assert report.verdict.startswith("No Tesseract was called")


def test_records_jit_compiled_before_check(bug_tess):
    loss_fn = jax.jit(lambda x: jnp.sum(square(bug_tess, x)))
    jax.value_and_grad(loss_fn)(jnp.asarray(X))
    report = check_gradients(loss_fn, jnp.asarray(X))
    (call,) = report.calls
    np.testing.assert_array_equal(call.inputs["x"], X)


def test_repeated_identical_call_site(bug_tess):
    loss_fn = jax.jit(lambda x: jnp.sum(square(bug_tess, square(bug_tess, x))))
    report = check_gradients(loss_fn, jnp.asarray(X))
    (call,) = report.calls
    assert call.n_calls == 2

    report = check_gradients(loss_fn, jnp.asarray(X), max_calls_per_site=2)
    first, second = report.calls
    np.testing.assert_array_equal(first.inputs["x"], X)
    np.testing.assert_allclose(second.inputs["x"], X**2)
    assert report.passed, report


def test_unchecked_calls_blame_nothing(bug_tess):
    # The glue between two calls of one site is wrong. With only the first
    # call checked, a later call could be the culprit as well.
    def loss_fn(x):
        return jnp.sum(square(bug_tess, _half_stop_gradient(square(bug_tess, x))))

    report = check_gradients(loss_fn, jnp.asarray(X))
    assert not report.pipeline.passed
    assert "Not every call was checked: #1 x -> y (1 of 2 calls)" in report.verdict

    report = check_gradients(loss_fn, jnp.asarray(X), max_calls_per_site=None)
    assert all(c.passed for c in report.calls)
    assert "The problem is in the JAX code" in report.verdict


def test_mixed_inputs_pass(
    static_input_tess, pytree_tess, pytree_tess_inputs, nonarray_output_tess
):
    def loss_fn(p):
        r = apply_tesseract(
            static_input_tess, {"a": p["a"], "scale": 2, "z": p["z"] * p["n"]}
        )["result"]
        inputs = {
            **pytree_tess_inputs,
            "alpha": {"x": r * p["a"], "y": pytree_tess_inputs["alpha"]["y"]},
        }
        out = apply_tesseract(pytree_tess, inputs)
        # Its outputs include a str and a bool.
        y = apply_tesseract(nonarray_output_tess, {"x": p["a"].astype("float64")})["y"]
        return jnp.sum(out["result"]) + jnp.sum(out["result_list"][0]) + jnp.sum(y)

    params = {
        "a": np.array([0.5, 1.0, 1.5], dtype="float32"),
        "z": np.array([2.0, 1.0, 0.5], dtype="float32"),
        "n": 3,
    }
    report = check_gradients(loss_fn, params)
    assert report.passed, report
    assert len(report.calls) == 3
    np.testing.assert_array_equal(
        report.calls[1].inputs["epsilon"]["k"], pytree_tess_inputs["epsilon"]["k"]
    )


def test_float32_tesseract_under_strict_promotion(static_input_tess):
    x = np.array([0.5, 1.0, 1.5], dtype="float32")
    with jax.numpy_dtype_promotion("strict"):
        report = check_gradients(
            lambda x: apply_tesseract(static_input_tess, {"a": x, "z": x})["result"], x
        )
    assert report.passed, report


def _raises(t, x):
    square(t, x)
    raise ValueError("boom")


@pytest.mark.parametrize(
    "loss_fn, error",
    [
        (_raises, "ValueError: boom"),
        (square, "TypeError: loss_fn must return a real scalar"),
    ],
    ids=["raises", "not_scalar"],
)
def test_loss_error_is_reported(bug_tess, loss_fn, error):
    report = check_gradients(lambda x: loss_fn(bug_tess, x), jnp.asarray(X))
    assert report.pipeline.error.startswith(error)
    (call,) = report.calls
    assert call.passed
    assert "could not be computed" in report.verdict
    assert primitive._apply_recorder is None


def test_untraceable_grad_still_records(univariate_tess):
    # When the gradient cannot even be traced, a plain evaluation still records
    # the calls.
    @jax.jit
    def loss_fn(p):
        y = jax.pure_callback(np.copy, jax.ShapeDtypeStruct((), p["y"].dtype), p["y"])
        return rosenbrock(univariate_tess, {"x": p["x"], "y": y})

    report = check_gradients(loss_fn, PARAMS)
    assert "Pure callbacks do not support JVP" in report.pipeline.error
    (call,) = report.calls
    assert call.passed
    assert "could not be computed" in report.verdict
    assert primitive._apply_recorder is None


def test_constants_held_fixed_when_grad_cannot_be_traced(univariate_tess, monkeypatch):
    # Without tangents, which inputs depend on params comes from comparing two
    # plain evaluations; y is a constant and must not be perturbed.
    seen = []
    apply = univariate_tess.apply

    def spy(inputs):
        seen.append(float(inputs["y"]))
        return apply(inputs)

    monkeypatch.setattr(univariate_tess, "apply", spy)

    @jax.jit
    def loss_fn(x):
        x = jax.pure_callback(np.copy, jax.ShapeDtypeStruct((), x.dtype), x)
        return rosenbrock(univariate_tess, {"x": x, "y": jnp.array(0.5)})

    report = check_gradients(loss_fn, jnp.array(0.3))
    assert "Pure callbacks do not support JVP" in report.pipeline.error
    (call,) = report.calls
    assert call.passed
    assert set(seen) == {0.5}


def test_not_reentrant(bug_tess):
    def nested(x):
        check_gradients(lambda v: jnp.sum(square(bug_tess, v)), x)
        return jnp.sum(square(bug_tess, x))

    report = check_gradients(nested, jnp.asarray(X))
    assert report.pipeline.error == "RuntimeError: check_gradients is not re-entrant"
    assert primitive._apply_recorder is None


def test_report_prints(univariate_tess, bug_tess):
    def loss_fn(p):
        r = rosenbrock(univariate_tess, p)
        return jnp.sum(square(bug_tess, r, bug="scale"))

    steps = (1e-1, 1e-2, 1e-3)
    report = check_gradients(loss_fn, PARAMS, steps=steps)
    text = str(report)
    assert repr(report) == text
    assert "pipeline" in text
    assert report.calls[1].name == "#2 x -> y"
    assert "#2 x -> y" in text
    assert "FAIL" in text
    assert text.endswith(report.verdict)
    for row in (report.pipeline, *report.calls):
        assert len(row.finite_differences) == len(steps)
