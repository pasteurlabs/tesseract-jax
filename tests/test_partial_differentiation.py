# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that a partial derivative request is narrowed to the live sub-block.

When only some of a Tesseract's differentiable inputs or outputs are used,
``tesseract-jax`` narrows the ``jacobian`` / ``jacobian_vector_product`` /
``vector_jacobian_product`` request so the endpoint computes only the needed
columns and rows. Two mechanisms drive this: trace-time ``has_tangent`` filtering
(always applied) and dead-code elimination (only when JAX runs DCE, i.e. under
``jit`` and for un-jitted reverse mode).
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract


@pytest.mark.parametrize("use_jit", [False, True])
def test_jacfwd_partial_diff_restricts_jac_inputs(
    univariate_tess, use_jit, monkeypatch
):
    """``jacfwd`` wrt one of several diff inputs requests only that column.

    Input restriction is trace-time (``has_tangent`` filtering, not DCE), so it
    holds jitted and un-jitted alike.
    """
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x):
        # `y` is also schema-differentiable but JAX won't carry a tangent for it.
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    captured: dict[str, Any] = {}
    orig = univariate_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        captured["jac_inputs"] = list(jac_inputs)
        captured["jac_outputs"] = list(jac_outputs)
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(univariate_tess, "jacobian", spy)
    jac_fn = jax.jit(jax.jacfwd(f)) if use_jit else jax.jacfwd(f)
    g = jac_fn(x)

    np.testing.assert_allclose(g, -400.0, rtol=1e-5)
    assert captured["jac_inputs"] == ["x"], (
        f"expected only 'x' to be requested, got {captured['jac_inputs']}"
    )


@pytest.mark.parametrize("use_jit", [False, True])
def test_jacrev_partial_diff_restricts_jac_inputs(
    univariate_tess, use_jit, monkeypatch
):
    """``jacrev`` wrt one of several diff inputs requests only that column."""
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x):
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    captured: dict[str, Any] = {}
    orig = univariate_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        captured["jac_inputs"] = list(jac_inputs)
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(univariate_tess, "jacobian", spy)
    jac_fn = jax.jit(jax.jacrev(f)) if use_jit else jax.jacrev(f)
    g = jac_fn(x)

    np.testing.assert_allclose(g, -400.0, rtol=1e-5)
    assert captured["jac_inputs"] == ["x"]


def test_jacrev_partial_output_restricts_jac_outputs(
    pytree_tess, pytree_tess_inputs, monkeypatch
):
    """``jacrev`` of one of several diff outputs requests only that output's rows.

    On the batched VJP shortcut the unused outputs carry symbolic-zero
    cotangents. ``has_cotangent`` records that, so the ``jacobian`` request must
    drop them rather than materialize (and contract against zero) every row.
    """
    inputs = {k: jax.tree.map(jnp.asarray, v) for k, v in pytree_tess_inputs.items()}

    captured: dict[str, Any] = {}
    orig_jac = pytree_tess.jacobian
    orig_vjp = pytree_tess.vector_jacobian_product

    def spy_jac(*, inputs, jac_inputs, jac_outputs):
        captured["jac_outputs"] = sorted(jac_outputs)
        return orig_jac(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    def spy_vjp(*, inputs, vjp_inputs, vjp_outputs, cotangent_vector):
        captured["vjp_outputs"] = sorted(vjp_outputs)
        return orig_vjp(
            inputs=inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=vjp_outputs,
            cotangent_vector=cotangent_vector,
        )

    monkeypatch.setattr(pytree_tess, "jacobian", spy_jac)
    monkeypatch.setattr(pytree_tess, "vector_jacobian_product", spy_vjp)

    def f(x):
        i = {**inputs, "alpha": {**inputs["alpha"], "x": x}}
        # Only `result` enters. result_dict / result_list stay unused, so JAX
        # hands their cotangents in as symbolic zeros.
        return apply_tesseract(pytree_tess, i)["result"]

    x = inputs["alpha"]["x"]
    got = jax.jacrev(f)(x)

    # Reference via the sequential VJP path (no jacobian-materialization shortcut).
    def f_seq(x):
        i = {**inputs, "alpha": {**inputs["alpha"], "x": x}}
        return apply_tesseract(
            pytree_tess, i, materialize_jacobian=False, vmap_method="sequential"
        )["result"]

    expected = jax.jacrev(f_seq)(x)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    assert captured["jac_outputs"] == ["result"], (
        f"expected only 'result' rows to be requested, got {captured['jac_outputs']}"
    )
    # The sequential reference path prunes the same unused outputs.
    assert captured["vjp_outputs"] == ["result"], (
        f"expected only 'result' cotangents to be requested, got {captured['vjp_outputs']}"
    )


@pytest.mark.parametrize("use_jit", [False, True])
def test_grad_restricts_vjp_inputs(
    pytree_tess, pytree_tess_inputs, use_jit, monkeypatch
):
    """``grad`` wrt one of several diff inputs requests only that input's cotangent.

    Input restriction on the reverse path is trace-time (``has_tangent`` filtering,
    not DCE), so the ``vector_jacobian_product`` request drops the undifferentiated
    columns jitted and un-jitted alike. Uses the sequential VJP path so the endpoint
    called is ``vector_jacobian_product`` rather than the batched ``jacobian``
    shortcut.
    """
    inp = jax.tree.map(jnp.asarray, pytree_tess_inputs)
    x = inp["alpha"]["x"]

    def f(x):
        full = {**inp, "alpha": {**inp["alpha"], "x": x}}
        # `alpha.y`, `beta.*`, `delta[*]` are all schema-differentiable but only
        # `alpha.x` carries a tangent, so only its cotangent should be requested.
        return apply_tesseract(
            pytree_tess, full, materialize_jacobian=False, vmap_method="sequential"
        )["result"].sum()

    # Reference computed before the spy is installed, so only the spied call below
    # is captured. The narrowing is internal to tesseract-jax either way, so this
    # is a self-consistency check on the returned gradient.
    expected = jax.grad(f)(x)

    captured: dict[str, Any] = {}
    orig = pytree_tess.vector_jacobian_product

    def spy(*, inputs, vjp_inputs, vjp_outputs, cotangent_vector):
        captured["vjp_inputs"] = sorted(vjp_inputs)
        return orig(
            inputs=inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=vjp_outputs,
            cotangent_vector=cotangent_vector,
        )

    monkeypatch.setattr(pytree_tess, "vector_jacobian_product", spy)
    grad_fn = jax.jit(jax.grad(f)) if use_jit else jax.grad(f)
    g = grad_fn(x)

    np.testing.assert_allclose(g, expected, rtol=1e-5)
    assert captured["vjp_inputs"] == ["alpha.{x}"], (
        f"expected only 'alpha.{{x}}' to be requested, got {captured['vjp_inputs']}"
    )


@pytest.mark.parametrize("use_jit", [False, True])
def test_grad_discarding_input_prunes_vjp_via_dce(
    pytree_tess, pytree_tess_inputs, use_jit, monkeypatch
):
    """Discarding a gradient downstream drops its VJP column, but only via DCE.

    Here both differentiable inputs carry a tangent (both are traced grad
    arguments), so trace-time ``has_tangent`` filtering keeps both columns.
    Returning only ``alpha.x``'s gradient and discarding ``beta.z``'s leaves the
    latter a dead outvar of the ``vector_jacobian_product`` equation, which the DCE
    rule prunes. Like all DCE it fires only under ``jit``; un-jitted, the discard
    happens in Python after ``grad`` returns and the full request stands.
    """
    inp = jax.tree.map(jnp.asarray, pytree_tess_inputs)
    x = inp["alpha"]["x"]
    z = inp["beta"]["z"]

    def g(x, z):
        full = {
            **inp,
            "alpha": {**inp["alpha"], "x": x},
            "beta": {**inp["beta"], "z": z},
        }
        return apply_tesseract(
            pytree_tess, full, materialize_jacobian=False, vmap_method="sequential"
        )["result"].sum()

    def f(x, z):
        # grad wrt both, but only alpha.x's gradient is returned. beta.z's is dead.
        gx, _gz = jax.grad(g, argnums=(0, 1))(x, z)
        return gx

    expected = f(x, z)  # before the spy, so it is not captured

    captured: dict[str, Any] = {}
    orig = pytree_tess.vector_jacobian_product

    def spy(*, inputs, vjp_inputs, vjp_outputs, cotangent_vector):
        captured["vjp_inputs"] = sorted(vjp_inputs)
        return orig(
            inputs=inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=vjp_outputs,
            cotangent_vector=cotangent_vector,
        )

    monkeypatch.setattr(pytree_tess, "vector_jacobian_product", spy)
    fn = jax.jit(f) if use_jit else f
    got = fn(x, z)

    np.testing.assert_allclose(got, expected, rtol=1e-5)
    if use_jit:
        # DCE drops beta.z's dead gradient column from the request.
        assert captured["vjp_inputs"] == ["alpha.{x}"], (
            f"expected only 'alpha.{{x}}' to be requested, got {captured['vjp_inputs']}"
        )
    else:
        # No DCE eagerly: both columns are still requested, the discard is in Python.
        assert captured["vjp_inputs"] == ["alpha.{x}", "beta.z"], (
            f"expected both columns un-jitted, got {captured['vjp_inputs']}"
        )


def test_jitted_jvp_endpoint_restricts_jvp_outputs(
    pytree_tess, pytree_tess_inputs, monkeypatch
):
    """Under ``jit``, a direct ``jvp`` (no jacobian shortcut) prunes dead outputs.

    Forward-mode over a single tangent goes through the ``jacobian_vector_product``
    endpoint directly (not the batched ``jacobian`` shortcut). DCE — which only
    runs under ``jit`` — narrows the requested output tangents to the ones consumed
    downstream via ``live_output_paths``.
    """
    inp = jax.tree.map(jnp.asarray, pytree_tess_inputs)
    x = inp["alpha"]["x"]
    tangent = jnp.ones_like(x)

    def f(x):
        full = {**inp, "alpha": {**inp["alpha"], "x": x}}
        # The jvp is taken of the whole Tesseract output; only `result`'s tangent
        # is consumed, so DCE should drop the other outputs' tangents from the bind.
        out_tan = jax.jvp(
            lambda xx: apply_tesseract(
                pytree_tess,
                {**full, "alpha": {**full["alpha"], "x": xx}},
                materialize_jacobian=False,
                vmap_method="sequential",
            ),
            (x,),
            (tangent,),
        )[1]
        return out_tan["result"]

    ref = f(x)  # un-jitted reference (DCE off, but the `result` block is identical)

    captured: dict[str, Any] = {}
    orig = pytree_tess.jacobian_vector_product

    def spy(*, inputs, jvp_inputs, jvp_outputs, tangent_vector):
        captured["jvp_inputs"] = sorted(jvp_inputs)
        captured["jvp_outputs"] = sorted(jvp_outputs)
        return orig(
            inputs=inputs,
            jvp_inputs=jvp_inputs,
            jvp_outputs=jvp_outputs,
            tangent_vector=tangent_vector,
        )

    monkeypatch.setattr(pytree_tess, "jacobian_vector_product", spy)
    out = jax.jit(f)(x)

    np.testing.assert_allclose(out, ref, rtol=1e-5)
    assert captured["jvp_outputs"] == ["result"], (
        f"expected only 'result' output tangents, got {captured['jvp_outputs']}"
    )
    assert captured["jvp_inputs"] == ["alpha.{x}"], (
        f"expected only 'alpha.{{x}}' tangent, got {captured['jvp_inputs']}"
    )


def test_jacfwd_of_tangent_fn_restricts_jac_inputs(univariate_tess, monkeypatch):
    """``jacfwd`` of a linearized function wrt one argument narrows the request too.

    The tangent function's other argument gets a symbolic-zero tangent, which is
    instantiated to dense zeros before it can cross a bind. Its Jacobian column
    would then be fetched only to be multiplied by those zeros, so the JVP rule
    recomputes ``has_tangent`` for the tangent bind rather than inheriting it.
    """
    x = jnp.array(1.0, dtype="float64")
    y = jnp.array(2.0, dtype="float64")

    def f(x, y):
        return apply_tesseract(univariate_tess, dict(x=x, y=y))["result"]

    _primal, tangent_fn = jax.linearize(f, x, y)
    expected = jax.jacfwd(f, argnums=0)(x, y)  # before the spy, so it is not captured

    captured: dict[str, Any] = {}
    orig = univariate_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        captured["jac_inputs"] = list(jac_inputs)
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(univariate_tess, "jacobian", spy)
    g = jax.jacfwd(tangent_fn, argnums=0)(x, y)

    np.testing.assert_allclose(g, expected, rtol=1e-5)
    assert captured["jac_inputs"] == ["x"], (
        f"expected only 'x' to be requested, got {captured['jac_inputs']}"
    )


@pytest.mark.parametrize("materialize_jacobian", [None, False])
def test_jitted_jacfwd_partial_diff_restrictions(
    pytree_tess, pytree_tess_inputs, materialize_jacobian, monkeypatch
):
    """Under ``jit``, ``jacfwd`` requests only the live (input, output) sub-block.

    Differentiates one output (``result``) wrt one input (``alpha.x``) of a
    multi-in / multi-out Tesseract and checks the request is pruned on *both*
    forward code paths: the materialised ``jacobian`` endpoint
    (``materialize_jacobian=None``) and the per-tangent
    ``jacobian_vector_product`` endpoint (``materialize_jacobian=False``).

    Output restriction relies on DCE, which only runs under ``jit`` — hence the
    ``jax.jit`` wrapper here (cf. the un-jitted input tests above).
    """
    inp = jax.tree.map(jnp.asarray, pytree_tess_inputs)
    x = inp["alpha"]["x"]

    def f(x):
        full = {**inp, "alpha": {**inp["alpha"], "x": x}}
        return apply_tesseract(
            pytree_tess,
            full,
            vmap_method="sequential",
            materialize_jacobian=materialize_jacobian,
        )["result"]

    jac: dict[str, list] = {"inputs": [], "outputs": []}
    jvp: dict[str, list] = {"inputs": [], "outputs": []}
    orig_jac = pytree_tess.jacobian
    orig_jvp = pytree_tess.jacobian_vector_product

    def spy_jac(*, inputs, jac_inputs, jac_outputs):
        jac["inputs"].append(sorted(jac_inputs))
        jac["outputs"].append(sorted(jac_outputs))
        return orig_jac(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    def spy_jvp(*, inputs, jvp_inputs, jvp_outputs, tangent_vector):
        jvp["inputs"].append(sorted(jvp_inputs))
        jvp["outputs"].append(sorted(jvp_outputs))
        return orig_jvp(
            inputs=inputs,
            jvp_inputs=jvp_inputs,
            jvp_outputs=jvp_outputs,
            tangent_vector=tangent_vector,
        )

    # Un-jitted reference for the value check (DCE off -> full request, but the
    # selected `result` block is identical). Computed before the spies are set so
    # only the jitted, pruned call is captured below.
    ref = jax.jacfwd(f)(x)

    monkeypatch.setattr(pytree_tess, "jacobian", spy_jac)
    monkeypatch.setattr(pytree_tess, "jacobian_vector_product", spy_jvp)
    out = jax.jit(jax.jacfwd(f))(x)

    np.testing.assert_allclose(out, ref, rtol=1e-5)

    if materialize_jacobian is None:
        assert jvp["outputs"] == [], "materialised path must not call the jvp endpoint"
        assert jac["inputs"] == [["alpha.{x}"]]
        assert jac["outputs"] == [["result"]]
    else:
        assert jac["outputs"] == [], (
            "per-tangent path must not call the jacobian endpoint"
        )
        assert jvp["outputs"], "expected the jvp endpoint to be called"
        assert all(o == ["result"] for o in jvp["outputs"]), jvp["outputs"]
        assert all(i == ["alpha.{x}"] for i in jvp["inputs"]), jvp["inputs"]


@pytest.mark.parametrize("use_jit", [True, False])
def test_jacfwd_of_tangent_fn_prunes_like_jacfwd(
    pytree_tess, pytree_tess_inputs, use_jit, monkeypatch
):
    """Differentiating a linearized function requests only the live sub-block.

    The batching rule has to honour the equation's ``live_output_paths`` for this:
    it recomputes the requested rows from the schema, so without threading the
    pruning through it would request every differentiable row -- and, worse, report
    a different output count than ``abstract_eval`` does for the same bind.

    The pruning holds whether or not this is jitted, because ``jax.linearize``
    partially evaluates and bakes it into the tangent function's jaxpr before DCE
    is ever asked. That makes the linearized path *narrower* than ``jacfwd(f)``
    un-jitted, where forward-mode DCE does not run -- hence the assertion is on
    the requested block itself rather than parity with the reference.
    """
    inp = jax.tree.map(jnp.asarray, pytree_tess_inputs)
    alpha, delta = inp["alpha"], inp["delta"]

    def f(alpha, delta):
        # Consumes one of several differentiable outputs.
        return apply_tesseract(pytree_tess, {**inp, "alpha": alpha, "delta": delta})[
            "result"
        ]

    # One spy for both phases -- stacking two would make the first also record the
    # second's calls.
    seen: list[tuple] = []
    orig = pytree_tess.jacobian

    def spy(*, inputs, jac_inputs, jac_outputs):
        seen.append((tuple(sorted(jac_inputs)), tuple(sorted(jac_outputs))))
        return orig(inputs=inputs, jac_inputs=jac_inputs, jac_outputs=jac_outputs)

    monkeypatch.setattr(pytree_tess, "jacobian", spy)

    def maybe_jit(fn):
        return jax.jit(fn) if use_jit else fn

    expected = maybe_jit(jax.jacfwd(f, argnums=0))(alpha, delta)
    reference_requests = list(seen)
    seen.clear()

    _primal, tangent_fn = jax.linearize(f, alpha, delta)
    got = maybe_jit(jax.jacfwd(tangent_fn, argnums=0))(alpha, delta)
    linearized_requests = list(seen)

    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5), expected, got
    )
    # One jacobian call, restricted to the differentiated argument's columns and
    # the single consumed output's row. `delta` is not differentiated, so none of
    # its columns appear; `result_dict` / `result_list` are not consumed, so none
    # of their rows do.
    assert linearized_requests == [(("alpha.{x}", "alpha.{y}"), ("result",))], (
        f"linearized path requested {linearized_requests}"
    )
    # Under jit the reference prunes identically; un-jitted it cannot, so it asks
    # for every differentiable row.
    if use_jit:
        assert reference_requests == linearized_requests
    else:
        assert reference_requests != linearized_requests, (
            "expected un-jitted jacfwd(f) to request more rows than the "
            "linearized path, since forward-mode DCE does not run eagerly"
        )
