# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``to_traceable`` beyond the ``vectoradd_jax`` parity tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract, to_traceable

# Exact host-callback lowering target (jax._src.callback.emit_python_callback).
# More specific than "custom-call" so an unrelated XLA custom call (e.g. a
# LAPACK op) in the Tesseract's own computation can't cause a false failure.
_CALLBACK_TARGET = "xla_ffi_python_cpu_callback"


def _maybe_traced(tess, traceable: bool):
    """``tess``, or its ``to_traceable`` view when ``traceable``."""
    return to_traceable(tess) if traceable else tess


def test_traceable_requires_in_process_tesseract(served_univariate_tesseract):
    """A served (HTTPClient) Tesseract has no importable Python function to trace."""
    with pytest.raises(ValueError, match="in-process Tesseract"):
        to_traceable(served_univariate_tesseract)


def test_traceable_works_with_an_in_process_gpu_transport_tesseract(vectoradd_jax_ab):
    """``to_traceable`` accepts an in-process Tesseract that requests a GPU transport.

    The transport is only resolved when compiling for CUDA, which this test
    does not cover.
    """
    from tesseract_core import Tesseract

    tess = to_traceable(
        Tesseract.from_tesseract_api(
            "examples/simple/vectoradd_jax/tesseract_api.py", gpu_transport="cuda_ipc"
        )
    )
    out = apply_tesseract(tess, {**vectoradd_jax_ab, "norm_ord": 2})
    np.testing.assert_allclose(out["vector_add"]["result"], [4.0, 6.5, 9.0], rtol=1e-6)


def test_traceable_apply_has_no_host_callback_in_lowered_hlo(
    vectoradd_apply, vectoradd_jax_ab
):
    """A traced ``apply`` lowers without a host callback.

    Checks the uncompiled lowering, which is fast and backend-independent.
    """

    def f(traceable):
        return vectoradd_apply(vectoradd_jax_ab, traceable)["vector_add"][
            "result"
        ].sum()

    direct_hlo = jax.jit(lambda: f(True)).lower().as_text()
    callback_hlo = jax.jit(lambda: f(False)).lower().as_text()

    assert _CALLBACK_TARGET not in direct_hlo
    # Positive control: confirms the string is actually present somewhere.
    assert _CALLBACK_TARGET in callback_hlo


def test_traceable_jvp_has_no_host_callback_in_lowered_hlo(
    vectoradd_apply, vectoradd_jax_ab
):
    """A traced JVP lowers without a host callback."""
    tangents = jax.tree.map(jnp.ones_like, vectoradd_jax_ab)

    def f(traceable):
        full = lambda ab: vectoradd_apply(ab, traceable)
        return jax.jvp(full, (vectoradd_jax_ab,), (tangents,))[1]["vector_add"][
            "result"
        ]

    direct_hlo = jax.jit(lambda: f(True)).lower().as_text()
    callback_hlo = jax.jit(lambda: f(False)).lower().as_text()

    assert _CALLBACK_TARGET not in direct_hlo
    assert _CALLBACK_TARGET in callback_hlo


@pytest.mark.parametrize("traceable", [False, True])
def test_input_dtype_is_cast_to_schema_dtype(
    vectoradd_apply, vectoradd_jax_ab, traceable
):
    """A float64 input is cast to the schema's float32 on both dispatch paths.

    Without the cast, the traced path returns float64, which fails MLIR's
    output-type check against ``abstract_eval``'s float32.
    """
    ab_f64 = jax.tree.map(lambda x: x.astype(jnp.float64), vectoradd_jax_ab)
    out = vectoradd_apply(ab_f64, traceable)
    assert out["vector_add"]["result"].dtype == jnp.float32


def test_unsafe_cast_is_left_alone():
    """A narrowing int64 -> int32 cast, which could wrap the value, is not applied."""
    from tesseract_jax.direct_trace import _safe_cast_dtype

    out_of_range = jnp.asarray(2**33, dtype=jnp.int64)
    result = _safe_cast_dtype(out_of_range, "int32")
    assert result.dtype == jnp.int64
    assert int(result) == 2**33


@pytest.mark.parametrize("traceable", [False, True])
def test_static_leaves_survive_traced_dispatch(nonarray_output_tess, traceable):
    """Non-array (str/bool) outputs come back unchanged on both dispatch paths."""
    tess = _maybe_traced(nonarray_output_tess, traceable)
    x = jnp.arange(3, dtype=jnp.float64)

    def loss(x):
        return apply_tesseract(tess, dict(x=x))["y"].sum()

    out = apply_tesseract(tess, dict(x=x))
    assert out["backend"] == "reference"
    assert out["converged"] is True
    np.testing.assert_allclose(out["y"], 2.0 * x)
    np.testing.assert_allclose(jax.grad(loss)(x), 2.0 * jnp.ones_like(x))


@pytest.mark.parametrize("traceable", [False, True])
def test_symbolic_zero_cotangent_matches_on_traced_dispatch(
    zero_cotangent_tess, traceable
):
    """An unused output's symbolic-zero cotangent is dropped on both dispatch paths.

    ``unsafe`` has a NaN gradient at x = 0, so using its cotangent would make
    the gradient NaN.
    """
    tess = _maybe_traced(zero_cotangent_tess, traceable)
    x = jnp.zeros(3, dtype=jnp.float64)

    def loss(x):
        return apply_tesseract(tess, dict(x=x))["safe"].sum()

    grad = jax.grad(loss)(x)
    np.testing.assert_array_equal(grad, [2.0, 2.0, 2.0])


@pytest.mark.parametrize("traceable", [False, True])
def test_gather_discarded_slots_match_on_traced_dispatch(gather_tess, traceable):
    """Discarded int/float/complex output tangents match on both dispatch paths."""
    tess = _maybe_traced(gather_tess, traceable)
    weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 2], dtype=jnp.int32)

    def apply_fn(w):
        return apply_tesseract(tess, dict(weights=w, indices=indices))

    _, tangents = jax.jvp(apply_fn, (weights,), (jnp.ones_like(weights),))
    magnitude = np.asarray(tangents["magnitude"])
    assert magnitude.dtype == np.float32
    assert np.isnan(magnitude).all()

    phase = np.asarray(tangents["phase"])
    assert phase.dtype == np.complex64
    assert np.isnan(phase.real).all()

    count = np.asarray(tangents["count"])
    assert count.dtype == jax.dtypes.float0
    assert count.shape == (3,)

    grad = jax.grad(lambda w: apply_fn(w)["gathered"].sum())(weights)
    np.testing.assert_allclose(grad, [1.0, 0.0, 2.0])


def test_traceable_requires_a_dict_returning_endpoint(mixed_dtype_tess):
    """An endpoint that returns ``OutputSchema(...)`` raises a tracer error."""
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    with pytest.raises(Exception, match=r"TracerArrayConversionError|Tracer"):
        apply_tesseract(to_traceable(mixed_dtype_tess), dict(x=x))


def test_omitted_array_field_recovers_its_real_default(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    """An omitted array field resolves to its real default, not a ``ShapeDType`` placeholder."""
    from tesseract_jax.direct_trace import (
        _abstract_input_schema,
        _patch_with_real_values,
    )
    from tesseract_jax.tree_util import to_shape_dtype_pytree

    api_module = vectoradd_jax_tess._client.api_module
    AbstractInputSchema = _abstract_input_schema(
        api_module.InputSchema, api_module.OutputSchema
    )

    a_no_s = {"v": vectoradd_jax_ab["a"]["v"]}  # omit "s"
    real_inputs_omitted = {"a": a_no_s, "b": vectoradd_jax_ab["b"], "norm_ord": 2}

    abstract_inputs = to_shape_dtype_pytree(real_inputs_omitted)
    abstract_instance = AbstractInputSchema.model_validate({"inputs": abstract_inputs})
    patched = _patch_with_real_values(
        abstract_instance.inputs, real_inputs_omitted, api_module.InputSchema
    )
    assert float(patched.a.s) == pytest.approx(1.0)


def test_omitted_array_field_raises_without_a_resolvable_real_class(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    """Without ``real_cls``, an omitted array field raises instead of using a placeholder."""
    from tesseract_jax.direct_trace import (
        _abstract_input_schema,
        _patch_with_real_values,
    )
    from tesseract_jax.tree_util import to_shape_dtype_pytree

    api_module = vectoradd_jax_tess._client.api_module
    AbstractInputSchema = _abstract_input_schema(
        api_module.InputSchema, api_module.OutputSchema
    )

    a_no_s = {"v": vectoradd_jax_ab["a"]["v"]}  # omit "s"
    real_inputs_omitted = {"a": a_no_s, "b": vectoradd_jax_ab["b"], "norm_ord": 2}

    abstract_inputs = to_shape_dtype_pytree(real_inputs_omitted)
    abstract_instance = AbstractInputSchema.model_validate({"inputs": abstract_inputs})
    with pytest.raises(ValueError, match=r"inputs\.a\.s.*omitted"):
        _patch_with_real_values(abstract_instance.inputs, real_inputs_omitted)


def test_apply_tesseract_recovers_an_omitted_array_default_end_to_end(
    vectoradd_jax_tess,
):
    """An omitted array field's real default is used in a traced ``apply_tesseract`` call."""
    out = apply_tesseract(
        to_traceable(vectoradd_jax_tess),
        {
            "a": {"v": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)},  # "s" omitted
            "b": {
                "v": jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
                "s": jnp.float32(0.5),
            },
            "norm_ord": 2,
        },
    )
    # a.s defaults to 1.0: result = a.v * a.s + b.v * b.s = [1,2,3] + [2,2.5,3].
    np.testing.assert_allclose(out["vector_add"]["result"], [3.0, 4.5, 6.0], rtol=1e-6)
