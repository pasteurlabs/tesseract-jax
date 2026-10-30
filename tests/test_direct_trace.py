# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traceable-specific behaviour a callback-path parity test can't exercise.

Eligibility, that inlining actually happened (not just that the answer is
right), and the documented limitation that an endpoint must return a plain
dict, not a schema instance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_jax import apply_tesseract

# Exact host-callback lowering target (jax._src.callback.emit_python_callback).
# More specific than "custom-call" so an unrelated XLA custom call (e.g. a
# LAPACK op) in the Tesseract's own computation can't cause a false failure.
_CALLBACK_TARGET = "xla_ffi_python_cpu_callback"


def test_traceable_requires_in_process_tesseract(served_univariate_tesseract):
    """A served (HTTPClient) Tesseract has no importable Python function to trace."""
    with pytest.raises(ValueError, match="in-process Tesseract"):
        apply_tesseract(
            served_univariate_tesseract,
            dict(x=np.array(0.0), y=np.array(0.0)),
            traceable=True,
        )


def test_traceable_works_with_an_in_process_gpu_transport_tesseract(vectoradd_jax_ab):
    """``traceable=True`` is not blocked by an in-process GPU-transport request.

    ``apply_tesseract`` no longer takes ``gpu_transport`` (it is resolved from
    the Tesseract itself, lazily, only when a call is compiled for CUDA), so
    there is nothing left for ``traceable=True`` to conflict with here on CPU.
    The CUDA-lowering fallback for ``traceable`` (``TracedClient`` has no
    ``resolve_gpu_transport``) needs a GPU to exercise and is not covered here.
    """
    from tesseract_core import Tesseract

    tess = Tesseract.from_tesseract_api(
        "examples/simple/vectoradd_jax/tesseract_api.py", gpu_transport="cuda_ipc"
    )
    out = apply_tesseract(tess, {**vectoradd_jax_ab, "norm_ord": 2}, traceable=True)
    np.testing.assert_allclose(out["vector_add"]["result"], [4.0, 6.5, 9.0], rtol=1e-6)


def test_traceable_apply_has_no_host_callback_in_lowered_hlo(
    vectoradd_apply, vectoradd_jax_ab
):
    """The whole point of traceable=True: no opaque call in the lowering.

    Checked against the uncompiled lowering: backend-independent and fast.
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
    """Same check for a derivative endpoint, not just the primal apply."""
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
    """A float64 input is cast to the schema's declared float32, on both dispatch paths.

    Regression test for the traced path: ``_patch_with_real_values`` used to
    patch the real (uncast) value straight through instead of casting it, so
    a float64 input produced a float64 output that disagreed with
    ``abstract_eval``'s declared float32 and failed MLIR's output-type check.
    """
    ab_f64 = jax.tree.map(lambda x: x.astype(jnp.float64), vectoradd_jax_ab)
    out = vectoradd_apply(ab_f64, traceable)
    assert out["vector_add"]["result"].dtype == jnp.float32


@pytest.mark.parametrize("traceable", [False, True])
def test_static_leaves_survive_traced_dispatch(nonarray_output_tess, traceable):
    """A non-array (str/bool) output leaf comes back unchanged on both paths.

    ``direct_trace.py`` only patches ``ShapeDType`` (array) leaves; a static
    leaf is carried separately by the primitive (see ``test_nonarray_outputs.py``),
    so it should be unaffected by which dispatch path produced it.
    """
    x = jnp.arange(3, dtype=jnp.float64)

    def loss(x):
        return apply_tesseract(nonarray_output_tess, dict(x=x), traceable=traceable)[
            "y"
        ].sum()

    out = apply_tesseract(nonarray_output_tess, dict(x=x), traceable=traceable)
    assert out["backend"] == "reference"
    assert out["converged"] is True
    np.testing.assert_allclose(out["y"], 2.0 * x)
    np.testing.assert_allclose(jax.grad(loss)(x), 2.0 * jnp.ones_like(x))


@pytest.mark.parametrize("traceable", [False, True])
def test_symbolic_zero_cotangent_matches_on_traced_dispatch(
    zero_cotangent_tess, traceable
):
    """A symbolic-zero cotangent is dropped identically on both dispatch paths.

    ``unsafe`` has a NaN gradient at x = 0; when only ``safe`` enters the loss,
    its cotangent must be a symbolic zero and not poison the gradient -- on
    the traced path just as on the callback path (see ``test_api.py``'s
    ``test_unused_output_cotangent_is_not_requested``).
    """
    x = jnp.zeros(3, dtype=jnp.float64)

    def loss(x):
        return apply_tesseract(zero_cotangent_tess, dict(x=x), traceable=traceable)[
            "safe"
        ].sum()

    grad = jax.grad(loss)(x)
    np.testing.assert_array_equal(grad, [2.0, 2.0, 2.0])


@pytest.mark.parametrize("traceable", [False, True])
def test_gather_discarded_slots_match_on_traced_dispatch(gather_tess, traceable):
    """Non-differentiable int/float/complex output tangents match on both paths.

    ``gather_tess`` covers all three discarded-tangent dtype classes (see
    ``test_api.py``'s ``test_discarded_tangent_fill_value``); the traced path's
    ``apply`` is rewritten in ``jnp`` (``bincount``, scatter-add) instead of
    ``np``, since ``inputs.weights``/``inputs.indices`` may be tracers.
    """
    weights = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 2], dtype=jnp.int32)

    def apply_fn(w):
        return apply_tesseract(
            gather_tess, dict(weights=w, indices=indices), traceable=traceable
        )

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
    """A schema-constructing (not dict-returning) endpoint fails loudly, not silently.

    ``OutputSchema(...)`` re-validates with ``np.asarray``, which fails on a
    tracer the same way a traced input would.
    """
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    with pytest.raises(Exception, match=r"TracerArrayConversionError|Tracer"):
        apply_tesseract(mixed_dtype_tess, dict(x=x), traceable=True)


def test_omitted_array_field_with_a_default_raises_a_clear_error(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    """An omitted array field with a default raises, instead of patching in a placeholder.

    tesseract-core (>=1.15) validates a field's default against the abstract
    schema too, so ``Vector_and_Scalar.s``'s default (``np.float32(1.0)``)
    resolves to a ``ShapeDType`` placeholder, not the real ``1.0`` -- there is
    no real value left to patch in. Before this, the fallback used that
    placeholder as if it were a real value (a regression from an older
    tesseract-core, where the default stayed unvalidated); before the
    fallback existed at all, omitting the field raised ``KeyError``. Tested
    against ``_patch_with_real_values`` directly, not ``apply_tesseract``:
    ``apply_tesseract``'s own eager ``abstract_eval`` call independently
    errors on this omission, for a pre-existing tesseract-core/example reason
    unrelated to this fix.
    """
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
