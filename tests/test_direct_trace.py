# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traceable-specific behaviour a callback-path parity test can't exercise.

Eligibility, the device_transport conflict, that inlining actually happened
(not just that the answer is right), and the documented limitation that an
endpoint must return a plain dict, not a schema instance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

from tesseract_jax import apply_tesseract

# Exact host-callback lowering target (jax._src.callback.emit_python_callback).
# More specific than "custom-call" so an unrelated XLA custom call (e.g. a
# LAPACK op) in the Tesseract's own computation can't cause a false failure.
_CALLBACK_TARGET = "xla_ffi_python_cpu_callback"


def test_traceable_requires_in_process_tesseract(served_univariate_tesseract_raw):
    """A served (HTTPClient) Tesseract has no importable Python function to trace."""
    served_tess = Tesseract.from_url(served_univariate_tesseract_raw)
    with pytest.raises(ValueError, match="in-process Tesseract"):
        apply_tesseract(
            served_tess, dict(x=np.array(0.0), y=np.array(0.0)), traceable=True
        )


def test_traceable_and_device_transport_are_mutually_exclusive(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_tesseract(
            vectoradd_jax_tess,
            {**vectoradd_jax_ab, "norm_ord": 2},
            traceable=True,
            device_transport="cuda_ipc",
        )


def test_traceable_apply_has_no_host_callback_in_lowered_hlo(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    """The whole point of traceable=True: no opaque call in the lowering.

    Checked against the uncompiled lowering, not the compiled one -- stays
    backend-independent and fast, and the target's presence/absence agrees
    at both stages anyway.
    """
    inputs = {**vectoradd_jax_ab, "norm_ord": 2}

    def f(traceable):
        return apply_tesseract(vectoradd_jax_tess, inputs, traceable=traceable)[
            "vector_add"
        ]["result"].sum()

    direct_hlo = jax.jit(lambda: f(True)).lower().as_text()
    callback_hlo = jax.jit(lambda: f(False)).lower().as_text()

    assert _CALLBACK_TARGET not in direct_hlo
    # Positive control: confirms the string is actually present somewhere.
    assert _CALLBACK_TARGET in callback_hlo


def test_traceable_jvp_has_no_host_callback_in_lowered_hlo(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    """Same check for a derivative endpoint, not just the primal apply."""
    tangents = jax.tree.map(jnp.ones_like, vectoradd_jax_ab)

    def f(traceable):
        def full(ab):
            return apply_tesseract(
                vectoradd_jax_tess, {**ab, "norm_ord": 2}, traceable=traceable
            )

        return jax.jvp(full, (vectoradd_jax_ab,), (tangents,))[1]["vector_add"][
            "result"
        ]

    direct_hlo = jax.jit(lambda: f(True)).lower().as_text()
    callback_hlo = jax.jit(lambda: f(False)).lower().as_text()

    assert _CALLBACK_TARGET not in direct_hlo
    assert _CALLBACK_TARGET in callback_hlo


def test_traceable_requires_a_dict_returning_endpoint(mixed_dtype_tess):
    """A schema-constructing (not dict-returning) endpoint fails loudly, not silently.

    ``OutputSchema(...)`` re-validates with ``np.asarray``, which fails on a
    tracer the same way a traced input would.
    """
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    with pytest.raises(Exception, match=r"TracerArrayConversionError|Tracer"):
        apply_tesseract(mixed_dtype_tess, dict(x=x), traceable=True)


def test_patch_falls_back_to_schema_default_for_an_omitted_field(
    vectoradd_jax_tess, vectoradd_jax_ab
):
    """An omitted field falls back to the schema default instead of raising.

    A field omitted from ``inputs`` (relying on the schema default) has no
    pytree leaf at all; before the fallback this raised ``KeyError``. Tested
    against ``_patch_inputs`` directly, not ``apply_tesseract``:
    ``apply_tesseract``'s own eager ``abstract_eval`` call independently
    errors on this omission, for a pre-existing tesseract-core/example
    reason unrelated to this fix.
    """
    from tesseract_jax.direct_trace import _abstract_inputs_schema_for, _patch_inputs

    _api_module, AbstractInputSchema = _abstract_inputs_schema_for(
        vectoradd_jax_tess, "apply"
    )

    a_no_s = {"v": vectoradd_jax_ab["a"]["v"]}  # omit "s"
    real_inputs_omitted = {"a": a_no_s, "b": vectoradd_jax_ab["b"], "norm_ord": 2}

    patched = _patch_inputs(AbstractInputSchema, real_inputs_omitted)
    # Falls back to Vector_and_Scalar.s's own default rather than raising.
    assert float(patched.a.s) == pytest.approx(1.0)
