# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract whose output dtype differs from its input dtype.

Exercises the cuda_ipc return path on a genuinely mixed-dtype call (float32 in,
float64 out) so the shim's per-buffer dtype handling is covered by more than the
all-float32 ``gpu_tesseract``. Since the float64 buffer is allocated by the
producer, ``TESSERACT_JAX_TEST_ARRAY_MODULE`` selects whether it computes with
``jax`` or ``cupy``, as in ``transport_tesseract``.
"""

import os

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Float64, ShapeDType

_ARRAY_MODULE = os.environ["TESSERACT_JAX_TEST_ARRAY_MODULE"]

if _ARRAY_MODULE == "jax":
    import jax

    # Without x64, JAX would silently compute the float64 output in float32.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as xp

    _adopt = xp.from_dlpack
elif _ARRAY_MODULE == "cupy":
    import cupy as xp

    _adopt = xp.asarray
else:
    raise ValueError(f"Unknown TESSERACT_JAX_TEST_ARRAY_MODULE {_ARRAY_MODULE!r}")


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]] = Field(description="Input vector x.")


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(description="2*x as float64.")


def apply(inputs: InputSchema) -> OutputSchema:
    x = _adopt(inputs.x)
    return OutputSchema(y=(x * 2.0).astype(xp.float64))


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return {"y": (2.0 * _adopt(tangent_vector["x"])).astype(xp.float64)}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": (2.0 * _adopt(cotangent_vector["y"])).astype(xp.float32)}


def jacobian(inputs, jac_inputs, jac_outputs):
    n = inputs.x.shape[0]
    return {"y": {"x": 2.0 * xp.eye(n, dtype=xp.float64)}}
