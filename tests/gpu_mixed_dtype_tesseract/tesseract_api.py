# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract whose output dtype differs from its input dtype.

Exercises the cuda_ipc return path on a genuinely mixed-dtype call (float32 in,
float64 out) so the shim's per-buffer dtype handling is covered by more than the
all-float32 ``gpu_tesseract``.

``lie_about_dtype`` makes ``apply`` return an array whose dtype disagrees with
what ``abstract_eval`` declared. tesseract-core does not pin the output dtype, so
the host path would cast it. The cuda_ipc path cannot cast and must reject the
mismatch, which is the validation branch this drives on purpose.
"""

import cupy
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Float64, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]] = Field(description="Input vector x.")
    lie_about_dtype: bool = Field(
        default=False,
        description="If set, apply returns float32 where the schema declares float64.",
    )


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(description="2*x as float64.")


def apply(inputs: InputSchema) -> OutputSchema:
    x = cupy.asarray(inputs.x)
    if inputs.lie_about_dtype:
        # Return float32 despite the float64 schema. The (None,) output shape and
        # equal-or-smaller itemsize mean nothing upstream catches this before the
        # shim compares against XLA's buffer.
        return OutputSchema(y=(x * 2.0).astype(cupy.float32))
    return OutputSchema(y=(x * 2.0).astype(cupy.float64))


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return {"y": (2.0 * cupy.asarray(tangent_vector["x"])).astype(cupy.float64)}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": (2.0 * cupy.asarray(cotangent_vector["y"])).astype(cupy.float32)}


def jacobian(inputs, jac_inputs, jac_outputs):
    n = inputs.x.shape[0]
    return {"y": {"x": 2.0 * cupy.eye(n, dtype=cupy.float64)}}
