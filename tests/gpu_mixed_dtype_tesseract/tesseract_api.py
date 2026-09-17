# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract whose output dtype differs from its input dtype.

Exercises the cuda_ipc return path on a genuinely mixed-dtype call (float32 in,
float64 out) so the shim's per-buffer dtype handling is covered by more than the
all-float32 ``gpu_tesseract``.
"""

import cupy
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Float64, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]] = Field(description="Input vector x.")


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(description="2*x as float64.")


def apply(inputs: InputSchema) -> OutputSchema:
    x = cupy.asarray(inputs.x)
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
