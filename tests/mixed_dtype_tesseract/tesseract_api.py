# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract with different input and output dtypes.

Used to exercise the jacfwd/jacrev dtype convention asymmetry:
jacfwd's result has output dtype, jacrev's result has input dtype.
"""

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Float64, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]] = Field(description="Input vector x.")


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(description="Output vector y.")


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=inputs.x.astype(np.float64) * 2.0)


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return {"y": (2.0 * tangent_vector["x"]).astype(np.float64)}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": (2.0 * cotangent_vector["y"]).astype(np.float32)}


def jacobian(inputs, jac_inputs, jac_outputs):
    n = inputs.x.shape[0]
    return {"y": {"x": 2.0 * np.eye(n, dtype=np.float64)}}
