# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract whose static output leaf differs between the two endpoints.

``abstract_eval`` always reports ``backend="reference"``, while ``apply`` reports
``"fallback"`` for a negative input. Static leaves are read at trace time, so
``apply``'s value is never used and ``apply_tesseract`` warns about the mismatch.
"""

from typing import Any

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    x: Differentiable[Array[(3,), Float64]] = Field(description="Input vector x.")


class OutputSchema(BaseModel):
    y: Differentiable[Array[(3,), Float64]] = Field(description="2 * x.")
    backend: str = Field(default="reference", description="Which solver ran.")


def apply(inputs: InputSchema) -> OutputSchema:
    backend = "fallback" if float(sum(inputs.x)) < 0.0 else "reference"
    return OutputSchema(y=inputs.x * 2.0, backend=backend)


def abstract_eval(abstract_inputs: Any) -> dict:
    return {
        "y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64"),
        "backend": "reference",
    }


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return {"y": 2.0 * tangent_vector["x"]}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": 2.0 * cotangent_vector["y"]}
