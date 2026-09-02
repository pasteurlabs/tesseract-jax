# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract whose static output leaf changes value between the two endpoints.

``abstract_eval`` always reports ``backend="reference"``. ``apply`` reports
``"fallback"`` when the input is negative, which is the honest shape of the
problem: a solver that only knows which backend it took once it has looked at
the numbers. Static leaves are read at trace time, so that answer arrives too
late to be used and ``apply_tesseract`` warns instead of silently discarding it.
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
