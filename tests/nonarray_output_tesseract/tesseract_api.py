# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract whose OutputSchema carries a non-array field alongside an array.

``backend`` and ``converged`` are the shape a provenance or status field takes in
practice: written by the endpoint, read by the caller, never differentiated. They
are legal in an ``OutputSchema`` and legal for ``abstract_eval`` to return, so
they reach ``apply_tesseract`` as plain leaves sitting next to real avals.
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
    converged: bool = Field(default=True, description="Whether the solve converged.")


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=inputs.x * 2.0, backend="reference", converged=True)


def abstract_eval(abstract_inputs: Any) -> dict:
    return {
        "y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64"),
        "backend": "reference",
        "converged": True,
    }


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return {"y": 2.0 * tangent_vector["x"]}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": 2.0 * cotangent_vector["y"]}
