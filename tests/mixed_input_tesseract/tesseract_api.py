# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract with mixed input types: ellipsis arrays + primitive numerics.

result = a * x^2 + b * y^2

where x, y are differentiable arrays with ellipsis shape and a, b are
primitive float/int (static inputs that should not be batched).
"""

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    x: Differentiable[Array[..., Float64]] = Field(description="Array input x.")
    y: Differentiable[Array[..., Float64]] = Field(description="Array input y.")
    a: float = Field(default=1.0, description="Scalar coefficient a.")
    b: float = Field(default=1.0, description="Scalar coefficient b.")


class OutputSchema(BaseModel):
    result: Differentiable[Array[..., Float64]] = Field(
        description="Result of a*x^2 + b*y^2."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    result = inputs.a * inputs.x**2 + inputs.b * inputs.y**2
    return OutputSchema(result=result)


def abstract_eval(abstract_inputs):
    return {"result": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    def f(x, y):
        return inputs.a * x**2 + inputs.b * y**2

    _, vjp_fn = jax.vjp(f, inputs.x, inputs.y)
    dx, dy = vjp_fn(cotangent_vector["result"])
    all_grads = {"x": dx, "y": dy}
    return {k: all_grads[k] for k in vjp_inputs if k in all_grads}
