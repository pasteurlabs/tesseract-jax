# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract with a static (non-array) int input for regression testing.

y = scale * sum(x^2 + y^2), where scale is a plain int (static input).
"""

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]]
    y: Differentiable[Array[(None,), Float32]]
    scale: int = Field(default=1)


class OutputSchema(BaseModel):
    result: Differentiable[Float32]


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(result=inputs.scale * jnp.sum(inputs.x**2 + inputs.y**2))


def abstract_eval(abstract_inputs):
    return {"result": ShapeDType(shape=(), dtype="float32")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    def f(x, y):
        return inputs.scale * jnp.sum(x**2 + y**2)

    _, vjp_fn = jax.vjp(f, inputs.x, inputs.y)
    dx, dy = vjp_fn(cotangent_vector["result"])
    all_grads = {"x": dx, "y": dy}
    return {k: all_grads[k] for k in vjp_inputs if k in all_grads}
