# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

#
# Schemas
#


class InputSchema(BaseModel):
    parameters: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters x and y as a dictionary.",
    )


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="Product of input parameters x and y."
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Evaluates the Rosenbrock function given input values and parameters."""
    x = inputs.parameters["x"]
    y = inputs.parameters["y"]
    result = x * y
    return OutputSchema(result=result)


#
# Optional endpoints
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    # dummy implementation for testing purposes

    jac = {}

    for dx in jac_inputs:
        jac[dx] = {}
        for dy in jac_outputs:
            if dx == "x" and dy == "result":
                jac[dy][dx] = inputs.parameters["y"]
            elif dx == "y" and dy == "result":
                jac[dy][dx] = inputs.parameters["x"]
            else:
                jac[dy][dx] = jax.numpy.zeros_like(inputs.parameters["x"])

    return jac


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    jac = jacobian(inputs, jvp_inputs, jvp_outputs)
    out = {}
    for dy in jvp_outputs:
        out[dy] = sum(jac[dy][dx] * tangent_vector[dx] for dx in jvp_inputs)
    return out


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    jac = jacobian(inputs, vjp_inputs, vjp_outputs)
    out = {}
    for dx in vjp_inputs:
        out[dx] = sum(jac[dy][dx] * cotangent_vector[dy] for dy in vjp_outputs)
    return out


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    x = abstract_inputs.parameters["x"]
    return {"result": ShapeDType(shape=(x.shape[0],), dtype="float32")}
