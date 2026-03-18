# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


def rosenbrock(x, y, a=1.0, b=100.0):
    return (a - x) ** 2 + b * (y - x**2) ** 2


#
# Schemas — all arrays have a variadic leading batch dimension
#


class InputSchema(BaseModel):
    x: Differentiable[Array[..., Float64]] = Field(
        description="Batched scalar value x."
    )
    y: Differentiable[Array[..., Float64]] = Field(
        description="Batched scalar value y."
    )


class OutputSchema(BaseModel):
    result: Differentiable[Array[..., Float64]] = Field(
        description="Result of batched Rosenbrock function evaluation."
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Evaluates the Rosenbrock function element-wise over a batch."""
    result = rosenbrock(inputs.x, inputs.y)
    return OutputSchema(result=result)


#
# Optional endpoints
#


def jacobian(inputs, jac_inputs, jac_outputs):
    rosenbrock_signature = ["x", "y"]
    jac_result = {dy: {} for dy in jac_outputs}
    for dx in jac_inputs:
        grad_func = jax.jacrev(rosenbrock, argnums=rosenbrock_signature.index(dx))
        for dy in jac_outputs:
            jac_result[dy][dx] = grad_func(inputs.x, inputs.y)
    return jac_result


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    jac = jacobian(inputs, jvp_inputs, jvp_outputs)
    out = {}
    for dy in jvp_outputs:
        out[dy] = sum(jac[dy][dx] * tangent_vector[dx] for dx in jvp_inputs)
    return out


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    jac = jacobian(inputs, vjp_inputs, vjp_outputs)
    out = {}
    for dx in vjp_inputs:
        out[dx] = sum(jac[dy][dx] * cotangent_vector[dy] for dy in vjp_outputs)
    return out


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {"result": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}
