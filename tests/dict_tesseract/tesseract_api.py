# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

#
# Schemas
#


class NestedParametersSchema(BaseModel):
    z: Differentiable[Array[(None,), Float32]] = Field(description="Input parameter x.")
    double_nested_dict: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters u and v as a dictionary.",
    )


class InputSchema(BaseModel):
    parameters: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters x and y as a dictionary.",
    )

    nested_parameters: NestedParametersSchema = Field(
        description="Nested input parameters.",
    )


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="x * y + z * v + u",
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Random elements wise operation combining all inputs."""
    x = inputs.parameters["x"]
    y = inputs.parameters["y"]

    z = inputs.nested_parameters.z
    u = inputs.nested_parameters.double_nested_dict["u"]
    v = inputs.nested_parameters.double_nested_dict["v"]

    result = x * y + z * v + u
    return OutputSchema(result=result)


#
# Optional endpoints
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    # compute jacobian using jax

    def f(x, y, z, u, v):
        return x * y + z * v + u

    full_jac = jax.jacfwd(f, argnums=(0, 1, 2, 3, 4))(
        inputs.parameters["x"],
        inputs.parameters["y"],
        inputs.nested_parameters.z,
        inputs.nested_parameters.double_nested_dict["u"],
        inputs.nested_parameters.double_nested_dict["v"],
    )

    # only return requested inputs and outputs
    jac = {}
    input_map = {
        "x": 0,
        "y": 1,
        "z": 2,
        "u": 3,
        "v": 4,
    }
    for out in jac_outputs:
        jac[out] = {}
        for inp in jac_inputs:
            jac[out][inp] = full_jac[input_map[inp]]
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
