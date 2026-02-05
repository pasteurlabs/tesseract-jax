# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

#
# Schemas
#


class Foo(BaseModel):
    z: Differentiable[Array[(None,), Float32]] = Field(description="Input parameter x.")
    gamma: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters u and v as a dictionary.",
    )


class InputSchema(BaseModel):
    alpha: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters x and y as a dictionary.",
    )

    beta: Foo = Field(
        description="Nested input parameters.",
    )

    delta: list[Differentiable[Array[(None,), Float32]]] = Field(
        description="List of input parameters.",
    )


class OutputSchema(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="x * y + z * v + u + delta[0] + delta[1]",
    )


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    """Random elements wise operation combining all inputs."""
    x = inputs.alpha["x"]
    y = inputs.alpha["y"]

    z = inputs.beta.z
    u = inputs.beta.gamma["u"]
    v = inputs.beta.gamma["v"]

    d0 = inputs.delta[0]
    d1 = inputs.delta[1]

    result = x * y + z * v + u + d0 + d1
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

    def f(x, y, z, u, v, d0, d1):
        return x * y + z * v + u + d0 + d1

    full_jac = jax.jacfwd(f, argnums=(0, 1, 2, 3, 4, 5, 6))(
        inputs.alpha["x"],
        inputs.alpha["y"],
        inputs.beta.z,
        inputs.beta.gamma["u"],
        inputs.beta.gamma["v"],
        inputs.delta[0],
        inputs.delta[1],
    )

    # only return requested inputs and outputs
    jac = {}
    input_map = {
        "alpha.{x}": 0,
        "alpha.{y}": 1,
        "beta.z": 2,
        "beta.gamma.{u}": 3,
        "beta.gamma.{v}": 4,
        "delta.[0]": 5,
        "delta.[1]": 6,
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
        result_terms = []
        for dx in jvp_inputs:
            # only diagonal because of element-wise operations in primal
            term = tangent_vector[dx] * jax.numpy.diag(jac[dy][dx])
            result_terms.append(term)
        out[dy] = sum(result_terms)
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
        result_terms = []
        for dy in vjp_outputs:
            # only diagonal because of element-wise operations in primal
            term = cotangent_vector[dy] * jax.numpy.diag(jac[dy][dx])
            result_terms.append(term)
        out[dx] = sum(result_terms)
    return out


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    x = abstract_inputs.alpha["x"]
    return {"result": ShapeDType(shape=(x.shape[0],), dtype="float32")}
