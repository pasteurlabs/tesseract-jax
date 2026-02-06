# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import equinox as eqx
import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

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

    epsilon: dict[str, Array[(None,), Float32]] = Field(
        description="Parameters k and m that are not differentiable.",
    )

    zeta: list[Array[(None,), Float32]] = Field(
        description="List of parameters that are not differentiable.",
    )


class OutputSchema(BaseModel):
    metadata: Array[(None,), Float32] = Field(
        description="Non-differentiable output: k + m",
    )
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="x * y + z * v + u + delta[0] + delta[1] * k + m + z0 + z1",
    )
    result_dict: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Dict output: a=x+y, b=z+u",
    )
    result_list: list[Differentiable[Array[(None,), Float32]]] = Field(
        description="List output: [d0+v, d1+u]",
    )


#
# Required endpoints
#


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    x = inputs["alpha"]["x"]
    y = inputs["alpha"]["y"]

    z = inputs["beta"]["z"]
    u = inputs["beta"]["gamma"]["u"]
    v = inputs["beta"]["gamma"]["v"]

    d0 = inputs["delta"][0]
    d1 = inputs["delta"][1]

    k = inputs["epsilon"]["k"]
    m = inputs["epsilon"]["m"]

    z0 = inputs["zeta"][0]
    z1 = inputs["zeta"][1]

    result = x * y + z * v + u + d0 + d1 * k + m + z0 + z1
    result_dict = {"a": x + y, "b": z + u}
    result_list = [d0 + v, d1 + u]

    return {
        "metadata": k + m,
        "result": result,
        "result_dict": result_dict,
        "result_list": result_list,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Random element-wise operation combining all inputs."""
    return apply_jit(inputs.model_dump())


#
# Optional endpoints
#


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: {"shape": x.shape, "dtype": str(x.dtype)}
        if is_shapedtype_struct(x)
        else x,
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )
