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
    z: Differentiable[Array[(5,), Float32]] = Field(
        description="Input parameter z, shape (5,)."
    )
    gamma: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters u shape (6,) and v shape (7,) as a dictionary.",
    )


class InputSchema(BaseModel):
    alpha: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Input parameters x shape (3,) and y shape (4,) as a dictionary.",
    )

    beta: Foo = Field(
        description="Nested input parameters.",
    )

    delta: list[Differentiable[Array[(None,), Float32]]] = Field(
        description="List of input parameters: delta[0] shape (8,), delta[1] shape (9,).",
    )

    epsilon: dict[str, Array[(None,), Float32]] = Field(
        description="Parameters k shape (2,) non-diff and m shape (10,) non-diff.",
    )

    zeta: list[Array[(None,), Float32]] = Field(
        description="List of parameters that are not differentiable: zeta[0] shape (11,), zeta[1] shape (12,).",
    )


class OutputSchema(BaseModel):
    metadata: Array[(2,), Float32] = Field(
        description="Non-differentiable output: x[:2] + y[:2], shape (2,)",
    )
    result: Differentiable[Array[(3,), Float32]] = Field(
        description="Complex combination of inputs, shape (3,)",
    )
    result_dict: dict[str, Differentiable[Array[(None,), Float32]]] = Field(
        description="Dict output: a shape (3,), b shape (5,)",
    )
    result_list: list[Differentiable[Array[(None,), Float32]]] = Field(
        description="List output: [shape (7,), shape (6,)]",
    )


#
# Required endpoints
#


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    x = inputs["alpha"]["x"]  # shape (3,)
    y = inputs["alpha"]["y"]  # shape (4,)

    z = inputs["beta"]["z"]  # shape (5,)
    u = inputs["beta"]["gamma"]["u"]  # shape (6,)
    v = inputs["beta"]["gamma"]["v"]  # shape (7,)

    d0 = inputs["delta"][0]  # shape (8,)
    d1 = inputs["delta"][1]  # shape (9,)

    k = inputs["epsilon"]["k"]  # shape (2,) non-differentiable
    m = inputs["epsilon"]["m"]  # shape (10,) non-differentiable

    z0 = inputs["zeta"][0]  # shape (11,) non-differentiable
    z1 = inputs["zeta"][1]  # shape (12,) non-differentiable

    # Complex operations with non-element-wise ops for non-trivial Jacobians
    # Since inputs have different shapes, we need to broadcast/slice appropriately

    # Element-wise terms (broadcast to shape 3)
    term1 = x * y[:3]
    # Dot product - couples all elements (use first 5 elements of v)
    term2 = jax.numpy.dot(z, v[:5])
    # Reductions - each output depends on all input elements
    term3 = u.sum()
    # Mixed reduction and element-wise (broadcast to shape 3)
    term4 = d0[:3] * d1.mean()
    # Non-linear with reduction (broadcast to shape 3)
    term5 = jax.numpy.exp(jax.numpy.clip(k.sum() * 0.1, -5, 5)) * m[:3]

    result = term1 + term2 + term3 + term4 + term5 + z0[:3] + z1[:3]

    # Dictionary outputs with various coupling
    result_dict = {
        "a": x + y[:3] + z.mean(),  # shape (3,), reduction couples z to outputs
        "b": z
        + u[:5]
        + jax.numpy.outer(x[:1], y[:1]).sum(),  # shape (5,), outer product coupling
    }

    # List outputs with reductions and cross-terms
    result_list = [
        d0[:7] + v + u.mean(),  # shape (7,), reduction couples all u elements
        d1[:6]
        + u
        + jax.numpy.sum(d0[:6] * v[:6]),  # shape (6,), dot-product-like coupling
    ]

    return {
        "metadata": x[:2] + y[:2],  # shape (2,), now depends on differentiable inputs
        "result": result,  # shape (3,)
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
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
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
