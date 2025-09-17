# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any
import jax.numpy as jnp
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    a : Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )


class OutputSchema(BaseModel):
    b : Differentiable[Array[(None,), Float32]] = Field(
        description="Vector s_a·a + s_b·b"
    )
    c : Array[(None,), Float32] = Field(
        description="Constant vector [1.0, 1.0, 1.0]"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Multiplies a vector `a` by `s`, and sums the result to `b`."""
    return {
        "b": 2.0 * inputs.a,
        "c": jnp.array([1.0, 1.0, 1.0], dtype="float32"),
    }


# def vector_jacobian_product(
#     inputs: InputSchema,
#     vjp_inputs: set[str],
#     vjp_outputs: set[str],
#     cotangent_vector: dict[str, Any],
# ):
    
#     return {
#         "a": 2.0 * cotangent_vector["b"],
#     }



# def abstract_eval(abstract_inputs):
#     """Calculate output shape of apply from the shape of its inputs."""
#     return {
#         "b": ShapeDType(shape=(None,), dtype="float32"),
#         "c": ShapeDType(shape=(None,), dtype="float32"),
#     }




def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "b": ShapeDType(shape=(abstract_inputs.a.shape[0],), dtype="float32"),
        "c": ShapeDType(shape=(3,), dtype="float32"),
    }

