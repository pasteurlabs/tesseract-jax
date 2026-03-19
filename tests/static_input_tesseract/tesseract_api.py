# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract with a static (non-array) int input for regression testing.

result = scale * sum(a^2 + z^2), where scale is a plain int (static input).

Field names are chosen so that the static field ("scale") sorts alphabetically
between the two array fields ("a" < "scale" < "z"). This matters because JAX
flattens dicts in sorted key order, so the static input ends up *between* the
array inputs in the flat list, which is the layout that triggers the VJP
index bug.
"""

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]]
    scale: int = Field(default=1)
    z: Differentiable[Array[(None,), Float32]]


class OutputSchema(BaseModel):
    result: Differentiable[Float32]


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(result=inputs.scale * jnp.sum(inputs.a**2 + inputs.z**2))


def abstract_eval(abstract_inputs):
    return {"result": ShapeDType(shape=(), dtype="float32")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    def f(a, z):
        return inputs.scale * jnp.sum(a**2 + z**2)

    _, vjp_fn = jax.vjp(f, inputs.a, inputs.z)
    da, dz = vjp_fn(cotangent_vector["result"])
    all_grads = {"a": da, "z": dz}
    return {k: all_grads[k] for k in vjp_inputs if k in all_grads}
