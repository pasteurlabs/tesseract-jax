# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tesseract with different input and output dtypes.

Used to exercise the jacfwd/jacrev dtype convention asymmetry:
jacfwd's result has output dtype, jacrev's result has input dtype. Served over
cuda_ipc, it also covers the shim's per-buffer dtype handling on a call that
returns a wider dtype than it takes.
"""

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32, Float64, ShapeDType
from tesseract_core.runtime.jax_recipes import as_jax_array

# Without x64, JAX would silently compute the float64 output in float32.
jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float32]] = Field(description="Input vector x.")


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(description="Output vector y.")


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=as_jax_array(inputs.x).astype(jnp.float64) * 2.0)


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def jacobian_vector_product(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    return {"y": (2.0 * as_jax_array(tangent_vector["x"])).astype(jnp.float64)}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    return {"x": (2.0 * as_jax_array(cotangent_vector["y"])).astype(jnp.float32)}


def jacobian(inputs, jac_inputs, jac_outputs):
    n = inputs.x.shape[0]
    return {"y": {"x": 2.0 * jnp.eye(n, dtype=jnp.float64)}}
