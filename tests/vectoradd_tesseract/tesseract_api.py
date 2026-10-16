# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.jax_recipes import as_jax_array


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Arbitrary vector a")
    b: Array[(None,), Float32] = Field(description="Arbitrary vector b")


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(
        description="Vector s_a·a + s_b·b"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Adds two vectors `a` and `b`."""
    return OutputSchema(
        c=as_jax_array(inputs.a) + as_jax_array(inputs.b),
    )


def abstract_eval(abstract_inputs):
    """Abstract evaluation of the addition operation."""
    return {
        "c": abstract_inputs.a,
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Forward-mode AD for c = a + b.

    Since dc/da = 1 and b is not differentiable, the JVP is just the tangent of a.
    """
    if "a" not in tangent_vector:
        return {"c": jnp.zeros(inputs.a.shape, dtype=inputs.a.dtype)}
    return {"c": as_jax_array(tangent_vector["a"])}


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    """Full Jacobian for c = a + b: dc/da = I; b is non-differentiable."""
    n = inputs.a.shape[0]
    eye = jnp.eye(n, dtype=jnp.float32)
    return {dy: {dx: eye for dx in jac_inputs} for dy in jac_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return {"a": as_jax_array(cotangent_vector["c"])}
