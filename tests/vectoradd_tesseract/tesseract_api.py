# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


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
        c=inputs.a + inputs.b,
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
    return {
        "c": tangent_vector.get("a", 0.0),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return {
        "a": cotangent_vector["c"],
    }
