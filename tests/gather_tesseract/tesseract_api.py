# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A gather Tesseract with integer array IO.

Exercises both discarded-slot paths with a non-inexact dtype: ``indices`` is a
non-differentiable integer *input* (hit by the vjp endpoint) and ``count`` is a
non-differentiable integer *output* (hit by the jvp endpoint).
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import (
    Array,
    Differentiable,
    Float32,
    Int32,
    ShapeDType,
)


class InputSchema(BaseModel):
    weights: Differentiable[Array[(None,), Float32]] = Field(
        description="Vector to gather from."
    )
    indices: Array[(None,), Int32] = Field(
        description="Gather indices (non-differentiable, integer)."
    )


class OutputSchema(BaseModel):
    gathered: Differentiable[Array[(None,), Float32]] = Field(
        description="weights[indices]."
    )
    count: Array[(None,), Int32] = Field(
        description="Times each index was gathered (non-differentiable, integer)."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Gather `weights` at `indices`, and count how often each index appears."""
    return OutputSchema(
        gathered=inputs.weights[inputs.indices],
        count=np.bincount(inputs.indices, minlength=inputs.indices.shape[0])[
            : inputs.indices.shape[0]
        ].astype(np.int32),
    )


def abstract_eval(abstract_inputs):
    """Both outputs take the shape of `indices`; only `gathered` is float."""
    shape = abstract_inputs.indices.shape
    return {
        "gathered": ShapeDType(shape=shape, dtype="float32"),
        "count": ShapeDType(shape=shape, dtype="int32"),
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """d(gathered)/d(weights) applied to a tangent: gather the tangent.

    `count` is absent from the result, so the framework fills its tangent slot
    with a discarded integer placeholder.
    """
    return {"gathered": tangent_vector["weights"][inputs.indices]}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Scatter-add the cotangent back into the weights.

    `indices` is absent from the result, so the framework fills its gradient
    slot with a discarded integer placeholder.
    """
    grad = np.zeros_like(inputs.weights)
    np.add.at(grad, inputs.indices, cotangent_vector["gathered"])
    return {"weights": grad}
