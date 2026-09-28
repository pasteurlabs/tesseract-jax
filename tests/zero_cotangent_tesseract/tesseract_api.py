# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two differentiable outputs, one with a NaN gradient at the tested input.

When only ``safe`` enters a loss, ``unsafe``'s cotangent is a symbolic zero and
its (NaN) gradient must not be requested or folded into the input gradient.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    safe: Differentiable[Array[(None,), Float64]]
    unsafe: Differentiable[Array[(None,), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    # d/dx sqrt(x) is 1 / (2 sqrt(x)), which is NaN/inf at x = 0.
    return OutputSchema(safe=inputs.x * 2.0, unsafe=np.sqrt(inputs.x))


def abstract_eval(abstract_inputs):
    shape = abstract_inputs.x.shape
    return {
        "safe": ShapeDType(shape=shape, dtype="float64"),
        "unsafe": ShapeDType(shape=shape, dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    grad = np.zeros_like(inputs.x)
    if "safe" in vjp_outputs:
        grad = grad + 2.0 * cotangent_vector["safe"]
    if "unsafe" in vjp_outputs:
        with np.errstate(divide="ignore"):
            grad = grad + (0.5 / np.sqrt(inputs.x)) * cotangent_vector["unsafe"]
    return {"x": grad}
