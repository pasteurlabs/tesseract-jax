# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""y = x**2, with a VJP that can be made wrong on purpose through ``bug``.

- ``"none"``: the correct VJP.
- ``"scale"``: the VJP is 5% too large.
- ``"zero"``: the VJP is zero.
- ``"nan"``: the VJP is NaN.
- ``"inf"``: the VJP is inf.
- ``"ones"``: the VJP ignores its cotangent, as if it were all ones.
- ``"raise"``: the VJP raises.
- ``"detached"``: apply ignores x and returns zeros, but the VJP is unchanged.

``noise`` multiplies y by ``1 + noise * sin(1e6 * x)``, a wiggle too fast for
any finite-difference step to resolve, like the tolerance of a solver. The VJP
ignores it.
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    x: Differentiable[Array[..., Float64]]
    bug: Literal["none", "scale", "zero", "nan", "inf", "ones", "raise", "detached"] = (
        "none"
    )
    noise: float = 0.0


class OutputSchema(BaseModel):
    y: Differentiable[Array[..., Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    x = np.asarray(inputs.x)
    if inputs.bug == "detached":
        return OutputSchema(y=np.zeros_like(x))
    return OutputSchema(y=x**2 * (1 + inputs.noise * np.sin(1e6 * x)))


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    x = np.asarray(inputs.x)
    grad = 2 * x * np.asarray(cotangent_vector["y"])
    if inputs.bug == "scale":
        grad = 1.05 * grad
    elif inputs.bug == "zero":
        grad = np.zeros_like(grad)
    elif inputs.bug == "nan":
        grad = np.full_like(grad, np.nan)
    elif inputs.bug == "inf":
        grad = np.full_like(grad, np.inf)
    elif inputs.bug == "ones":
        grad = 2 * x
    elif inputs.bug == "raise":
        raise ValueError("adjoint solve failed")
    return {"x": grad}
