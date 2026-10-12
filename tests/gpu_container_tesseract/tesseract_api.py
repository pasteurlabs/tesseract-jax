# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract served from a container in the GPU tests, computing ``c = 2a + b``.

It computes with CuPy when the container sees a GPU and with NumPy when it does
not, so one image covers both a server that can take part in cuda_ipc and one
that cannot (standing in for a server on another host).
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


def _array_module() -> Any:
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:  # noqa: BLE001 - no usable GPU, whatever the reason
        return np
    return np


xp = _array_module()


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Array[(None,), Float32] = Field(description="Vector b")


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="2a + b")


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(c=2 * xp.asarray(inputs.a) + xp.asarray(inputs.b))


def abstract_eval(abstract_inputs):
    return {"c": abstract_inputs.a}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return {"c": 2 * xp.asarray(tangent_vector["a"])}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return {"a": 2 * xp.asarray(cotangent_vector["c"])}
