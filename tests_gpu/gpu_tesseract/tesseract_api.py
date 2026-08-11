# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract for exercising cuda_ipc dispatch.

``apply`` computes on CuPy so its outputs stay in GPU memory, which is what lets
the runtime export them via CUDA IPC (no device->host copy). The math is a
simple elementwise ``c = a * scale + b`` so parity against a NumPy/JAX reference
is trivial to check.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Differentiable[Array[(None,), Float32]] = Field(description="Vector b")
    scale: Float32 = Field(default=np.float32(2.0), description="Scalar scale")


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="a*scale + b")


def _to_cupy(x):
    import cupy

    # x may arrive as a numpy array (base64 inputs) or a cupy array (cuda_ipc
    # inputs). asarray keeps cupy on-device and moves numpy onto the device.
    return cupy.asarray(x)


def apply(inputs: InputSchema) -> OutputSchema:
    a = _to_cupy(inputs.a)
    b = _to_cupy(inputs.b)
    scale = float(inputs.scale)
    c = a * scale + b  # stays on GPU (cupy)
    return OutputSchema(c=c)


def abstract_eval(abstract_inputs):
    return {"c": abstract_inputs.a}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    import cupy

    scale = float(inputs.scale)
    out = cupy.zeros_like(_to_cupy(inputs.a))
    if "a" in tangent_vector:
        out = out + _to_cupy(tangent_vector["a"]) * scale
    if "b" in tangent_vector:
        out = out + _to_cupy(tangent_vector["b"])
    return {"c": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    scale = float(inputs.scale)
    ct = _to_cupy(cotangent_vector["c"])
    out = {}
    if "a" in vjp_inputs:
        out["a"] = ct * scale
    if "b" in vjp_inputs:
        out["b"] = ct
    return out
