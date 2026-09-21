# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract for exercising cuda_ipc dispatch.

``apply`` computes on CuPy so its outputs stay in GPU memory, which is what lets
the runtime export them via CUDA IPC (no device->host copy). The core math is a
simple elementwise ``c = (a * scale + b) * mask`` so parity against a NumPy/JAX
reference is trivial to check.

``mask`` is a *non-differentiable* array input (contrast ``a`` / ``b``, which are
``Differentiable``). It defaults to all-ones so callers that omit it get the
plain ``a * scale + b``, but when supplied it exercises the derivative code's
handling of a non-differentiable, non-static array input -- the slot for which
JAX still expects a (placeholder) gradient.

The ``jacobian`` endpoint returns dense CuPy device arrays, so it exercises the
GPU-direct (cuda_ipc) return path for a materialized Jacobian.
"""

from typing import Any

import cupy
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Differentiable[Array[(None,), Float32]] = Field(description="Vector b")
    scale: Float32 = Field(default=np.float32(2.0), description="Scalar scale")
    mask: Array[(None,), Float32] | None = Field(
        default=None,
        description="Non-differentiable elementwise mask; defaults to all-ones.",
    )


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="(a*scale + b)*mask")
    c_sum: Array[(1,), Float32] = Field(
        description="Non-differentiable diagnostic: sum(c), shape (1,)."
    )


def _to_cupy(x):
    # x may arrive as a numpy array (base64 inputs) or a cupy array (cuda_ipc
    # inputs). asarray keeps cupy on-device and moves numpy onto the device.
    return cupy.asarray(x)


def _compute_c(inputs):
    a = _to_cupy(inputs.a)
    b = _to_cupy(inputs.b)
    scale = float(inputs.scale)
    c = a * scale + b  # stays on GPU (cupy)
    mask = _to_cupy(inputs.mask) if inputs.mask is not None else cupy.ones_like(a)
    return c * mask


def apply(inputs: InputSchema) -> OutputSchema:
    c = _compute_c(inputs)
    # c_sum is a non-differentiable output: it makes the derivative endpoints
    # emit a placeholder for it, exercising the GPU-direct return path for a
    # non-differentiable output slot.
    return OutputSchema(c=c, c_sum=c.sum().reshape(1))


def abstract_eval(abstract_inputs):
    return {
        "c": abstract_inputs.a,
        "c_sum": {"shape": (1,), "dtype": "float32"},
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    scale = float(inputs.scale)
    mask = (
        _to_cupy(inputs.mask)
        if inputs.mask is not None
        else cupy.ones_like(_to_cupy(inputs.a))
    )
    out = cupy.zeros_like(_to_cupy(inputs.a))
    if "a" in tangent_vector:
        out = out + _to_cupy(tangent_vector["a"]) * scale * mask
    if "b" in tangent_vector:
        out = out + _to_cupy(tangent_vector["b"]) * mask
    return {"c": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    scale = float(inputs.scale)
    mask = (
        _to_cupy(inputs.mask)
        if inputs.mask is not None
        else cupy.ones_like(_to_cupy(inputs.a))
    )
    ct = _to_cupy(cotangent_vector["c"]) * mask
    out = {}
    if "a" in vjp_inputs:
        out["a"] = ct * scale
    if "b" in vjp_inputs:
        out["b"] = ct
    return out


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    a = _to_cupy(inputs.a)
    n = a.shape[0]
    scale = float(inputs.scale)
    mask = _to_cupy(inputs.mask) if inputs.mask is not None else cupy.ones_like(a)
    # c_i = (a_i*scale + b_i)*mask_i, so the Jacobian is diagonal:
    #   dc/da = diag(scale * mask),  dc/db = diag(mask)
    eye = cupy.eye(n, dtype=cupy.float32)
    out: dict[str, dict[str, Any]] = {dy: {} for dy in jac_outputs}
    for dy in jac_outputs:  # only "c"
        for dx in jac_inputs:
            diag = (scale * mask) if dx == "a" else mask
            out[dy][dx] = eye * diag  # dense (n, n) CuPy array, stays on GPU
    return out
