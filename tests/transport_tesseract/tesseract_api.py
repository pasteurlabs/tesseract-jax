# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An array-module-agnostic Tesseract for exercising both dispatch transports.

The same endpoint bodies serve the host (device->host->device round-trip) and the
cuda_ipc (GPU-direct) transports. The array module ``xp`` is ``cupy`` when the
runtime's ``gpu_transport`` is ``cuda_ipc`` and ``numpy`` otherwise. Writing the
math against ``xp`` (rather than duplicating a NumPy and a CuPy Tesseract) is how a
single parametrised fixture can drive the platform-sensitive tests on both
transports and be sure they compute the same thing.

On the cuda_ipc leg the compute stays in GPU memory, which is what lets the runtime
export the outputs by CUDA IPC with no device->host copy.

The math mirrors the all-float32 ``gpu_tesseract``: ``c = (a * scale + b) * mask``
with a *non-differentiable* array input ``mask`` (default all-ones) and a
*non-differentiable* output ``c_sum``. Between them they exercise the transport's
handling of a non-differentiable, non-static array input and a discarded output
slot -- the placeholder paths that differ between host and device.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.config import get_config


def _xp():
    """The array module to compute with, importing CuPy only for the cuda_ipc leg."""
    if get_config().gpu_transport == "cuda_ipc":
        import cupy

        return cupy
    return np


def _asxp(x):
    # x may arrive as a numpy array (base64 inputs) or a device array (cuda_ipc
    # inputs). asarray keeps a device array on-device and moves a host array onto
    # the device; on the numpy leg it is a plain host asarray.
    return _xp().asarray(x)


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


def _mask_for(inputs, a):
    xp = _xp()
    return _asxp(inputs.mask) if inputs.mask is not None else xp.ones_like(a)


def apply(inputs: InputSchema) -> OutputSchema:
    a = _asxp(inputs.a)
    b = _asxp(inputs.b)
    scale = float(inputs.scale)
    c = (a * scale + b) * _mask_for(inputs, a)
    # c_sum is a non-differentiable output: it makes the derivative endpoints emit
    # a placeholder for it, exercising the discarded-output-slot path per transport.
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
    xp = _xp()
    a = _asxp(inputs.a)
    scale = float(inputs.scale)
    mask = _mask_for(inputs, a)
    out = xp.zeros_like(a)
    if "a" in tangent_vector:
        out = out + _asxp(tangent_vector["a"]) * scale * mask
    if "b" in tangent_vector:
        out = out + _asxp(tangent_vector["b"]) * mask
    return {"c": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    scale = float(inputs.scale)
    mask = _mask_for(inputs, _asxp(inputs.a))
    ct = _asxp(cotangent_vector["c"]) * mask
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
    xp = _xp()
    a = _asxp(inputs.a)
    n = a.shape[0]
    scale = float(inputs.scale)
    mask = _mask_for(inputs, a)
    # c_i = (a_i*scale + b_i)*mask_i, so the Jacobian is diagonal:
    #   dc/da = diag(scale * mask),  dc/db = diag(mask)
    eye = xp.eye(n, dtype=xp.float32)
    out: dict[str, dict[str, Any]] = {dy: {} for dy in jac_outputs}
    for dy in jac_outputs:  # only "c"
        for dx in jac_inputs:
            diag = (scale * mask) if dx == "a" else mask
            out[dy][dx] = eye * diag  # dense (n, n) array, host or device per xp
    return out
