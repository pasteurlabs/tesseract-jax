# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract for exercising cuda_ipc dispatch.

``apply`` computes with JAX so its outputs stay in GPU memory, which is what lets
the runtime export them via CUDA IPC (no device->host copy). The core math is a
simple elementwise ``c = (a * scale + b) * mask`` so parity against a NumPy/JAX
reference is trivial to check.

``mask`` is a *non-differentiable* array input (contrast ``a`` / ``b``, which are
``Differentiable``). It defaults to all-ones so callers that omit it get the
plain ``a * scale + b``, but when supplied it exercises the derivative code's
handling of a non-differentiable, non-static array input -- the slot for which
JAX still expects a (placeholder) gradient.

The ``jacobian`` endpoint returns dense JAX device arrays, so it exercises the
GPU-direct (cuda_ipc) return path for a materialized Jacobian.

Unlike ``transport_tesseract`` it also accepts inputs in host memory, so the same
server can answer a host-callback dispatch for comparison.
"""

from typing import Any

import jax.numpy as jnp
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


# DLPack's device type code for CUDA memory (``kDLCUDA``).
_DLPACK_CUDA = 2


def _to_jax(x):
    # Inputs arrive in host memory (base64) or on the device (cuda_ipc). DLPack
    # adopts a device array without a copy but takes over its buffer, so each
    # input may be adopted only once.
    if x.__dlpack_device__()[0] == _DLPACK_CUDA:
        return jnp.from_dlpack(x)
    return jnp.asarray(x)


def _mask_for(inputs, a):
    return _to_jax(inputs.mask) if inputs.mask is not None else jnp.ones_like(a)


def apply(inputs: InputSchema) -> OutputSchema:
    a = _to_jax(inputs.a)
    b = _to_jax(inputs.b)
    scale = float(inputs.scale)
    c = (a * scale + b) * _mask_for(inputs, a)  # stays on GPU
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
    a = _to_jax(inputs.a)
    scale = float(inputs.scale)
    mask = _mask_for(inputs, a)
    out = jnp.zeros_like(a)
    if "a" in tangent_vector:
        out = out + _to_jax(tangent_vector["a"]) * scale * mask
    if "b" in tangent_vector:
        out = out + _to_jax(tangent_vector["b"]) * mask
    return {"c": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    scale = float(inputs.scale)
    mask = _mask_for(inputs, _to_jax(inputs.a))
    ct = _to_jax(cotangent_vector["c"]) * mask
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
    a = _to_jax(inputs.a)
    n = a.shape[0]
    scale = float(inputs.scale)
    mask = _mask_for(inputs, a)
    # c_i = (a_i*scale + b_i)*mask_i, so the Jacobian is diagonal:
    #   dc/da = diag(scale * mask),  dc/db = diag(mask)
    eye = jnp.eye(n, dtype=jnp.float32)
    out: dict[str, dict[str, Any]] = {dy: {} for dy in jac_outputs}
    for dy in jac_outputs:  # only "c"
        for dx in jac_inputs:
            diag = (scale * mask) if dx == "a" else mask
            out[dy][dx] = eye * diag  # dense (n, n) JAX array, stays on GPU
    return out
