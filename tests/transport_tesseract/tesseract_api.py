# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An array-module-agnostic Tesseract for exercising every dispatch transport and producer.

``TESSERACT_JAX_TEST_ARRAY_MODULE`` selects the array module ``xp`` the endpoints
compute with: ``numpy`` (the default) for the host transport, and ``jax`` or
``cupy`` for cuda_ipc. Writing the math once against ``xp`` lets one parametrised
fixture run the platform-sensitive tests on every leg and be sure they compute the
same thing.

On the GPU legs, an array input that arrives in host memory fails the call, so a
host copy on the way in is caught. ``TESSERACT_JAX_TEST_ALLOW_HOST_INPUTS=1``
moves such inputs onto the device instead, for an in-process Tesseract loaded
without a GPU transport, which receives host inputs by design.

The math is ``c = (a * scale + b) * mask``, with a non-differentiable array input
``mask`` (default all-ones) and a non-differentiable output ``c_sum``. Between
them they exercise the placeholder paths for non-differentiable inputs and
discarded outputs, which differ between host and device.

``return_host`` makes ``apply`` and ``vector_jacobian_product`` return host
NumPy arrays instead, for a Tesseract whose results end up on the host whatever
it computes on.
"""

import os
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

_ARRAY_MODULE = os.environ.get("TESSERACT_JAX_TEST_ARRAY_MODULE", "numpy")
_ALLOW_HOST_INPUTS = os.environ.get("TESSERACT_JAX_TEST_ALLOW_HOST_INPUTS") == "1"

if _ARRAY_MODULE == "numpy":
    xp = np
elif _ARRAY_MODULE == "jax":
    import jax.numpy as xp
elif _ARRAY_MODULE == "cupy":
    import cupy as xp
else:
    raise ValueError(f"Unknown TESSERACT_JAX_TEST_ARRAY_MODULE {_ARRAY_MODULE!r}")


def _asxp(x: Any) -> Any:
    """Adopt an input array with ``xp``, without copying a GPU-resident one."""
    if xp is np:
        return np.asarray(x)
    # Served device inputs and the views an in-process Tesseract receives both
    # expose __cuda_array_interface__, while only the former also speak DLPack.
    if not hasattr(x, "__cuda_array_interface__"):
        if _ALLOW_HOST_INPUTS:
            return xp.asarray(x)
        raise RuntimeError(
            f"Expected a GPU-resident input, got a {type(x).__name__} in host memory."
        )
    return xp.from_dlpack(x) if _ARRAY_MODULE == "jax" else xp.asarray(x)


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Differentiable[Array[(None,), Float32]] = Field(description="Vector b")
    scale: Float32 = Field(default=np.float32(2.0), description="Scalar scale")
    mask: Array[(None,), Float32] | None = Field(
        default=None,
        description="Non-differentiable elementwise mask; defaults to all-ones.",
    )
    return_host: bool = Field(
        default=False, description="Return host NumPy arrays instead of xp ones."
    )


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="(a*scale + b)*mask")
    c_sum: Array[(1,), Float32] = Field(
        description="Non-differentiable diagnostic: sum(c), shape (1,)."
    )


def _mask_for(inputs, a):
    return _asxp(inputs.mask) if inputs.mask is not None else xp.ones_like(a)


def _maybe_to_host(x, inputs):
    if not inputs.return_host:
        return x
    return xp.asnumpy(x) if _ARRAY_MODULE == "cupy" else np.asarray(x)


def apply(inputs: InputSchema) -> OutputSchema:
    a = _asxp(inputs.a)
    b = _asxp(inputs.b)
    scale = float(inputs.scale)
    c = (a * scale + b) * _mask_for(inputs, a)
    # c_sum is a non-differentiable output: it makes the derivative endpoints emit
    # a placeholder for it, exercising the discarded-output-slot path per transport.
    return OutputSchema(
        c=_maybe_to_host(c, inputs),
        c_sum=_maybe_to_host(c.sum().reshape(1), inputs),
    )


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
        out["a"] = _maybe_to_host(ct * scale, inputs)
    if "b" in vjp_inputs:
        out["b"] = _maybe_to_host(ct, inputs)
    return out


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
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
