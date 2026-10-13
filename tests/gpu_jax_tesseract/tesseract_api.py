# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU Tesseract that computes with JAX, for exercising cuda_ipc end-to-end.

Unlike the CuPy-based GPU test Tesseracts, it adopts inputs with
``jnp.from_dlpack`` instead of ``__cuda_array_interface__``, and returns JAX
arrays, whose buffers come from XLA's allocator.

Every endpoint rejects inputs that arrive in host memory, so a host copy on the
way in fails the call. A host copy on the way out is caught on the client side by
the FFI residency check armed in ``test_gpu_direct.py``.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Differentiable[Array[(None,), Float32]] = Field(description="Vector b")


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="a * 2 + b")


# DLPack's device type code for CUDA memory (``kDLCUDA``).
_DLPACK_CUDA = 2


def _from_device(x: Any) -> jax.Array:
    """Adopt a cuda_ipc input as a JAX array without copying it."""
    device_type, _ = x.__dlpack_device__()
    if device_type != _DLPACK_CUDA:
        raise RuntimeError(
            f"Expected a GPU-resident input, got a {type(x).__name__} in host memory."
        )
    return jnp.from_dlpack(x)


def _c(a: jax.Array, b: jax.Array) -> jax.Array:
    return a * np.float32(2.0) + b


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(c=_c(_from_device(inputs.a), _from_device(inputs.b)))


def abstract_eval(abstract_inputs):
    return {"c": abstract_inputs.a}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    _, pullback = jax.vjp(_c, _from_device(inputs.a), _from_device(inputs.b))
    da, db = pullback(_from_device(cotangent_vector["c"]))
    return {k: v for k, v in {"a": da, "b": db}.items() if k in vjp_inputs}
