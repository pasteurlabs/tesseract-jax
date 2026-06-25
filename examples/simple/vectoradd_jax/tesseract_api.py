# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths


class Vector_and_Scalar(BaseModel):
    v: Differentiable[Array[..., Float32]] = Field(description="An arbitrary vector")
    s: Differentiable[Array[..., Float32]] = Field(
        description="A scalar (or, under vmap, a batch of scalars broadcast-compatible "
        "with v's leading dims)",
        # Default is a 0-d ``Float32`` array (not a Python ``float``) so that
        # downstream array ops like ``s[..., None]`` work even when the field
        # is not explicitly supplied. Pydantic v2 skips default validation, so
        # a bare ``1.0`` here would NOT be coerced to an array.
        default=np.float32(1.0),
    )

    @model_validator(mode="after")
    def validate_v_s_alignment(self) -> Self:
        # ``s * v`` is implemented as ``s[..., None] * v``; the validator
        # requires those two shapes to be broadcast-compatible. Use
        # ``len(.shape)`` instead of ``.ndim`` so the check also runs on
        # ``ShapeDType`` avals.
        v_shape, s_shape = tuple(self.v.shape), tuple(self.s.shape)
        try:
            jnp.broadcast_shapes(v_shape, (*s_shape, 1))
        except ValueError as exc:
            raise ValueError(
                f"v.shape={v_shape} and s.shape={s_shape} must be "
                "broadcast-compatible (with a trailing singleton added to s "
                "for the vector axis)."
            ) from exc
        return self

    def scale(self) -> Differentiable[Array[..., Float32]]:
        return self.s[..., None] * self.v


class InputSchema(BaseModel):
    a: Vector_and_Scalar = Field(
        description="An arbitrary vector and a scalar to multiply it by"
    )
    b: Vector_and_Scalar = Field(
        description="An arbitrary vector and a scalar to multiply it by "
        "must be of same shape as b"
    )
    norm_ord: int = Field(
        description="Order of norm (see numpy.linalg.norm)",
        default=2,
    )

    @model_validator(mode="after")
    def validate_shape_inputs(self) -> Self:
        a_shape, b_shape = tuple(self.a.v.shape), tuple(self.b.v.shape)
        try:
            jnp.broadcast_shapes(a_shape, b_shape)
        except ValueError as exc:
            raise ValueError(
                f"a.v and b.v must be broadcast-compatible. "
                f"Got {a_shape} and {b_shape} instead."
            ) from exc
        return self


class Result_and_Norm(BaseModel):
    result: Differentiable[Array[..., Float32]] = Field(
        description="Vector s_a·a + s_b·b"
    )
    normed_result: Differentiable[Array[..., Float32]] = Field(
        description="Normalized Vector s_a·a + s_b·b/|s_a·a + s_b·b|"
    )


class OutputSchema(BaseModel):
    vector_add: Result_and_Norm
    vector_min: Result_and_Norm


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    # ``s`` is one rank lower than ``v``; add a trailing axis so it broadcasts
    # against ``v`` correctly under vmap (where ``s`` may pick up leading dims).
    a_scaled = inputs["a"]["s"][..., None] * inputs["a"]["v"]
    b_scaled = inputs["b"]["s"][..., None] * inputs["b"]["v"]
    add_result = a_scaled + b_scaled
    min_result = a_scaled - b_scaled

    def safe_norm(x, ord):
        # Compute the norm along the last axis (with a small epsilon to ensure
        # differentiability and avoid division by zero). ``keepdims=True`` lets
        # the result broadcast against ``x`` under leading batch dims.
        return jnp.power(
            jnp.power(jnp.abs(x), ord).sum(axis=-1, keepdims=True) + 1e-8, 1 / ord
        )

    return {
        "vector_add": {
            "result": add_result,
            "normed_result": add_result / safe_norm(add_result, ord=inputs["norm_ord"]),
        },
        "vector_min": {
            "result": min_result,
            "normed_result": min_result / safe_norm(min_result, ord=inputs["norm_ord"]),
        },
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Multiplies a vector `a` by `s`, and sums the result to `b`."""
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    # ``jacfwd`` is the right default for this Tesseract: inputs and outputs
    # have similar dimensionality (array-in, array-out), so the per-trace cost
    # of forward-mode wins over reverse-mode's vmap-of-VJP.
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacfwd(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )
