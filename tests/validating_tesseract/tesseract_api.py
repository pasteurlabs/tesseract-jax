# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract whose failures span the ways one can go wrong.

Used to check that error reporting stays clean and attributable, in particular
for failures that *cannot* surface at trace time because they depend on values
rather than shapes.

Note the positivity check on ``x`` lives in ``apply`` rather than as a pydantic
``Field(gt=0)``: value constraints are also applied when ``abstract_eval``
validates the schema against ``ShapeDType`` placeholders, where they raise
``TypeError: '>' not supported between instances of 'ShapeDType' and 'float'``.
"""

import jax
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


class InputSchema(BaseModel):
    x: Differentiable[Float64] = Field(
        description="Scalar x; apply requires it to be strictly positive."
    )
    v: Array[(3,), Float64] = Field(
        default=(0.0, 0.0, 0.0), description="Fixed-shape vector of length 3."
    )
    mode: str = Field(
        default="ok",
        description="'ok', 'bad_output' (violate OutputSchema) or 'raise'.",
    )


class OutputSchema(BaseModel):
    result: Differentiable[Float64] = Field(description="sqrt(x).")


def apply(inputs: InputSchema) -> OutputSchema:
    """Return sqrt(x), or misbehave on request."""
    if inputs.mode == "raise":
        raise RuntimeError("deliberate failure inside the apply endpoint")
    if inputs.mode == "bad_output":
        # Valid Python, but not a Float64 scalar -> OutputSchema rejects it.
        return OutputSchema(result="not a number")
    if inputs.x <= 0:
        raise ValueError(
            f"x must be strictly positive, got {inputs.x}. "
            f"This depends on the value, so it cannot be caught at trace time."
        )
    return OutputSchema(result=inputs.x**0.5)


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return {"result": {"x": 0.5 * inputs.x**-0.5}}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    jac = jacobian(inputs, jvp_inputs, jvp_outputs)
    return {"result": jac["result"]["x"] * tangent_vector["x"]}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    jac = jacobian(inputs, vjp_inputs, vjp_outputs)
    return {"x": jac["result"]["x"] * cotangent_vector["result"]}


def abstract_eval(abstract_inputs):
    """Shapes only -- value bounds are unknowable here, which is the point."""
    return {"result": ShapeDType(shape=(), dtype="float64")}
