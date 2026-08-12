# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract whose failures span the ways one can go wrong.

Used to check that error reporting stays clean and attributable, in particular
for failures that *cannot* surface at trace time because they depend on values
rather than shapes.

The positivity bound on ``x`` is expressed as an ``AfterValidator`` that returns
``ShapeDType`` untouched, rather than as ``Field(gt=0)``. A plain ``Field``
constraint is also applied when ``abstract_eval`` validates the schema against
``ShapeDType`` placeholders, where it dies with ``TypeError: '>' not supported
between instances of 'ShapeDType' and 'float'``. Passing abstract values through
is the workaround tesseract-core documents, and mirrors what ``ShapeDType``'s own
shape validator does.
"""

from typing import Annotated, Any

import jax
from pydantic import AfterValidator, BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

jax.config.update("jax_enable_x64", True)


def _strictly_positive(value: Any) -> Any:
    """Require a positive value, while leaving abstract evaluation alone.

    ``abstract_eval`` validates this schema with ``ShapeDType`` placeholders in
    place of values, so a value bound has to opt out of that pass explicitly.
    """
    if isinstance(value, ShapeDType):
        return value
    if value <= 0:
        raise ValueError(f"x must be strictly positive, got {value}")
    return value


class InputSchema(BaseModel):
    x: Annotated[Differentiable[Float64], AfterValidator(_strictly_positive)] = Field(
        description="Scalar x; must be strictly positive."
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
