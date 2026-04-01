# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    data: Differentiable[Array[(..., None), Float32]]


class OutputSchema(BaseModel):
    result: Differentiable[Array[(..., None), Float32]]


def apply(inputs: InputSchema) -> OutputSchema:
    """Identity function - returns input unchanged."""
    return OutputSchema(result=inputs.data)


def abstract_eval(abstract_inputs):
    """Return output shapes from input shapes, with correct output dtype."""
    return {"result": {"shape": abstract_inputs.data.shape, "dtype": "float32"}}
