# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dict-valued differentiable field whose keys are not plain words.

A state dict is the obvious thing to put in one, and its keys carry dots.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    params: dict[str, Differentiable[Array[(3,), Float32]]]


class OutputSchema(BaseModel):
    result: Differentiable[Array[(3,), Float32]]


def apply(inputs: InputSchema) -> OutputSchema:
    p = inputs.model_dump()["params"]
    return {"result": sum(2.0 * np.asarray(v, np.float32) for v in p.values())}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    ct = np.asarray(cotangent_vector["result"], np.float32)
    return {k: 2.0 * ct for k in vjp_inputs}


def abstract_eval(abstract_inputs: Any) -> dict:
    return {"result": {"shape": (3,), "dtype": "float32"}}
