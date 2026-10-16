# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for testing code that passes GPU arrays through Tesseracts."""

from typing import Any

import jax
import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Copy a CuPy, JAX or NumPy array to a host NumPy array."""
    get = getattr(x, "get", None)
    return get() if callable(get) else np.asarray(x)


def is_on_gpu(x: jax.Array) -> bool:
    """Whether a JAX array lives on a GPU."""
    return any(d.platform == "gpu" for d in x.devices())
