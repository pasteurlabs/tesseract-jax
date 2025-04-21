# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from . import _version

__version__ = _version.get_versions()["version"]

# import public API of the package
# SIDE EFFECT: Register Tesseract as a pytree node
import jax
from tesseract_core import Tesseract

from tesseract_jax.primitive import apply_tesseract

jax.tree_util.register_pytree_node(
    Tesseract,
    lambda x: ((), x),
    lambda x, _: x,
)
del jax
del Tesseract

# add public API as strings here, for example __all__ = ["obj"]
__all__ = [
    "apply_tesseract",
]
