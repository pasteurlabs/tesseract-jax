# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ._version import __version__ as scm_version

__version__ = scm_version

# import public API of the package
# SIDE EFFECT: Register Tesseract as a pytree node
import jax
from tesseract_core import Tesseract

from tesseract_jax.primitive import apply_tesseract
from tesseract_jax.sow import save_intermediates, sow

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
    "save_intermediates",
    "sow",
]
