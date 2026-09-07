# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parameter bundle threaded through the ``tesseract_dispatch`` primitive.

Every rule of the primitive (abstract eval, jvp, transpose, batching, lowering,
impl) and every batching strategy needs the same set of descriptors: how the
flat operands map back to the input pytree, what the outputs look like, which
endpoint to call, and how to handle vmap. Passing them as a dozen individual
keyword arguments made the signatures unwieldy, so they live together in a
single frozen dataclass that travels as one primitive bind parameter.

The dataclass is frozen and every field is hashable, which is what lets JAX
compare two binds for equality and lets XLA common up identical Tesseract
calls (see the CSE note in ``primitive.py``).
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef

if TYPE_CHECKING:
    from tesseract_jax.batching import VmapMethod
    from tesseract_jax.tesseract_compat import Jaxeract


@dataclass(frozen=True)
class DispatchParams:
    """Descriptors carried by the ``tesseract_dispatch`` primitive.

    Attributes:
        static_args: Non-traced leaves of the input pytree, wrapped so they hash.
        input_pytreedef: Treedef to reassemble the flat operands into inputs.
        output_pytreedef: Treedef for the outputs; ``None`` when the Tesseract
            has no ``abstract_eval`` endpoint and outputs stay unflattened.
        output_avals: Shape/dtype of each flat output; ``None`` alongside
            ``output_pytreedef``.
        is_static_mask: One flag per input leaf, ``True`` where the leaf is static.
        has_tangent: One flag per non-static input, ``True`` where a (co)tangent
            is carried for it.
        client: The Tesseract wrapper the call dispatches to.
        eval_func: Which endpoint to invoke (``apply``, ``jacobian_vector_product``,
            ``vector_jacobian_product`` or ``jacobian``).
        vmap_method: Strategy for ``jax.vmap`` batching; see ``batching.py``.
        materialize_jacobian: Strategy for batching (co)tangents at a single primal.
        jac_input_paths: When set, restrict a ``jacobian`` call to these input columns.
        jac_output_paths: When set, restrict a ``jacobian`` call to these output rows.
        jac_mode: Dtype convention for a ``jacobian`` call (``"bwd"`` / ``"fwd"``).
    """

    static_args: tuple[Any, ...]
    input_pytreedef: PyTreeDef
    output_pytreedef: PyTreeDef | None
    output_avals: tuple[ShapeDtypeStruct, ...] | None
    is_static_mask: tuple[bool, ...]
    has_tangent: tuple[bool, ...]
    client: "Jaxeract"
    eval_func: str
    vmap_method: "VmapMethod" = None
    materialize_jacobian: bool | None = None
    jac_input_paths: tuple[str, ...] | None = None
    jac_output_paths: tuple[str, ...] | None = None
    jac_mode: Literal["fwd", "bwd"] = "bwd"

    @property
    def n_primals(self) -> int:
        """Number of non-static input leaves."""
        return len(self.is_static_mask) - sum(self.is_static_mask)

    def replace(self, **changes: Any) -> "DispatchParams":
        """Return a copy with the given fields overridden."""
        return replace(self, **changes)
