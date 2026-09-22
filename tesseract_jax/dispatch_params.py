# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parameter bundle threaded through the ``tesseract_dispatch`` primitive.

Every rule of the primitive and every batching strategy needs the same
set of descriptors. Passing them as a single frozen dataclass is neater than
a dozen individual keyword arguments on every rule signatures.

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
        static_args: Non-array leaves of the input pytree (a str, int or bool).
        input_pytreedef: Treedef to reassemble the flat operands into inputs.
        output_pytreedef: Treedef for the outputs, taken from ``abstract_eval``.
        output_avals: Shape/dtype of each flat output, taken from ``abstract_eval``.
        static_input_mask: One flag per input leaf, ``True`` where the leaf is a
            non-array (static) value; arrays, concrete or traced, are ``False``.
        has_tangent: One flag per non-static input, ``True`` where it carries a
            nonzero tangent (apply/JVP) or receives a nonzero cotangent (VJP).
        static_output_mask: One flag per output leaf, ``True`` where the leaf is a
            non-array (static) value that never enters the bind.
        has_cotangent: One flag per non-static output, ``True`` where a non-zero
            cotangent is carried for it. A ``vector_jacobian_product`` skips the
            outputs whose cotangent is a symbolic zero, since they add nothing to
            the input gradients. Empty outside a ``vector_jacobian_product``.
        static_output_values: The value of each static output leaf, as reported by
            ``abstract_eval``. ``apply`` compares these against what the endpoint
            returns; the other endpoints leave it empty.
        check_static_outputs: Whether ``apply`` makes that comparison. Set per call
            by ``apply_tesseract``, defaulting to ``TESSERACT_JAX_CHECK_STATIC_OUTPUTS``.
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
    output_pytreedef: PyTreeDef
    output_avals: tuple[ShapeDtypeStruct, ...]
    static_input_mask: tuple[bool, ...]
    has_tangent: tuple[bool, ...]
    client: "Jaxeract"
    eval_func: str
    static_output_mask: tuple[bool, ...] = ()
    has_cotangent: tuple[bool, ...] = ()
    static_output_values: tuple[Any, ...] = ()
    check_static_outputs: bool = True
    vmap_method: "VmapMethod" = None
    materialize_jacobian: bool | None = None
    jac_input_paths: tuple[str, ...] | None = None
    jac_output_paths: tuple[str, ...] | None = None
    jac_mode: Literal["fwd", "bwd"] = "bwd"

    @property
    def n_primals(self) -> int:
        """Number of non-static input leaves."""
        return len(self.static_input_mask) - sum(self.static_input_mask)

    def replace(self, **changes: Any) -> "DispatchParams":
        """Return a copy with the given fields overridden."""
        return replace(self, **changes)
