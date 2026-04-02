# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batching strategies for vmap over Tesseract primitives.

See :doc:`/content/vmap-methods` for a guide on choosing the right method.
"""

from itertools import compress
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.tree_util import PyTreeDef

from tesseract_jax.tesseract_compat import Jaxeract
from tesseract_jax.tree_util import (
    _merge_path,
    _pytree_to_tesseract_flat,
    combine_args,
    split_args,
)

VmapMethod = (
    Literal["sequential", "auto_experimental", "expand_dims", "broadcast_all"] | None
)


def _get_batch_size(new_args: list, is_batched_mask: list[bool]) -> int:
    """Get the batch size from the first batched arg."""
    return next(
        arg.shape[0]
        for arg, batched in zip(new_args, is_batched_mask, strict=True)
        if batched
    )


def _is_ellipsis_template(template: str | None, diff_paths: dict[str, Any]) -> bool:
    """Check if a matched schema template has ellipsis shape (no "shape" key)."""
    return template is not None and "shape" not in diff_paths[template]


def _dispatch_vectorized(
    new_args: list,
    is_batched_mask: list[bool],
    batch_size: int,
    n_primals: int,
    *,
    static_args: tuple,
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: "VmapMethod",
    tesseract_dispatch_p: Any,
) -> tuple[tuple, tuple]:
    """Common vectorized dispatch: broadcast JVP tangents, prepend batch dim, bind."""
    # JVP: broadcast primal/tangent to match if batch dims differ
    if eval_func == "jacobian_vector_product":
        for i in range(n_primals):
            if is_batched_mask[i] != is_batched_mask[i + n_primals]:
                new_args[i], new_args[i + n_primals] = jnp.broadcast_arrays(
                    new_args[i], new_args[i + n_primals]
                )

    batched_output_avals = tuple(
        ShapeDtypeStruct(shape=(batch_size, *aval.shape), dtype=aval.dtype)
        for aval in output_avals
    )
    outvals = tesseract_dispatch_p.bind(
        *new_args,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=batched_output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func=eval_func,
        vmap_method=vmap_method,
    )
    return tuple(outvals), (0,) * len(outvals)


# ---------------------------------------------------------------------------
# Batching methods
# ---------------------------------------------------------------------------


def sequential(
    new_args: list,
    is_batched_mask: list[bool],
    *,
    static_args: tuple,
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: "VmapMethod",
    tesseract_dispatch_p: Any,
) -> tuple[tuple, tuple]:
    """One Tesseract call per batch element via ``jax.lax.map``."""
    unbatched_args, batched_args = split_args(new_args, is_batched_mask)

    def _batch_fun(batched_args: tuple):
        combined_args = combine_args(unbatched_args, batched_args, is_batched_mask)
        return tesseract_dispatch_p.bind(
            *combined_args,
            static_args=static_args,
            input_pytreedef=input_pytreedef,
            output_pytreedef=output_pytreedef,
            output_avals=output_avals,
            is_static_mask=is_static_mask,
            has_tangent=has_tangent,
            client=client,
            eval_func=eval_func,
            vmap_method=vmap_method,
        )

    outvals = jax.lax.map(_batch_fun, batched_args)
    return tuple(outvals), (0,) * len(outvals)


def expand_dims(
    new_args: list,
    is_batched_mask: list[bool],
    *,
    static_args: tuple,
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: "VmapMethod",
    tesseract_dispatch_p: Any,
) -> tuple[tuple, tuple]:
    """Add a leading ``(1,)`` dim to unbatched array args; single Tesseract call.

    The Tesseract is responsible for broadcasting ``(1, ...)`` against
    ``(batch, ...)`` internally (e.g. via NumPy broadcasting rules).
    """
    kwargs = dict(
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func=eval_func,
        vmap_method=vmap_method,
        tesseract_dispatch_p=tesseract_dispatch_p,
    )
    n_primals = len(is_static_mask) - sum(is_static_mask)
    batch_size = _get_batch_size(new_args, is_batched_mask)

    # Tesseracts don't support batched cotangents
    if eval_func == "vector_jacobian_product" and any(is_batched_mask[n_primals:]):
        return sequential(new_args, is_batched_mask, **kwargs)

    new_args = [
        arg if batched else jnp.expand_dims(arg, 0)
        for arg, batched in zip(new_args, is_batched_mask, strict=True)
    ]

    return _dispatch_vectorized(
        new_args, is_batched_mask, batch_size, n_primals, **kwargs
    )


def broadcast_all(
    new_args: list,
    is_batched_mask: list[bool],
    *,
    static_args: tuple,
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: "VmapMethod",
    tesseract_dispatch_p: Any,
) -> tuple[tuple, tuple]:
    """Broadcast unbatched array args to ``(batch, ...)``; single Tesseract call.

    All array args sent to the Tesseract will have an identical leading batch
    dimension. This is useful for Tesseracts that require all inputs to have
    matching shapes.
    """
    kwargs = dict(
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func=eval_func,
        vmap_method=vmap_method,
        tesseract_dispatch_p=tesseract_dispatch_p,
    )
    n_primals = len(is_static_mask) - sum(is_static_mask)
    batch_size = _get_batch_size(new_args, is_batched_mask)

    # Tesseracts don't support batched cotangents
    if eval_func == "vector_jacobian_product" and any(is_batched_mask[n_primals:]):
        return sequential(new_args, is_batched_mask, **kwargs)

    new_args = [
        arg if batched else jnp.broadcast_to(arg, (batch_size, *arg.shape))
        for arg, batched in zip(new_args, is_batched_mask, strict=True)
    ]

    return _dispatch_vectorized(
        new_args, is_batched_mask, batch_size, n_primals, **kwargs
    )


def auto_experimental(
    new_args: list,
    is_batched_mask: list[bool],
    *,
    static_args: tuple,
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: "VmapMethod",
    tesseract_dispatch_p: Any,
) -> tuple[tuple, tuple]:
    """Auto-detect whether to vectorize based on the schema.

    Inspects the differentiable input schema at trace time: if all batched primal
    args have ellipsis shape (``Array[..., dtype]``), adds a leading ``(1,)``
    dimension to unbatched args and sends a single batched Tesseract call.
    Otherwise falls back to sequential.
    """
    kwargs = dict(
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func=eval_func,
        vmap_method=vmap_method,
        tesseract_dispatch_p=tesseract_dispatch_p,
    )
    n_primals = len(is_static_mask) - sum(is_static_mask)
    batch_size = _get_batch_size(new_args, is_batched_mask)

    # Tesseracts don't support batched cotangents
    if eval_func == "vector_jacobian_product" and any(is_batched_mask[n_primals:]):
        return sequential(new_args, is_batched_mask, **kwargs)

    # Determine which primal args need to support a batch dimension
    if eval_func == "jacobian_vector_product":
        needs_ellipsis = [
            b_p or b_t
            for b_p, b_t in zip(
                is_batched_mask[:n_primals], is_batched_mask[n_primals:], strict=True
            )
        ]
    else:
        needs_ellipsis = is_batched_mask[:n_primals]

    # Match each primal arg to its differentiable schema template.
    # A field has "ellipsis" shape if its template has no "shape" key.
    diff_paths = client.differentiable_input_paths
    dummy_tree = jax.tree.unflatten(input_pytreedef, range(len(is_static_mask)))
    flat_info = _pytree_to_tesseract_flat(dummy_tree, schema_paths=diff_paths)
    primal_info = compress(flat_info.items(), (not s for s in is_static_mask))
    primal_templates = [
        _merge_path(path, diff_paths)[1] if val is not None else None
        for path, val in primal_info
    ]
    is_ellipsis = [_is_ellipsis_template(t, diff_paths) for t in primal_templates]

    # All args that need batching must be ellipsis-shaped
    can_vectorize = all(
        e for e, needed in zip(is_ellipsis, needs_ellipsis, strict=True) if needed
    )

    if can_vectorize:
        # expand_dims on unbatched args to avoid incorrect broadcasting under nested vmap
        new_args = [
            arg if batched else jnp.expand_dims(arg, 0)
            for arg, batched in zip(new_args, is_batched_mask, strict=True)
        ]
        return _dispatch_vectorized(
            new_args, is_batched_mask, batch_size, n_primals, **kwargs
        )

    return sequential(new_args, is_batched_mask, **kwargs)


def _raise_not_implemented(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError(
        "vmap/jacobian is not supported with vmap_method=None. "
        "Pass vmap_method='sequential', 'expand_dims', 'broadcast_all', or 'auto_experimental' "
        "to apply_tesseract to enable vmap support. "
        "See https://docs.pasteurlabs.ai/projects/tesseract-jax/latest/content/vmap-methods.html"
    )


VMAP_METHOD_DISPATCH: dict[VmapMethod, Any] = {
    None: _raise_not_implemented,
    "sequential": sequential,
    "auto_experimental": auto_experimental,
    "expand_dims": expand_dims,
    "broadcast_all": broadcast_all,
}
