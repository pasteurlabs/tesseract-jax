# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.core as jc
import jax.numpy as jnp
import jax.tree
import numpy as np
from jax import ShapeDtypeStruct, dtypes, extend
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from tesseract_core import Tesseract

from tesseract_jax.batching import VMAP_METHOD_DISPATCH, VmapMethod
from tesseract_jax.tesseract_compat import Jaxeract
from tesseract_jax.tree_util import (
    _pytree_to_tesseract_flat,
    split_args,
    unflatten_args,
)

tesseract_dispatch_p = extend.core.Primitive("tesseract_dispatch")
tesseract_dispatch_p.multiple_results = True


class _Hashable:
    """A wrapper class to make non-hashable objects hashable by using their id.

    This is not a proper hash function, as two identical objects with different memory
    addresses will have different hashes.
    """

    def __init__(self, obj: Any) -> None:
        self.wrapped = obj

    def __hash__(self) -> int:
        try:
            return hash(self.wrapped)
        except TypeError:
            return id(self.wrapped)


@tesseract_dispatch_p.def_abstract_eval
def tesseract_dispatch_abstract_eval(
    *array_args: ArrayLike | ShapedArray,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
    jac_input_paths: tuple[str, ...] | None = None,
    jac_output_paths: tuple[str, ...] | None = None,
    jac_mode: Literal["fwd", "bwd"] = "bwd",
) -> tuple:
    """Define how to dispatch evals and pipe arguments."""
    if eval_func not in (
        "apply",
        "jacobian_vector_product",
        "vector_jacobian_product",
        "jacobian",
    ):
        raise NotImplementedError(eval_func)

    n_primals = len(is_static_mask) - sum(is_static_mask)

    if eval_func == "vector_jacobian_product":
        # We mustn't run forward evaluation of shapes, as out
        # of vjp has the same shapes as the primals; thus we can return early
        return tuple(array_args[:n_primals])

    if eval_func == "jacobian":
        # One array per (diff_output, diff_input) pair, shape = out_shape + in_shape.
        # `jac_input_paths` / `jac_output_paths` (when provided) restrict the
        # request to a sub-block of the Jacobian.
        primal_avals = array_args[:n_primals]
        primal_inputs = unflatten_args(
            primal_avals, static_args, input_pytreedef, is_static_mask
        )
        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=client.differentiable_input_paths
        )
        path_to_shape = {
            p: (tuple(v.shape), v.dtype)
            for p, v in flat_inputs.items()
            if v is not None
        }
        output_flat = _pytree_to_tesseract_flat(
            jax.tree.unflatten(output_pytreedef, range(len(output_avals))),
            schema_paths=client.differentiable_output_paths,
        )
        out_path_to_aval = {
            path: aval
            for (path, v), aval in zip(output_flat.items(), output_avals, strict=True)
            if v is not None
        }
        jac_inputs = (
            list(jac_input_paths)
            if jac_input_paths is not None
            else list(path_to_shape.keys())
        )
        jac_outputs = (
            list(jac_output_paths)
            if jac_output_paths is not None
            else list(out_path_to_aval.keys())
        )
        # Per JAX convention: fwd-mode → output dtype (jacfwd), bwd-mode →
        # input dtype (jacrev / `jax.jacobian`). Mirrors lineax's mode names.
        avals_out = []
        for op_path in jac_outputs:
            out_aval = out_path_to_aval[op_path]
            for ip in jac_inputs:
                in_shape, in_dtype = path_to_shape[ip]
                dtype = in_dtype if jac_mode == "bwd" else out_aval.dtype
                avals_out.append(
                    jax.core.ShapedArray(tuple(out_aval.shape) + in_shape, dtype)
                )
        return tuple(avals_out)

    # Those have the same shape as the outputs
    assert eval_func in ("apply", "jacobian_vector_product")
    return tuple(jax.core.ShapedArray(aval.shape, aval.dtype) for aval in output_avals)


def tesseract_dispatch_jvp_rule(
    in_args: tuple[ArrayLike, ...],
    tan_args: tuple[ArrayLike | ad.Zero, ...],
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
) -> tuple[tuple[ArrayLike, ...], tuple[ArrayLike, ...]]:
    """Defines how to dispatch jvp operation.

    Note this function is also called when evaluating a VJP or doing
    reverse-mode autodiff.

    """
    if eval_func != "apply":
        raise RuntimeError("Cannot take higher-order derivatives")

    #  https://github.com/jax-ml/jax/issues/16303#issuecomment-1585295819
    #  mattjj: taking a narrow pigeon-holed view, anywhere you see a symbolic
    #          zero `Zero(AbstractToken)`, i.e. in a JVP or transpose rule
    #          (not in ad.py's backward_pass), you probably want to instantiate
    #          it so that it's no longer symbolic

    # Compute which primals have non-zero tangents
    has_tangent = tuple(
        not (isinstance(t, jax._src.ad_util.Zero) or t is None) for t in tan_args
    )

    # Raise if a non-symbolic-zero tangent is provided for a non-differentiable input.
    _tangents_for_check = tuple(
        t if h else None for t, h in zip(tan_args, has_tangent, strict=True)
    )
    _tangent_inputs = unflatten_args(
        _tangents_for_check,
        static_args,
        input_pytreedef,
        is_static_mask,
        remove_static_args=True,
    )
    _flat_tangents = _pytree_to_tesseract_flat(
        _tangent_inputs, schema_paths=client.differentiable_input_paths
    )
    for path, val in _flat_tangents.items():
        if val is None:
            raise ValueError(
                f"Non-symbolic-zero tangent provided for non-differentiable input '{path}'. "
                f"If this input should be differentiable, mark it as "
                f"`Differentiable[...]` in the Tesseract input schema. Otherwise, "
                f"exclude it from the differentiated function's argument list "
                f"(using a closure or the `argnums` parameter), or apply "
                f"jax.lax.stop_gradient to it before passing to apply_tesseract."
            )

    tan_args_ = tuple(
        (jax.numpy.zeros_like(arg.aval) if not has_tan else arg)
        for arg, has_tan in zip(tan_args, has_tangent, strict=True)
    )
    # this leads to an abstract_eval call and a jvp
    jvp = tesseract_dispatch_p.bind(
        *in_args,
        *tan_args_,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func="jacobian_vector_product",
        vmap_method=vmap_method,
        materialise_jacobian=materialise_jacobian,
    )

    res = tesseract_dispatch_p.bind(
        *in_args,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func="apply",
        vmap_method=vmap_method,
        materialise_jacobian=materialise_jacobian,
    )

    return tuple(res), tuple(jvp)


ad.primitive_jvps[tesseract_dispatch_p] = tesseract_dispatch_jvp_rule


def tesseract_dispatch_transpose_rule(
    cotangent: Sequence[ArrayLike | ad.Zero],
    *args: ArrayLike,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
) -> tuple[ArrayLike | None, ...]:
    """Defines how to dispatch vjp operation."""
    assert eval_func in ("jacobian_vector_product",)

    n_primals = len(is_static_mask) - sum(is_static_mask)
    primal_args = args[:n_primals]

    # Raise if a cotangent for a non-differentiable output is not a symbolic zero.
    # Symbolic zeros (ad.Zero) are produced by JAX when gradients are blocked
    # (e.g. via jax.lax.stop_gradient) or when the output is not used in the loss.
    # Any other cotangent means the user accidentally included a non-diff output
    # in the gradient computation, likely due to a missing Differentiable[] annotation.
    dummy_output = jax.tree.unflatten(output_pytreedef, range(len(output_avals)))
    flat_output_info = _pytree_to_tesseract_flat(
        dummy_output, schema_paths=client.differentiable_output_paths
    )
    for cotan, (path, is_diff) in zip(cotangent, flat_output_info.items(), strict=True):
        if is_diff is None and not isinstance(cotan, jax._src.ad_util.Zero):
            raise ValueError(
                f"Non-symbolic-zero cotangent passed for non-differentiable output '{path}'. "
                f"If this output should be differentiable, mark it as "
                f"`Differentiable[...]` in the Tesseract output schema. Otherwise, "
                f"exclude it from the function return value (using pop or has_aux=True), "
                f"or wrap it with jax.lax.stop_gradient to produce a symbolic zero."
            )

    # Raise if a gradient is requested for a non-differentiable input.
    _primal_inputs = unflatten_args(
        primal_args, static_args, input_pytreedef, is_static_mask
    )
    _flat_inputs = _pytree_to_tesseract_flat(
        _primal_inputs, schema_paths=client.differentiable_input_paths
    )
    _non_static_paths = [
        p for p, m in zip(_flat_inputs, is_static_mask, strict=True) if not m
    ]
    _vjp_inputs_with_tangent = [
        p for p, h in zip(_non_static_paths, has_tangent, strict=True) if h
    ]
    for path in _vjp_inputs_with_tangent:
        if _flat_inputs[path] is None:
            raise ValueError(
                f"Non-symbolic-zero tangent provided for non-differentiable input '{path}'. "
                f"If this input should be differentiable, mark it as "
                f"`Differentiable[...]` in the Tesseract input schema. Otherwise, "
                f"exclude it from the differentiated function's argument list "
                f"(using a closure or the `argnums` parameter), or apply "
                f"jax.lax.stop_gradient to it before passing to apply_tesseract."
            )

    cotan_args_ = tuple(
        (
            jax.numpy.zeros_like(arg.aval)
            if isinstance(arg, jax._src.ad_util.Zero)
            else arg
        )
        for arg in cotangent
    )

    vjp = tesseract_dispatch_p.bind(
        *primal_args,
        *cotan_args_,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func="vector_jacobian_product",
        vmap_method=vmap_method,
        materialise_jacobian=materialise_jacobian,
    )

    return tuple([None] * len(primal_args) + list(vjp))


ad.primitive_transposes[tesseract_dispatch_p] = tesseract_dispatch_transpose_rule


def _raise_if_unimplemented(eval_func: str, client: Jaxeract) -> None:
    if eval_func not in client.available_methods:
        raise NotImplementedError(
            f"Endpoint '{eval_func}' not implemented for this Tesseract. "
            f"Available endpoints: {', '.join(client.available_methods)}. "
            f"To use this endpoint, implement the '{eval_func}' endpoint in your Tesseract object."
        )


def tesseract_dispatch(
    *array_args: ArrayLike | ShapedArray | Any,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef | None,
    output_avals: tuple[ShapeDtypeStruct, ...] | None,
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
    jac_input_paths: tuple[str, ...] | None = None,
    jac_output_paths: tuple[str, ...] | None = None,
    jac_mode: Literal["fwd", "bwd"] = "bwd",
) -> Any:
    """Defines how to dispatch lowering the computation.

    The dispatch that is not lowered is only called in cases where abstract eval is not needed.
    """
    _raise_if_unimplemented(eval_func, client)

    extra_kwargs: dict[str, Any] = {}
    if eval_func == "jacobian":
        extra_kwargs["jac_input_paths"] = jac_input_paths
        extra_kwargs["jac_output_paths"] = jac_output_paths
        extra_kwargs["jac_mode"] = jac_mode

    def _dispatch(*args: ArrayLike) -> Any:
        static_args_ = tuple(_unpack_hashable(arg) for arg in static_args)
        out = getattr(client, eval_func)(
            args,
            static_args_,
            input_pytreedef,
            output_pytreedef,
            output_avals,
            is_static_mask,
            has_tangent,
            **extra_kwargs,
        )
        if not isinstance(out, tuple) and output_avals is not None:
            out = (out,)
        return out

    result = _dispatch(*array_args)
    return result


tesseract_dispatch_p.def_impl(tesseract_dispatch)


def tesseract_dispatch_lowering(
    ctx: Any,
    *array_args: ArrayLike | ShapedArray | Any,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
    jac_input_paths: tuple[str, ...] | None = None,
    jac_output_paths: tuple[str, ...] | None = None,
    jac_mode: Literal["fwd", "bwd"] = "bwd",
) -> Any:
    """Defines how to dispatch lowering the computation."""
    _raise_if_unimplemented(eval_func, client)

    extra_kwargs: dict[str, Any] = {}
    if eval_func == "jacobian":
        extra_kwargs["jac_input_paths"] = jac_input_paths
        extra_kwargs["jac_output_paths"] = jac_output_paths
        extra_kwargs["jac_mode"] = jac_mode

    def _dispatch(*args: ArrayLike) -> Any:
        static_args_ = tuple(_unpack_hashable(arg) for arg in static_args)
        out = getattr(client, eval_func)(
            args,
            static_args_,
            input_pytreedef,
            output_pytreedef,
            output_avals,
            is_static_mask,
            has_tangent,
            **extra_kwargs,
        )
        if not isinstance(out, tuple):
            out = (out,)
        return out

    result, _, keepalive = mlir.emit_python_callback(
        ctx,
        _dispatch,
        None,
        array_args,
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
    )
    ctx.module_context.add_keepalive(keepalive)
    return result


mlir.register_lowering(tesseract_dispatch_p, tesseract_dispatch_lowering)


def tesseract_dispatch_batching(
    array_args: ArrayLike | ShapedArray | Any,
    axes: Sequence[Any],
    *,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
    jac_input_paths: tuple[str, ...] | None = None,
    jac_output_paths: tuple[str, ...] | None = None,
    jac_mode: Literal["fwd", "bwd"] = "bwd",
) -> Any:
    """Defines how to dispatch batch operations such as vmap (which is used by jax.jacobian)."""
    _raise_if_unimplemented(eval_func, client)

    # An unmapped axis is denoted by ``None``. (Older JAX versions exposed this
    # as the ``batching.not_mapped`` sentinel, which has since been removed; it
    # was always equal to ``None``.)
    n_primals = len(is_static_mask) - sum(is_static_mask)

    # When jacfwd/jacrev vmap a JVP/VJP with primals unbatched and (co)tangents
    # batched, materialise the entire Jacobian and apply to batch with matmul.
    # Gated by ``materialise_jacobian`` (None = auto, True = force, False = skip).
    if eval_func in ("jacobian_vector_product", "vector_jacobian_product"):
        primal_axes = axes[:n_primals]
        tangent_axes = axes[n_primals:]
        primals_unbatched = all(ax is None for ax in primal_axes)
        tangents_batched = any(ax is not None for ax in tangent_axes)
        endpoint_available = "jacobian" in client.available_methods
        if materialise_jacobian is True and not endpoint_available:
            raise RuntimeError(
                "materialise_jacobian=True but the Tesseract does not expose a "
                "'jacobian' endpoint."
            )
        use_shortcut = (
            primals_unbatched
            and tangents_batched
            and materialise_jacobian is not False
            and (materialise_jacobian is True or endpoint_available)
        )
        if use_shortcut:
            return _batched_via_jacobian(
                array_args,
                axes,
                static_args=static_args,
                input_pytreedef=input_pytreedef,
                output_pytreedef=output_pytreedef,
                output_avals=output_avals,
                is_static_mask=is_static_mask,
                has_tangent=has_tangent,
                client=client,
                eval_func=eval_func,
            )

    new_args = [
        arg if ax is None else jnp.moveaxis(arg, ax, 0)
        for arg, ax in zip(array_args, axes, strict=True)
    ]
    is_batched_mask = [ax is not None for ax in axes]

    batch_fn = VMAP_METHOD_DISPATCH[vmap_method]
    return batch_fn(
        new_args,
        is_batched_mask,
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


def _batched_via_jacobian(
    array_args: Sequence[Any],
    axes: Sequence[Any],
    *,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    has_tangent: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
) -> tuple[tuple, tuple]:
    """Batched JVP / VJP via one ``jacobian`` endpoint call + ``tensordot``.

    Assumes primals are unbatched and at least one (co)tangent is batched
    (caller checks). Returns ``(out_vals, out_axes)`` per JAX batching rule
    convention; output batch axis is always 0.
    """
    n_primals = len(is_static_mask) - sum(is_static_mask)
    primals = tuple(array_args[:n_primals])  # unbatched
    raw_tans = array_args[n_primals:]
    tan_axes = axes[n_primals:]

    # Identify the batch size from the first batched (co)tangent.
    batch_size = next(
        arg.shape[ax]
        for arg, ax in zip(raw_tans, tan_axes, strict=True)
        if ax is not None
    )

    # Bring each (co)tangent's batch axis to position 0; broadcast unbatched
    # ones across the new leading axis.
    def _to_batched(arg: Any, ax: Any) -> Any:
        if ax is None:
            return jnp.broadcast_to(arg, (batch_size, *arg.shape))
        return jnp.moveaxis(arg, ax, 0)

    tans = jax.tree.map(_to_batched, raw_tans, tan_axes)

    primal_inputs = unflatten_args(
        primals, static_args, input_pytreedef, is_static_mask
    )
    flat_inputs = _pytree_to_tesseract_flat(
        primal_inputs, schema_paths=client.differentiable_input_paths
    )
    output_flat = _pytree_to_tesseract_flat(
        jax.tree.unflatten(output_pytreedef, range(len(output_avals))),
        schema_paths=client.differentiable_output_paths,
    )

    # Map each (schema-diff) ∧ (has_tangent) input path to its position in
    # ``tans``. ``keys()`` give the path order of the Jacobian's columns;
    # ``values()`` give the corresponding ``tans`` positions. Excludes
    # JAX-zero tangents (JVP) and non-requested gradients (VJP).
    diff_input_path_to_pos: dict[str, int] = {}
    non_static_idx = 0
    for (p, v), is_static in zip(flat_inputs.items(), is_static_mask, strict=True):
        if is_static:
            continue
        if v is not None and has_tangent[non_static_idx]:
            diff_input_path_to_pos[p] = non_static_idx
        non_static_idx += 1

    # Map each diff output path to its leaf index in the output pytree.
    # ``keys()`` give the path order of the Jacobian's rows; ``values()`` give
    # the corresponding ``tans`` / ``output_avals`` positions.
    diff_output_path_to_pos: dict[str, int] = {
        p: i for i, (p, v) in enumerate(output_flat.items()) if v is not None
    }

    jac_arrays = tesseract_dispatch_p.bind(
        *primals,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        has_tangent=has_tangent,
        client=client,
        eval_func="jacobian",
        vmap_method=None,
        jac_input_paths=tuple(diff_input_path_to_pos),
        jac_output_paths=tuple(diff_output_path_to_pos),
        jac_mode="fwd" if eval_func == "jacobian_vector_product" else "bwd",
    )
    n_in = len(diff_input_path_to_pos)
    n_out = len(diff_output_path_to_pos)
    # Reshape the row-major (out, in) flat tuple into a 2D list-of-lists so
    # indexing reads as ``jac_blocks[out_i][in_j]``.
    jac_blocks = [[jac_arrays[i * n_in + j] for j in range(n_in)] for i in range(n_out)]

    # TODO(perf): for high-block-count tesseracts (many diff inputs by many
    # diff outputs), consider replacing the per-block matvec loop with a
    # single flattened matmul (``v_flat @ J_flat.T`` + split/reshape). The
    # current loop is clearer and XLA's fuser handles typical low-block-count
    # cases; revisit if a real workload shows it's a bottleneck.

    def _tree_sum(contribs: list[Any], slot: Any) -> Any:
        """Sum ``contribs``, falling back to a ``(batch_size, *slot.shape)`` zero array."""
        return jax.tree.reduce_associative(
            operator.add,
            contribs,
            identity=jnp.zeros((batch_size, *slot.shape), dtype=slot.dtype),
        )

    def _pad_nans(
        diff_results: list[Any],
        full_order: Sequence[Any],  # one entry per slot with .shape and .dtype
        is_diff: list[bool],  # True where the slot is a diff (computed) result
    ) -> tuple:
        """Assemble ``diff_results`` into ``full_order``, NaN-padding non-diff slots.

        NaN consistency with historical sequential jvp/vjp approach.
        """
        out, k = [], 0
        for item, diff in zip(full_order, is_diff, strict=True):
            if diff:
                out.append(diff_results[k])
                k += 1
            else:
                out.append(
                    jnp.full((batch_size, *item.shape), jnp.nan, dtype=item.dtype)
                )
        return tuple(out)

    # Using ``diff_avals`` / ``diff_primals`` as the first arg to
    # ``jax.tree.map`` pins iteration depth to the outer (n_out or n_in) list
    # level: those are flat lists of leaves, so each rest-arg's subtree at
    # that position is passed whole.

    if eval_func == "jacobian_vector_product":
        # Filter tangents to matrix-column order, then for each diff row sum
        # the per-column matvec contributions.
        filtered_tans = [tans[pos] for pos in diff_input_path_to_pos.values()]
        diff_avals = [output_avals[i] for i in diff_output_path_to_pos.values()]
        diff_outs = jax.tree.map(
            lambda slot, jac_row: _tree_sum(
                jax.tree.map(_matmul, jac_row, filtered_tans), slot
            ),
            diff_avals,
            jac_blocks,
        )
        outs = _pad_nans(
            diff_outs,
            output_avals,
            [v is not None for _p, v in output_flat.items()],
        )
        return outs, (0,) * len(outs)

    # VJP: transpose the outer/inner list structure to iterate
    # by column (one per diff input).
    filtered_cots = [tans[pos] for pos in diff_output_path_to_pos.values()]
    jac_cols = [list(col) for col in zip(*jac_blocks, strict=True)]
    diff_primals = [primals[pos] for pos in diff_input_path_to_pos.values()]
    diff_grads = jax.tree.map(
        lambda slot, jac_col: _tree_sum(
            jax.tree.map(_rmatmul, jac_col, filtered_cots), slot
        ),
        diff_primals,
        jac_cols,
    )
    diff_pos_set = set(diff_input_path_to_pos.values())
    grads = _pad_nans(
        diff_grads,
        primals,
        [i in diff_pos_set for i in range(len(primals))],
    )
    return grads, (0,) * len(grads)


def _matmul(matrix: Any, batched_vector: Any) -> Any:
    """Contract ``matrix``'s trailing axes against each row of ``batched_vector``."""
    batched_vector = batched_vector.astype(matrix.dtype)
    return jax.vmap(lambda v: jnp.tensordot(matrix, v, axes=v.ndim))(batched_vector)


def _rmatmul(matrix: Any, batched_vector: Any) -> Any:
    """Contract each row of ``batched_vector`` against ``matrix``'s leading axes."""
    batched_vector = batched_vector.astype(matrix.dtype)
    return jnp.tensordot(batched_vector, matrix, axes=batched_vector.ndim - 1)


batching.primitive_batchers[tesseract_dispatch_p] = tesseract_dispatch_batching


def _check_dtype(dtype: Any) -> None:
    dt = np.dtype(dtype)
    if dtypes.canonicalize_dtype(dt) != dt:
        raise ValueError(
            "Cannot return 64-bit values when `jax_enable_x64` is disabled. "
            "Try enabling it with `jax.config.update('jax_enable_x64', True)`."
        )


def _make_hashable(obj: Any) -> _Hashable:
    return _Hashable(obj)


def _unpack_hashable(obj: _Hashable) -> Any:
    return obj.wrapped


def _is_array_schema(prop_schema: dict) -> bool:
    """Check if a schema property describes an array type."""
    if "array_flags" in prop_schema:
        return True
    props = prop_schema.get("properties", {})
    obj_type = props.get("object_type", {})
    return obj_type.get("const") == "array"


def _resolve_ref(ref: str, all_schemas: dict) -> dict:
    """Resolve a $ref string like '#/components/schemas/Foo' to its schema dict.

    OpenAPI schemas use JSON References to avoid duplication. Nested or shared types
    in the Tesseract input schema are expressed as ``{"$ref": "#/components/schemas/Name"}``,
    and this helper dereferences them so we can inspect the actual schema properties.
    """
    parts = ref.lstrip("#/").split("/")
    return all_schemas[parts[-1]]


def _is_scalar(value: Any) -> bool:
    """Check if a value is a Python or NumPy scalar."""
    return isinstance(value, (int, float, complex, bool, np.number, np.bool_))


def _coerce_array_input(value: Any, field_name: str) -> Any:
    """Validate and coerce a single value that the schema expects to be an array.

    Accepts scalars (int, float, etc.), JAX/NumPy arrays, and any object implementing
    the ``__array__`` protocol. Raises ``TypeError`` for Python sequences and other
    unsupported types.
    """
    if isinstance(value, (jax.Array, np.ndarray, jc.Tracer)):
        return value
    if _is_scalar(value) or hasattr(value, "__array__"):
        return jnp.asarray(value)
    if isinstance(value, (list, tuple)):
        raise TypeError(
            f"Input '{field_name}' expects an array, but got {type(value).__name__}. "
            f"Please convert it to a JAX or NumPy array first, e.g. "
            f"jnp.array({field_name}) or np.array({field_name})."
        )
    raise TypeError(
        f"Input '{field_name}' expects an array, but got {type(value).__name__}. "
        f"Accepted types are: JAX/NumPy arrays or scalars (int, float, bool, complex)."
    )


def _validate_and_coerce_inputs(
    inputs: Any, input_schema: dict, all_schemas: dict
) -> Any:
    """Recursively validate and coerce inputs to JAX arrays where the schema expects arrays.

    Walks the input data alongside the OpenAPI schema. When a leaf field is expected
    to be an array (detected via array_flags or object_type):
    - Values implementing the ``__array__`` protocol (numpy/jax arrays) are converted
      via ``jnp.asarray()``.
    - Python scalars (int, float, bool, complex) are converted via ``jnp.asarray()``.
    - Python sequences (list, tuple) and other types are rejected with a ``TypeError``.
    """
    if not isinstance(inputs, dict):
        return inputs

    properties = input_schema.get("properties", {})
    if not properties:
        return inputs

    result = {}
    for key, value in inputs.items():
        if key not in properties:
            result[key] = value
            continue

        prop_schema = properties[key]

        # Resolve $ref if present
        if "$ref" in prop_schema:
            prop_schema = _resolve_ref(prop_schema["$ref"], all_schemas)

        if _is_array_schema(prop_schema):
            result[key] = _coerce_array_input(value, key)
        elif prop_schema.get("type") == "object" or "properties" in prop_schema:
            # Nested object - recurse
            result[key] = _validate_and_coerce_inputs(value, prop_schema, all_schemas)
        elif prop_schema.get("type") == "array" and "items" in prop_schema:
            # Schema-level array (list of items) - validate items
            items_schema = prop_schema["items"]
            if "$ref" in items_schema:
                items_schema = _resolve_ref(items_schema["$ref"], all_schemas)
            if _is_array_schema(items_schema) and isinstance(value, (list, tuple)):
                result[key] = type(value)(
                    _coerce_array_input(v, f"{key}[{i}]") for i, v in enumerate(value)
                )
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def apply_tesseract(
    tesseract_client: Tesseract,
    inputs: Any,
    *,
    vmap_method: VmapMethod = None,
    materialise_jacobian: bool | None = None,
) -> Any:
    """Applies the given Tesseract object to the inputs.

    This function is fully traceable and can be used in JAX transformations like
    jit, grad, etc. It will automatically dispatch to the appropriate Tesseract
    endpoint based on the requested operation.

    Scalar inputs (such as Python floats and ints) and objects implementing the
    ``__array__`` protocol are automatically converted to JAX arrays where the
    Tesseract's input schema expects arrays. Python sequences (lists, tuples) are
    rejected with a ``TypeError`` — convert them explicitly via ``jnp.array()``.

    Example:
        >>> from tesseract_core import Tesseract
        >>> from tesseract_jax import apply_tesseract
        >>>
        >>> # Create a Tesseract object and some inputs
        >>> tesseract_client = Tesseract.from_image("univariate")
        >>> tesseract_client.serve()
        >>> inputs = {"x": jax.numpy.array(1.0), "y": jax.numpy.array(2.0)}
        >>>
        >>> # Apply the Tesseract object to the inputs
        >>> # (this calls tesseract_client.apply under the hood)
        >>> apply_tesseract(tesseract_client, inputs)
        {'result': Array(100., dtype=float64)}
        >>>
        >>> # Scalar values are automatically converted to arrays
        >>> apply_tesseract(tesseract_client, {"x": 1.0, "y": 2.0})
        {'result': Array(100., dtype=float64)}
        >>>
        >>> # Compute the gradient of the outputs with respect to the inputs
        >>> # (this calls tesseract_client.vector_jacobian_product under the hood)
        >>> def apply_fn(x):
        ...     res = apply_tesseract(tesseract_client, x)
        ...     return res["result"].sum()
        >>> grad_fn = jax.grad(apply_fn)
        >>> grad_fn(inputs)
        {'x': Array(-400., dtype=float64, weak_type=True), 'y': Array(200., dtype=float64, weak_type=True)}

    Args:
        tesseract_client: The Tesseract object to apply.
        inputs: The inputs to apply to the Tesseract object.
        vmap_method: Strategy for handling ``jax.vmap`` batching. Must be set
            explicitly when using ``jax.vmap``; raises ``NotImplementedError``
            if ``jax.vmap`` is applied with the default ``None``.

            ``None`` (default)
                No vmap support. Raises ``NotImplementedError`` if ``jax.vmap``
                is applied. All other JAX transforms (jit, grad) work normally.

            ``"sequential"``
                Calls the Tesseract once per batch element via ``jax.lax.map``.
                Safe for all Tesseracts regardless of schema.

            ``"auto_experimental"``
                Experimental. Inspects the differentiable input schema at trace
                time. When all batched differentiable inputs use
                ``Array[..., dtype]`` (ellipsis shape), adds a leading ``(1,)``
                dim to unbatched args and sends a single batched call. Falls
                back to sequential otherwise. Only considers differentiable
                inputs; non-differentiable array inputs are not yet supported.

            ``"expand_dims"``
                Adds a leading ``(1,)`` dimension to every unbatched array arg
                and sends a single batched call. The Tesseract must broadcast
                ``(1, ...)`` against ``(batch, ...)`` internally. Use this when
                the Tesseract accepts a leading batch dimension on all inputs.

            ``"broadcast_all"``
                Broadcasts every unbatched array arg to ``(batch, ...)``, so all
                args share the same leading dimension. Use this when the Tesseract
                requires all inputs to have identical shapes.

            Python scalars (``float``, ``int``, ``bool``) are always static and
            are never batched regardless of the chosen method. Scalar arrays
            (0-d, e.g. ``Float64``) are treated as regular array args and will
            be transformed according to the method.

            See :doc:`/content/vmap-methods` for a detailed guide.
        materialise_jacobian: Strategy for batching (co)tangents at a single
            primal evaluation (i.e. ``vmap(lambda t: jvp(f, (x,), (t,)))``).
            If you need access to the entire Jacobian through
            ``jacfwd``/``jacrev`` the default of using the ``jacobian``
            endpoint should be much more performant. However, if your Jacobian
            is large and you are batching over a small number of (co)tangents
            (e.g. to perform low-rank approximations or apply coloring
            methods) the sequential per-(co)tangent path (``False``) may be
            preferable. Note that the ``False`` path uses the ``vmap_method``
            dispatch table: all non-``None`` methods route batched
            (co)tangents through the sequential fall-back.

            ``True``
                Whenever a ``jvp`` or ``vjp`` (co)tangent is batched, call the
                Tesseract's ``jacobian`` endpoint directly. Then use a PyTree-
                compatible matmul to compute its action on the batch of
                (co)tangents. Raises ``RuntimeError`` if the Tesseract does
                not expose a ``jacobian`` endpoint.

            ``False``
                Call the ``jacobian_vector_product``/``vector_jacobian_product``
                endpoint once per (co)tangent (``jax.lax.map``). Useful for
                low-rank approximations and coloring methods.

            ``None`` (default)
                Auto: use ``True`` when the Tesseract has a ``jacobian``
                endpoint, else fall back to ``False``.

    Returns:
        The outputs of the Tesseract object after applying the inputs.
    """
    if not isinstance(tesseract_client, Tesseract):
        raise TypeError(
            "The first argument must be a Tesseract object. "
            f"Got {type(tesseract_client)} instead."
        )

    if vmap_method not in VMAP_METHOD_DISPATCH:
        raise ValueError(
            f"Unknown vmap_method: {vmap_method!r}. "
            f"Must be one of {tuple(VMAP_METHOD_DISPATCH)}."
        )

    # Validate and coerce scalar / array-like inputs to JAX arrays where the schema expects them
    all_schemas = tesseract_client.openapi_schema["components"]["schemas"]
    input_schema = all_schemas.get("Apply_InputSchema", {})
    inputs = _validate_and_coerce_inputs(inputs, input_schema, all_schemas)

    has_func_transformation = False

    # determine if any array in the input pytree is a tracer
    inputs_flat, _ = jax.tree.flatten(inputs)
    for inp in inputs_flat:
        if isinstance(inp, jc.Tracer):
            has_func_transformation = True
            break

    if (
        has_func_transformation
        and "abstract_eval" not in tesseract_client.available_endpoints
    ):
        raise ValueError(
            "Given Tesseract object does not support abstract_eval, "
            "it is however called in combination with a JAX transformation "
            "like jit, grad, vmap, or pmap. "
            "Either remove the transformation or add an abstract_eval endpoint "
            "to the Tesseract object."
        )

    client = Jaxeract(tesseract_client)

    flat_args, input_pytreedef = jax.tree.flatten(inputs)
    is_static_mask = tuple(not isinstance(arg, jax.core.Tracer) for arg in flat_args)
    array_args, static_args = split_args(flat_args, is_static_mask)
    static_args = tuple(_make_hashable(arg) for arg in static_args)
    has_tangent = (True,) * len(array_args)

    if "abstract_eval" in tesseract_client.available_endpoints:
        # Get abstract values for outputs, so we can unflatten them later
        output_pytreedef, avals = None, None
        avals = client.abstract_eval(inputs)

        is_aval = lambda x: isinstance(x, dict) and "dtype" in x and "shape" in x
        flat_avals, output_pytreedef = jax.tree.flatten(avals, is_leaf=is_aval)
        for aval in flat_avals:
            if not is_aval(aval):
                continue
            _check_dtype(aval["dtype"])

        flat_avals = tuple(
            jax.ShapeDtypeStruct(shape=tuple(aval["shape"]), dtype=aval["dtype"])
            for aval in flat_avals
        )

        # Apply the primitive
        out = tesseract_dispatch_p.bind(
            *array_args,
            static_args=static_args,
            input_pytreedef=input_pytreedef,
            output_pytreedef=output_pytreedef,
            output_avals=flat_avals,
            is_static_mask=is_static_mask,
            has_tangent=has_tangent,
            client=client,
            eval_func="apply",
            vmap_method=vmap_method,
            materialise_jacobian=materialise_jacobian,
        )

        # Unflatten the output
        return jax.tree.unflatten(output_pytreedef, out)

    else:
        # If there is no abstract_eval endpoint, we cannot determine the output structure
        # In this case we send None for output_pytreedef and output_avals
        # and the primitive will return an unflattened output
        out = tesseract_dispatch_p.bind(
            *array_args,
            static_args=static_args,
            input_pytreedef=input_pytreedef,
            output_pytreedef=None,
            output_avals=None,
            is_static_mask=is_static_mask,
            has_tangent=has_tangent,
            client=client,
            eval_func="apply",
            vmap_method=vmap_method,
            materialise_jacobian=materialise_jacobian,
        )

        # Unflatten the output
        return out
