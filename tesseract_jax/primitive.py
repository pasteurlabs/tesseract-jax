# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, TypeVar

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

from tesseract_jax.tesseract_compat import Jaxeract, combine_args

T = TypeVar("T")

tesseract_dispatch_p = extend.core.Primitive("tesseract_dispatch")
tesseract_dispatch_p.multiple_results = True


class _Hashable:
    """A wrapper class to make non-hashable objects hashable by using their id.

    This is not a proper solution, as two identical objects with different memory
    addresses will have different hashes.
    However

    """

    def __init__(self, obj: Any) -> None:
        self.wrapped = obj

    def __hash__(self) -> int:
        try:
            return hash(self.wrapped)
        except TypeError:
            return id(self.wrapped)


def split_args(
    flat_args: Sequence[T], mask: Sequence[bool]
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Split a flat argument tuple according to mask (mask_False, mask_True)."""
    lists = ([], [])
    for a, m in zip(flat_args, mask, strict=True):
        lists[m].append(a)
    return tuple(tuple(args) for args in lists)


@tesseract_dispatch_p.def_abstract_eval
def tesseract_dispatch_abstract_eval(
    *array_args: ArrayLike | ShapedArray,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
) -> tuple:
    """Define how to dispatch evals and pipe arguments.

    We dont actually need to communicate with the Tesseract in abstract_eval,
    as the we already computed the abstract outputs, we just need to put them
    in the right format and return them.
    """
    if eval_func not in (
        "apply",
        "jacobian_vector_product",
        "vector_jacobian_product",
    ):
        raise NotImplementedError(eval_func)

    n_primals = len(is_static_mask) - sum(is_static_mask)

    if eval_func == "vector_jacobian_product":
        # We mustn't run forward evaluation of shapes, as out
        # of vjp has the same shapes as the primals; thus we can return early
        return tuple(array_args[:n_primals])

    # Those have the same shape as the outputs
    assert eval_func in ("apply", "jacobian_vector_product")
    return tuple(jax.core.ShapedArray(aval.shape, aval.dtype) for aval in output_avals)


def filter_zeros(
    primal_args: Sequence[ArrayLike],
    tan_args: Sequence[ArrayLike],
    static_args: tuple[_Hashable, ...],
    is_static_mask: tuple[bool, ...],
) -> tuple[
    tuple[ArrayLike, ...],
    tuple[ArrayLike, ...],
    tuple[_Hashable, ...],
    tuple[bool, ...],
]:
    """Filter out Zero tangents and their corresponding primals to avoid unnecessary computation.

    For positions where tangents are Zero, we move both the primal and the zero tangent
    to static_args, since they don't participate in differentiation.

    Args:
        primal_args: Primal array arguments (length = number of non-static args)
        tan_args: Tangent arguments (length = number of non-static args)
        static_args: Static arguments from original inputs
        is_static_mask: Boolean mask for original flat inputs

    Returns:
        Filtered primals, filtered tangents, adjusted static args, adjusted is_static mask
    """
    return primal_args, tan_args, static_args, is_static_mask
    # Identify which tangent positions are Zero
    zeros_mask = tuple(isinstance(arg, jax._src.ad_util.Zero) for arg in tan_args)

    # Filter out the Zero positions from both primals and tangents
    primal_args_filtered = tuple(
        arg for arg, is_zero in zip(primal_args, zeros_mask, strict=True) if not is_zero
    )
    tan_args_filtered = tuple(
        arg for arg, is_zero in zip(tan_args, zeros_mask, strict=True) if not is_zero
    )

    # Reconstruct which positions in the original mask were arrays
    array_positions = [i for i, is_static in enumerate(is_static_mask) if not is_static]

    # Build new mask: original statics stay static, Zero positions become static
    new_is_static_mask = list(is_static_mask)
    for i, is_zero in enumerate(zeros_mask):
        if is_zero:
            original_position = array_positions[i]
            new_is_static_mask[original_position] = True

    # Insert primals that had zero tangents into static_args at the right positions
    static_args_list = list(static_args)
    zero_primal_positions = []
    for i, is_zero in enumerate(zeros_mask):
        if is_zero:
            original_position = array_positions[i]
            zero_primal_positions.append((original_position, primal_args[i]))

    # Sort by position and insert in correct order
    for original_position, primal in sorted(zero_primal_positions, key=lambda x: x[0]):
        # Count how many statics come before this position in the original mask
        insert_position = sum(1 for j in range(original_position) if is_static_mask[j])
        # Adjust for already-inserted zeros
        insert_position += sum(
            1 for pos, _ in zero_primal_positions if pos < original_position
        )
        static_args_list.insert(insert_position, _make_hashable(primal))

    return (
        primal_args_filtered,
        tan_args_filtered,
        tuple(static_args_list),
        tuple(new_is_static_mask),
    )


def tesseract_dispatch_jvp_rule(
    in_args: tuple[ArrayLike, ...],
    tan_args: tuple[ArrayLike, ...],
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
) -> tuple[tuple[ArrayLike, ...], tuple[ArrayLike, ...]]:
    """Defines how to dispatch jvp operation."""
    if eval_func != "apply":
        raise RuntimeError("Cannot take higher-order derivatives")

    #  https://github.com/jax-ml/jax/issues/16303#issuecomment-1585295819
    #  mattjj: taking a narrow pigeon-holed view, anywhere you see a symbolic
    #          zero `Zero(AbstractToken)`, i.e. in a JVP or transpose rule
    #          (not in ad.py's backward_pass), you probably want to instantiate
    #          it so that it's no longer symbolic

    # TODO: create a mask for Zero (essentially, jvp_in)? or maybe substitute it
    #       with something that jax still likes, while not wasting memory and time?

    in_args_, tan_args_, static_args_, is_static_mask_ = filter_zeros(
        in_args, tan_args, static_args, is_static_mask
    )

    jvp = tesseract_dispatch_p.bind(
        *in_args_,
        *tan_args_,
        static_args=static_args_,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask_,
        client=client,
        eval_func="jacobian_vector_product",
    )

    res = tesseract_dispatch_p.bind(
        *in_args,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        client=client,
        eval_func="apply",
    )

    return tuple(res), tuple(jvp)


ad.primitive_jvps[tesseract_dispatch_p] = tesseract_dispatch_jvp_rule


def tesseract_dispatch_transpose_rule(
    cotangent: Sequence[ArrayLike],
    *args: ArrayLike,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef,
    output_avals: tuple[ShapeDtypeStruct, ...],
    is_static_mask: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
) -> tuple[ArrayLike | None, ...]:
    """Defines how to dispatch vjp operation."""
    assert eval_func in ("jacobian_vector_product",)

    n_primals = len(is_static_mask) - sum(is_static_mask)
    args = args[:n_primals]

    args_, cotan_args_, static_args_, is_static_mask_ = filter_zeros(
        args, cotangent, static_args, is_static_mask
    )

    vjp = tesseract_dispatch_p.bind(
        *args_,
        *cotan_args_,
        static_args=static_args_,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask_,
        client=client,
        eval_func="vector_jacobian_product",
    )
    # TODO: I'm not sure this makes sense given these docs:
    #       https://jax.readthedocs.io/en/latest/jax-primitives.html#transposition
    #       "A tuple with the cotangent of the inputs, with the value None corresponding to the constant arguments"
    #       ...but if I provide only cotangent, jax complains, and if I investigate its internals,
    #       I see it chokes on map(partial(write_cotangent, eqn.primitive), eqn.invars, cts_out),
    #       where eqn.invars ends up being longer than cts_out.

    return tuple([None] * len(args) + list(vjp))


ad.primitive_transposes[tesseract_dispatch_p] = tesseract_dispatch_transpose_rule


def tesseract_dispatch(
    *array_args: ArrayLike | ShapedArray | Any,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef | None,
    output_avals: tuple[ShapeDtypeStruct, ...] | None,
    is_static_mask: tuple[bool, ...],
    client: Jaxeract,
    eval_func: str,
) -> Any:
    """Defines how to dispatch lowering the computation.

    The dispatch that is not lowered is only called in cases where abstract eval is not needed.
    """

    def _dispatch(*args: ArrayLike) -> Any:
        static_args_ = tuple(_unpack_hashable(arg) for arg in static_args)
        out = getattr(client, eval_func)(
            args,
            static_args_,
            input_pytreedef,
            output_pytreedef,
            output_avals,
            is_static_mask,
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
    client: Jaxeract,
    eval_func: str,
) -> Any:
    """Defines how to dispatch lowering the computation."""

    def _dispatch(*args: ArrayLike) -> Any:
        static_args_ = tuple(_unpack_hashable(arg) for arg in static_args)
        out = getattr(client, eval_func)(
            args,
            static_args_,
            input_pytreedef,
            output_pytreedef,
            output_avals,
            is_static_mask,
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
    client: Jaxeract,
    eval_func: str,
) -> Any:
    """Defines how to dispatch batch operations such as vmap (which is used by jax.jacobian)."""
    new_args = [
        arg if ax is batching.not_mapped else jnp.moveaxis(arg, ax, 0)
        for arg, ax in zip(array_args, axes, strict=True)
    ]

    is_batched_mask = [d is not batching.not_mapped for d in axes]
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
            client=client,
            eval_func=eval_func,
        )

    outvals = jax.lax.map(_batch_fun, batched_args)

    return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[tesseract_dispatch_p] = tesseract_dispatch_batching


def _check_dtype(dtype: Any) -> None:
    dt = np.dtype(dtype)
    if dtypes.canonicalize_dtype(dt) != dt:
        raise ValueError(
            "Cannot return 64-bit values when `jax_enable_x64` is disabled. "
            "Try enabling it with `jax.config.update('jax_enable_x64', True)`."
        )


def _is_static(x: Any) -> bool:
    # This is not right!
    # A traced array that is traced because of JIT and can be differentiable will be marked as differentiable
    if isinstance(x, jax.core.Tracer):
        return False
    return True


def _make_hashable(obj: Any) -> _Hashable:
    return _Hashable(obj)


def _unpack_hashable(obj: _Hashable) -> Any:
    return obj.wrapped


def apply_tesseract(
    tesseract_client: Tesseract,
    inputs: Any,
) -> Any:
    """Applies the given Tesseract object to the inputs.

    This function is fully traceable and can be used in JAX transformations like
    jit, grad, etc. It will automatically dispatch to the appropriate Tesseract
    endpoint based on the requested operation.

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

    Returns:
        The outputs of the Tesseract object after applying the inputs.
    """
    if not isinstance(tesseract_client, Tesseract):
        raise TypeError(
            "The first argument must be a Tesseract object. "
            f"Got {type(tesseract_client)} instead."
        )

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

    # We are splitting the arguments into static and array arguments
    # This is because JAX primitives require all non-array arguments to be hashable
    # This way we can use our custom _Hashable class to wrap non-hashable args
    flat_args, input_pytreedef = jax.tree.flatten(inputs)
    is_static_mask = tuple(_is_static(arg) for arg in flat_args)
    array_args, static_args = split_args(flat_args, is_static_mask)
    static_args = tuple(_make_hashable(arg) for arg in static_args)

    if "abstract_eval" in tesseract_client.available_endpoints:
        # Get abstract values for outputs, so we can unflatten them later
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

        out = tesseract_dispatch_p.bind(
            *array_args,
            static_args=static_args,
            input_pytreedef=input_pytreedef,
            output_pytreedef=output_pytreedef,
            output_avals=flat_avals,
            is_static_mask=is_static_mask,
            client=client,
            eval_func="apply",
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
            client=client,
            eval_func="apply",
        )

        return out
