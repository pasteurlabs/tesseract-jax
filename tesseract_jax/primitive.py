# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Sequence
from typing import Any, TypeVar

import jax.numpy as jnp
import jax.tree
import numpy as np
from jax import ShapeDtypeStruct, dtypes, extend
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from tesseract_core import Tesseract

from tesseract_jax.tesseract_compat import Jaxeract, combine_args

T = TypeVar("T")

tesseract_dispatch_p = extend.core.Primitive("tesseract_dispatch")
tesseract_dispatch_p.multiple_results = True
tesseract_dispatch_p.def_impl(
    functools.partial(xla.apply_primitive, tesseract_dispatch_p)
)


class _Hashable:
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
    """Define how to dispatch evals and pipe arguments."""
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

    tan_args_ = tuple(
        (
            jax.numpy.zeros_like(arg.aval)
            if isinstance(arg, jax._src.ad_util.Zero)
            else arg
        )
        for arg in tan_args
    )

    jvp = tesseract_dispatch_p.bind(
        *in_args,
        *tan_args_,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
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

    cotan_args_ = tuple(
        (
            jax.numpy.zeros_like(arg.aval)
            if isinstance(arg, jax._src.ad_util.Zero)
            else arg
        )
        for arg in cotangent
    )

    vjp = tesseract_dispatch_p.bind(
        *args,
        *cotan_args_,
        static_args=static_args,
        input_pytreedef=input_pytreedef,
        output_pytreedef=output_pytreedef,
        output_avals=output_avals,
        is_static_mask=is_static_mask,
        client=client,
        eval_func="vector_jacobian_product",
    )
    # TODO: I'm not sure this makes sense given these docs:
    #       https://jax.readthedocs.io/en/latest/jax-primitives.html#transposition
    #       "A tuple with the cotangent of the inputs, with the value None corresponding to the constant arguments"
    #       ...but if I provide only cotangent, jax complains, and if I investigate its internals,
    #       I see it chokes on map(partial(write_cotangent, eqn.primitive), eqn.invars, cts_out),
    #       where eqn.invars ends up being longer than cts_out.

    return tuple([None] * len(args) + vjp)


ad.primitive_transposes[tesseract_dispatch_p] = tesseract_dispatch_transpose_rule


def tesseract_dispatch_lowering(
    ctx: Any,
    *array_args: ArrayLike | ShapedArray | Any,
    static_args: tuple[_Hashable, ...],
    input_pytreedef: PyTreeDef,
    output_pytreedef: PyTreeDef | None,
    output_avals: tuple[ShapeDtypeStruct, ...] | None,
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

    # if "abstract_eval" not in tesseract_client.available_endpoints:
    #     raise ValueError(
    #         "Given Tesseract object does not support abstract_eval, "
    #         "which is required for compatibility with JAX."
    #     )

    client = Jaxeract(tesseract_client)

    flat_args, input_pytreedef = jax.tree.flatten(inputs)
    is_static_mask = tuple(_is_static(arg) for arg in flat_args)
    array_args, static_args = split_args(flat_args, is_static_mask)
    static_args = tuple(_make_hashable(arg) for arg in static_args)

   
    # Get abstract values for outputs, so we can unflatten them later
    output_pytreedef, avals, flat_avals = None, None, None
    if "abstract_eval" in tesseract_client.available_endpoints:
        avals = client.abstract_eval(
            array_args,
            static_args,
            input_pytreedef,
            output_pytreedef,
            avals,
            is_static_mask,
        )

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
        client=client,
        eval_func="apply",
    )

    # else:
    #     # Apply the primitive
    #     out = tesseract_dispatch_p.bind(
    #         *array_args,
    #         static_args=static_args,
    #         input_pytreedef=input_pytreedef,
    #         is_static_mask=is_static_mask,
    #         client=client,
    #         eval_func="apply",
    #     )

    # Unflatten the output
    return jax.tree.unflatten(output_pytreedef, out)
