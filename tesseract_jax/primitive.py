# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import os
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import jax
import jax.core as jc
import jax.numpy as jnp
import jax.tree
import numpy as np
from jax import dtypes, extend
from jax._src import dispatch
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike
from tesseract_core import Tesseract

from tesseract_jax.batching import VMAP_METHOD_DISPATCH, VmapMethod
from tesseract_jax.dispatch_params import DispatchParams
from tesseract_jax.tesseract_compat import Jaxeract
from tesseract_jax.tree_util import (
    TransportArray,
    _pytree_to_tesseract_flat,
    combine_args,
    dummy_output_tree,
    split_args,
    unflatten_args,
)

tesseract_dispatch_p = extend.core.Primitive("tesseract_dispatch")
tesseract_dispatch_p.multiple_results = True


CHECK_STATIC_OUTPUTS_ENV_VAR = "TESSERACT_JAX_CHECK_STATIC_OUTPUTS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable, or fall back to ``default``.

    Read per call rather than once at import, so a program can change the
    variable at runtime.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{name} is set to {raw!r}, which is not a boolean. "
        f"Use one of 1/0, true/false, yes/no, on/off."
    )


def _instantiate_zeros(tangents: Sequence[Any]) -> tuple[ArrayLike, ...]:
    """Densify symbolic zeros, which cannot be passed to ``bind`` as-is."""
    return tuple(
        jax.numpy.zeros_like(t.aval) if isinstance(t, jax._src.ad_util.Zero) else t
        for t in tangents
    )


@tesseract_dispatch_p.def_abstract_eval
def tesseract_dispatch_abstract_eval(
    *array_args: ArrayLike | ShapedArray,
    params: DispatchParams,
) -> tuple:
    """Define how to dispatch evals and pipe arguments."""
    if params.eval_func not in (
        "apply",
        "jacobian_vector_product",
        "vector_jacobian_product",
        "jacobian",
    ):
        raise NotImplementedError(params.eval_func)

    n_primals = params.n_primals

    if params.eval_func == "vector_jacobian_product":
        # A VJP output has the same shape as the primal it differentiates, so we
        # can read the shapes off the primals without a forward evaluation. Only
        # differentiated primals (has_tangent) carry a cotangent back; a
        # non-differentiated input's slot is never consumed by JAX's transpose,
        # so we omit it entirely rather than return a placeholder for it.
        return tuple(
            aval
            for aval, h in zip(array_args[:n_primals], params.has_tangent, strict=True)
            if h
        )

    if params.eval_func == "jacobian":
        # One array per (diff_output, diff_input) pair, shape = out_shape + in_shape.
        # `jac_input_paths` / `jac_output_paths` (when provided) restrict the
        # request to a sub-block of the Jacobian.
        primal_avals = array_args[:n_primals]
        primal_inputs = unflatten_args(
            primal_avals,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
        )
        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=params.client.differentiable_input_paths
        )
        path_to_shape = {
            p: (tuple(v.shape), v.dtype)
            for p, v in flat_inputs.items()
            if v is not None
        }
        output_flat = _pytree_to_tesseract_flat(
            dummy_output_tree(
                params.output_pytreedef,
                len(params.output_avals),
                params.static_output_mask,
            ),
            schema_paths=params.client.differentiable_output_paths,
        )
        out_path_to_aval = {
            path: aval
            for (path, v), aval in zip(
                output_flat.items(), params.output_avals, strict=True
            )
            if v is not None
        }
        jac_inputs = (
            list(params.jac_input_paths)
            if params.jac_input_paths is not None
            else list(path_to_shape.keys())
        )
        jac_outputs = (
            list(params.jac_output_paths)
            if params.jac_output_paths is not None
            else list(out_path_to_aval.keys())
        )
        # Per JAX convention: fwd-mode → output dtype (jacfwd), bwd-mode →
        # input dtype (jacrev / `jax.jacobian`). Mirrors lineax's mode names.
        avals_out = []
        for op_path in jac_outputs:
            out_aval = out_path_to_aval[op_path]
            for ip in jac_inputs:
                in_shape, in_dtype = path_to_shape[ip]
                dtype = in_dtype if params.jac_mode == "bwd" else out_aval.dtype
                avals_out.append(
                    jax.core.ShapedArray(tuple(out_aval.shape) + in_shape, dtype)
                )
        return tuple(avals_out)

    # Those have the same shape as the outputs
    assert params.eval_func in ("apply", "jacobian_vector_product")
    return tuple(
        jax.core.ShapedArray(aval.shape, aval.dtype) for aval in params.output_avals
    )


def tesseract_dispatch_jvp_rule(
    in_args: tuple[ArrayLike, ...],
    tan_args: tuple[ArrayLike | ad.Zero, ...],
    params: DispatchParams,
) -> tuple[tuple[ArrayLike, ...], tuple[ArrayLike, ...]]:
    """Defines how to dispatch jvp operation.

    Note this function is also called when evaluating a VJP or doing
    reverse-mode autodiff.

    """
    if params.eval_func not in (
        "apply",
        "jacobian_vector_product",
        "vector_jacobian_product",
    ):
        raise RuntimeError(
            f"Cannot take higher-order derivatives of {params.eval_func!r}"
        )

    #  https://github.com/jax-ml/jax/issues/16303#issuecomment-1585295819
    #  mattjj: taking a narrow pigeon-holed view, anywhere you see a symbolic
    #          zero `Zero(AbstractToken)`, i.e. in a JVP or transpose rule
    #          (not in ad.py's backward_pass), you probably want to instantiate
    #          it so that it's no longer symbolic

    n_primals = params.n_primals
    has_tangent = params.has_tangent

    if params.eval_func == "apply":
        # Compute which primals have non-zero tangents
        has_tangent = tuple(not isinstance(t, jax._src.ad_util.Zero) for t in tan_args)

        # Raise if a non-symbolic-zero tangent is provided for a non-differentiable input.
        _tangents_for_check = tuple(
            t if h else None for t, h in zip(tan_args, has_tangent, strict=True)
        )
        _tangent_inputs = unflatten_args(
            _tangents_for_check,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
            remove_static_args=True,
        )
        _flat_tangents = _pytree_to_tesseract_flat(
            _tangent_inputs, schema_paths=params.client.differentiable_input_paths
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
        # Differentiating `apply` means differentiating wrt its primals.
        tan_args_ = _instantiate_zeros(tan_args)
    else:
        # A derivative endpoint is linear in its (co)tangent slots, so its JVP is
        # that same endpoint at the new (co)tangents. Its primals are another
        # matter: that needs a second derivative, which Tesseract does not expose.
        if not all(isinstance(t, jax._src.ad_util.Zero) for t in tan_args[:n_primals]):
            raise RuntimeError(
                "Cannot differentiate a Tesseract derivative endpoint with respect "
                "to its primal inputs, as this needs a second derivative."
            )
        tan_args_ = _instantiate_zeros(tan_args[n_primals:])

    # `has_tangent` describes a bind's own (co)tangent operands, so the derivative
    # bind below needs one for `tan_args_` rather than the mask it inherited. Only
    # a `jacobian_vector_product` can have one: its linear slots are tangents, one
    # per primal, which is how `has_tangent` is indexed. A
    # `vector_jacobian_product`'s are cotangents, one per output, so they cannot
    # size such a mask and it keeps the inherited value -- as does `res`, which
    # reproduces the original call over the original operands.
    #
    # Not cosmetic: the batching rule turns `has_tangent` into `jac_input_paths`,
    # i.e. which columns of the Jacobian get requested, so an inherited mask
    # over-fetches whenever only some arguments are differentiated.
    # `jacfwd(lin_fn, argnums=0)` would ask for every column and then multiply the
    # unwanted ones by the zeros instantiated above.
    deriv_has_tangent = has_tangent
    if params.eval_func == "jacobian_vector_product":
        deriv_has_tangent = tuple(
            not isinstance(t, jax._src.ad_util.Zero) for t in tan_args[n_primals:]
        )

    # this leads to an abstract_eval call and a jvp
    jvp = tesseract_dispatch_p.bind(
        *in_args[:n_primals],
        *tan_args_,
        params=params.replace(
            has_tangent=deriv_has_tangent,
            eval_func=(
                "vector_jacobian_product"
                if params.eval_func == "vector_jacobian_product"
                else "jacobian_vector_product"
            ),
            jac_input_paths=None,
            jac_output_paths=None,
            jac_mode="bwd",
        ),
    )

    res = tesseract_dispatch_p.bind(
        *in_args,
        params=params.replace(
            has_tangent=has_tangent,
            jac_input_paths=None,
            jac_output_paths=None,
            jac_mode="bwd",
        ),
    )

    return tuple(res), tuple(jvp)


ad.primitive_jvps[tesseract_dispatch_p] = tesseract_dispatch_jvp_rule


def tesseract_dispatch_transpose_rule(
    cotangent: Sequence[ArrayLike | ad.Zero],
    *args: ArrayLike | ad.UndefinedPrimal,
    params: DispatchParams,
) -> tuple[ArrayLike | None, ...]:
    """Defines how to dispatch vjp operation."""
    assert params.eval_func in ("jacobian_vector_product",)

    n_primals = params.n_primals
    primal_args = args[:n_primals]

    # Primal slots must hold concrete residuals. An UndefinedPrimal here means the
    # caller asked us to transpose with respect to a primal -- J(x)·v is linear in
    # the tangent v but not in x, so no endpoint can serve it. JAX itself refuses
    # the equivalent transpose, so bail with guidance instead of letting the
    # UndefinedPrimal fall through into the checks below.
    if any(ad.is_undefined_primal(p) for p in primal_args):
        raise ValueError(
            "Transpose of the Tesseract primitive requires concrete primal "
            "values, but received UndefinedPrimal. This typically happens when "
            "jax.linear_transpose is applied to a function whose arguments "
            "include both primals and tangents. Close over the primals instead, "
            "e.g.:\n"
            "  primals = (x,)\n"
            "  jax.linear_transpose(lambda t: jax.jvp(f, primals, (t,))[1], x)"
        )

    # Raise if a cotangent for a non-differentiable output is not a symbolic zero.
    # Symbolic zeros (ad.Zero) are produced by JAX when gradients are blocked
    # (e.g. via jax.lax.stop_gradient) or when the output is not used in the loss.
    # Any other cotangent means the user accidentally included a non-diff output
    # in the gradient computation, likely due to a missing Differentiable[] annotation.
    dummy_output = dummy_output_tree(
        params.output_pytreedef,
        len(params.output_avals),
        params.static_output_mask,
    )
    flat_output_info = _pytree_to_tesseract_flat(
        dummy_output, schema_paths=params.client.differentiable_output_paths
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
        primal_args,
        params.static_args,
        params.input_pytreedef,
        params.static_input_mask,
    )
    _flat_inputs = _pytree_to_tesseract_flat(
        _primal_inputs, schema_paths=params.client.differentiable_input_paths
    )
    _non_static_paths = [
        p for p, m in zip(_flat_inputs, params.static_input_mask, strict=True) if not m
    ]
    _vjp_inputs_with_tangent = [
        p for p, h in zip(_non_static_paths, params.has_tangent, strict=True) if h
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

    # An output whose cotangent is a symbolic zero adds nothing to the input
    # gradients, so record which outputs carry a real cotangent and skip the rest
    # when calling the endpoint. This also stops a NaN in an unused output's
    # gradient from poisoning the result. ``cotangent`` is already the non-static
    # outputs.
    has_cotangent = tuple(not isinstance(c, jax._src.ad_util.Zero) for c in cotangent)

    # Pass only the real cotangents to the bind. The symbolic-zero ones would be
    # instantiated to dense zeros the endpoint never reads, so drop them from the
    # operands entirely; the endpoint call scatters the survivors back to full
    # output width via ``has_cotangent`` (see ``Jaxeract.vector_jacobian_product``).
    cotan_args_ = tuple(c for c, h in zip(cotangent, has_cotangent, strict=True) if h)

    vjp = tesseract_dispatch_p.bind(
        *primal_args,
        *cotan_args_,
        params=params.replace(
            eval_func="vector_jacobian_product", has_cotangent=has_cotangent
        ),
    )

    # The bind returns a cotangent only for each differentiated primal
    # (has_tangent). Scatter them back into full primal order, leaving None where
    # no cotangent flows -- JAX reads None as a symbolic zero for that operand.
    vjp_iter = iter(vjp)
    input_cotangents = [next(vjp_iter) if h else None for h in params.has_tangent]
    return tuple([None] * len(primal_args) + input_cotangents)


ad.primitive_transposes[tesseract_dispatch_p] = tesseract_dispatch_transpose_rule


def _raise_if_unimplemented(eval_func: str, client: Jaxeract) -> None:
    if eval_func not in client.available_methods:
        raise NotImplementedError(
            f"Endpoint '{eval_func}' not implemented for this Tesseract. "
            f"Available endpoints: {', '.join(client.available_methods)}. "
            f"To use this endpoint, implement the '{eval_func}' endpoint in your Tesseract object."
        )


# An eager ``bind`` traces the primitive to a jaxpr, compiles it and runs the
# compiled program, exactly as a call under ``jit`` would. This routes eager and
# traced calls through the same ``abstract_eval`` + lowering path, so a Tesseract
# behaves identically in both. It also means eager use requires an
# ``abstract_eval`` endpoint, which ``apply_tesseract`` checks up front.
tesseract_dispatch_p.def_impl(partial(dispatch.apply_primitive, tesseract_dispatch_p))


def _build_dispatch_closure(params: DispatchParams) -> Callable[..., tuple]:
    """Build the endpoint dispatch closure shared by the CPU and GPU lowerings.

    Returns ``dispatch(*args) -> tuple`` calling ``getattr(params.client,
    params.eval_func)(args, params)``. The CPU lowering runs it via a host
    callback; the GPU lowering runs it via the native FFI handler. Dispatching by
    ``eval_func`` keeps every endpoint transport-agnostic.
    """

    def dispatch(*args: TransportArray) -> tuple:
        out = getattr(params.client, params.eval_func)(args, params)
        if not isinstance(out, tuple):
            out = (out,)
        return out

    return dispatch


def tesseract_dispatch_lowering(
    ctx: Any,
    *array_args: ArrayLike | ShapedArray | Any,
    params: DispatchParams,
) -> Any:
    """CPU lowering: run the dispatch closure via a host callback."""
    _raise_if_unimplemented(params.eval_func, params.client)

    dispatch = _build_dispatch_closure(params)

    # A Tesseract endpoint is a pure function of its inputs, so declare it as one.
    # This is what lets XLA's CSE fold repeated identical calls into a single
    # request -- the same treatment a LAPACK custom call such as ``lu_factor``
    # already gets. It relies on the bind params comparing equal for identical
    # calls, which is why ``Jaxeract`` defines ``__eq__`` / ``__hash__``.
    #
    # Marking the callback side-effecting would *not* buy a guarantee that it
    # always runs: JAX's own DCE already drops a Tesseract call whose outputs are
    # unused, before XLA ever sees it. All it did was suppress CSE and pin the
    # call's order against other effects.
    result, _, keepalive = mlir.emit_python_callback(
        ctx,
        dispatch,
        None,
        array_args,
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=False,
    )
    ctx.module_context.add_keepalive(keepalive)
    return result


def tesseract_dispatch_gpu_lowering(
    ctx: Any,
    *array_args: ArrayLike | ShapedArray | Any,
    params: DispatchParams,
) -> Any:
    """GPU lowering: run the dispatch closure via the native FFI handler.

    Falls back to the host-callback lowering when the caller did not select a
    device transport (``client._device_transport``), so a host-transport call
    behaves exactly as on CPU. When the caller did select one but the native shim
    is unavailable (e.g. a CPU-only install where it wasn't compiled), this raises
    rather than silently falling back, since ``device_transport`` is an explicit
    opt-in to the GPU-direct path.
    """
    from tesseract_jax import gpu_ffi

    client = params.client

    if client._device_transport is None:
        return tesseract_dispatch_lowering(ctx, *array_args, params=params)

    if not gpu_ffi.is_available():
        raise RuntimeError(
            f"device_transport={client._device_transport!r} was requested but "
            "the native GPU FFI shim is unavailable (not compiled or failed to "
            "import), so GPU-direct dispatch cannot run. Reinstall tesseract-jax "
            "with the shim built (a source install compiles it via the hatch "
            "build hook; set TESSERACT_JAX_GPU_REQUIRED=1 to make a build failure "
            "fatal), or drop device_transport to use the host-callback transport."
        )

    # Every supported device transport is CUDA-based, so this lowering cannot run
    # without a CUDA device. Reaching here means the caller selected a transport
    # and the program is being lowered for the GPU, so a missing device is a
    # misconfiguration worth raising over rather than the (much slower) host path.
    try:
        jax.devices("cuda")
    except RuntimeError as exc:
        raise RuntimeError(
            f"device_transport={client._device_transport!r} was requested but "
            "JAX sees no CUDA device. Install a CUDA-enabled jaxlib and run on a "
            "GPU host, or drop device_transport to use the host-callback transport."
        ) from exc

    _raise_if_unimplemented(params.eval_func, client)

    inner = _build_dispatch_closure(params)

    # Run the dispatch with the client in device-transport mode, so GPU inputs
    # are exported by reference and outputs come back on-device.
    def gpu_dispatch(args: tuple) -> tuple:
        with client.device_transport_encoding():
            return inner(*args)

    target = gpu_ffi.ensure_registered()
    # The token must outlive lowering (the FFI call reads it at execution time),
    # so it is never released. Keying on ``params`` -- a frozen, value-equal
    # DispatchParams -- means re-lowering the same dispatch (a re-trace, cache
    # eviction, or fresh jit) reuses one entry instead of leaking a fresh closure
    # (and the Jaxeract/client/session it pins) each time.
    token = gpu_ffi.register_dispatch(gpu_dispatch, key=params)

    rule = jax.ffi.ffi_lowering(target)
    return rule(ctx, *array_args, token=np.int64(token))


mlir.register_lowering(tesseract_dispatch_p, tesseract_dispatch_lowering)
mlir.register_lowering(
    tesseract_dispatch_p, tesseract_dispatch_gpu_lowering, platform="cuda"
)


def tesseract_dispatch_batching(
    array_args: ArrayLike | ShapedArray | Any,
    axes: Sequence[Any],
    *,
    params: DispatchParams,
) -> Any:
    """Defines how to dispatch batch operations such as vmap (which is used by jax.jacobian)."""
    _raise_if_unimplemented(params.eval_func, params.client)

    # An unmapped axis is denoted by ``None``. (Older JAX versions exposed this
    # as the ``batching.not_mapped`` sentinel, which has since been removed; it
    # was always equal to ``None``.)
    n_primals = params.n_primals

    # When jacfwd/jacrev vmap a JVP/VJP with primals unbatched and (co)tangents
    # batched, materialize the entire Jacobian and apply to batch with matmul.
    # Gated by ``materialize_jacobian`` (None = auto, True = force, False = skip).
    if params.eval_func in ("jacobian_vector_product", "vector_jacobian_product"):
        primal_axes = axes[:n_primals]
        tangent_axes = axes[n_primals:]
        primals_unbatched = all(ax is None for ax in primal_axes)
        tangents_batched = any(ax is not None for ax in tangent_axes)
        endpoint_available = "jacobian" in params.client.available_methods
        if params.materialize_jacobian is True and not endpoint_available:
            raise RuntimeError(
                "materialize_jacobian=True but the Tesseract does not expose a "
                "'jacobian' endpoint."
            )
        use_shortcut = (
            primals_unbatched
            and tangents_batched
            and params.materialize_jacobian is not False
            and (params.materialize_jacobian is True or endpoint_available)
        )
        if use_shortcut:
            return _batched_via_jacobian(array_args, axes, params=params)

    new_args = [
        arg if ax is None else jnp.moveaxis(arg, ax, 0)
        for arg, ax in zip(array_args, axes, strict=True)
    ]
    is_batched_mask = [ax is not None for ax in axes]

    if params.eval_func == "jacobian":
        # The vectorized strategies hand batched primals to the endpoint, which
        # then answers with a (batch, *out, batch, *in) block rather than
        # (batch, *out, *in). Only sequential wraps a jacobian call correctly,
        # and the mismatch is silent rather than an error, so pin it here.
        params = params.replace(vmap_method="sequential")

    batch_fn = VMAP_METHOD_DISPATCH[params.vmap_method]
    return batch_fn(
        new_args,
        is_batched_mask,
        params=params,
        tesseract_dispatch_p=tesseract_dispatch_p,
    )


def _batched_via_jacobian(
    array_args: Sequence[Any],
    axes: Sequence[Any],
    *,
    params: DispatchParams,
) -> tuple[tuple, tuple]:
    """Batched JVP / VJP via one ``jacobian`` endpoint call + ``tensordot``.

    Assumes primals are unbatched and at least one (co)tangent is batched
    (caller checks). Returns ``(out_vals, out_axes)`` per JAX batching rule
    convention; output batch axis is always 0.
    """
    n_primals = params.n_primals
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
        primals, params.static_args, params.input_pytreedef, params.static_input_mask
    )
    flat_inputs = _pytree_to_tesseract_flat(
        primal_inputs, schema_paths=params.client.differentiable_input_paths
    )
    output_flat = _pytree_to_tesseract_flat(
        dummy_output_tree(
            params.output_pytreedef,
            len(params.output_avals),
            params.static_output_mask,
        ),
        schema_paths=params.client.differentiable_output_paths,
    )

    # Map each (schema-diff) ∧ (has_tangent) input path to its position in
    # ``tans``. ``keys()`` give the path order of the Jacobian's columns;
    # ``values()`` give the corresponding ``tans`` positions. Excludes
    # JAX-zero tangents (JVP) and non-requested gradients (VJP).
    diff_input_path_to_pos: dict[str, int] = {}
    non_static_idx = 0
    for (p, v), is_static in zip(
        flat_inputs.items(), params.static_input_mask, strict=True
    ):
        if is_static:
            continue
        if v is not None and params.has_tangent[non_static_idx]:
            diff_input_path_to_pos[p] = non_static_idx
        non_static_idx += 1

    # Map each diff output path to its leaf index in the output pytree.
    # ``keys()`` give the path order of the Jacobian's rows; ``values()`` give
    # the corresponding ``tans`` / ``output_avals`` positions, dropping unrequested
    # outputs on the reverse-mode path (signalled by ``has_cotangent``).
    is_jvp = params.eval_func == "jacobian_vector_product"
    diff_output_path_to_pos: dict[str, int] = {
        p: i
        for i, (p, v) in enumerate(output_flat.items())
        if v is not None and (is_jvp or params.has_cotangent[i])
    }

    jac_arrays = tesseract_dispatch_p.bind(
        *primals,
        params=params.replace(
            eval_func="jacobian",
            jac_input_paths=tuple(diff_input_path_to_pos),
            jac_output_paths=tuple(diff_output_path_to_pos),
            jac_mode="fwd" if params.eval_func == "jacobian_vector_product" else "bwd",
        ),
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

        Non-diff output tangents are NaN so the batched JVP matches the
        sequential one (see ``_discarded_slot``).
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

    if params.eval_func == "jacobian_vector_product":
        # Filter tangents to matrix-column order, then for each diff row sum
        # the per-column matvec contributions.
        filtered_tans = [tans[pos] for pos in diff_input_path_to_pos.values()]
        diff_avals = [params.output_avals[i] for i in diff_output_path_to_pos.values()]
        diff_outs = jax.tree.map(
            lambda slot, jac_row: _tree_sum(
                jax.tree.map(_matmul, jac_row, filtered_tans), slot
            ),
            diff_avals,
            jac_blocks,
        )
        outs = _pad_nans(
            diff_outs,
            params.output_avals,
            [v is not None for _p, v in output_flat.items()],
        )
        return outs, (0,) * len(outs)

    # VJP: transpose the outer/inner list structure to iterate
    # by column (one per diff input). ``tans`` is already the reduced cotangent
    # tuple (only ``has_cotangent``-True outputs, symbolic zeros dropped at the
    # bind), in the same order as the surviving rows of ``diff_output_path_to_pos``,
    # so take them straight through.
    jac_cols = [list(col) for col in zip(*jac_blocks, strict=True)]
    diff_primals = [primals[pos] for pos in diff_input_path_to_pos.values()]
    diff_grads = jax.tree.map(
        lambda slot, jac_col: _tree_sum(
            jax.tree.map(_rmatmul, jac_col, list(tans)), slot
        ),
        diff_primals,
        jac_cols,
    )
    # A VJP bind returns a gradient only for each differentiated primal; the
    # abstract_eval declares that shorter arity. ``diff_input_path_to_pos`` is
    # built in primal positional order, so ``diff_grads`` is already emitted in
    # that order, with the non-diff slots that JAX's transpose never consumes
    # omitted.
    grads = tuple(diff_grads)
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
    materialize_jacobian: bool | None = None,
    device_transport: str | None = None,
    check_static_outputs: bool | None = None,
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
        materialize_jacobian: Strategy for batching (co)tangents at a single
            primal evaluation (e.g. ``jacfwd``, ``jacrev``,
            ``vmap(lambda t: jvp(f, (x,), (t,)))``,
            ``vmap(vjp(f, x)[1])``, ``vmap(linearize(f, x)[1])`` ...).

            ``True``
                Call the Tesseract's ``jacobian`` endpoint directly and use a
                PyTree-compatible matmul to compute its action on the batch of
                (co)tangents.

            ``False``
                Batch ``jacobian_vector_product`` according to ``vmap_method`` (e.g.
                ``vmap_method=auto_experimental`` would broadcast primals to match tangents).
                If ``vmap_method`` is not ``None``, ``vector_jacobian_product`` will always
                be called sequentially once per cotangent. ⚠️ If ``vmap_method=None``
                and ``materialize_jacobian=False`` then batched (co)tangents are not supported
                (a ``NotImplementedError`` will be raised).

            ``None`` (default)
                Auto: use ``True`` when the Tesseract has a ``jacobian``
                endpoint, else fall back to ``False``.

            When batching over a large number of (co)tangents or
            if you need access to the entire Jacobian through
            ``jacfwd``/``jacrev`` the default of using the ``jacobian``
            endpoint should be much more performant. However, if your Jacobian
            is large and you are batching over a small number of (co)tangents
            (e.g. to perform low-rank approximations or apply coloring
            methods) ``False`` may be more efficient.
        device_transport: Name of the on-device transport used to exchange GPU
            arrays with the Tesseract instead of a host round-trip (currently
            ``"cuda_ipc"``). Requires a served Tesseract (``HTTPClient``) started
            with the matching ``gpu_transport`` in its ``runtime_config`` and a
            GPU-backed JAX (arrays on a ``cuda`` device); has no effect on CPU
            arrays or a local (in-process) client, which already shares memory.
            For ``cuda_ipc`` both processes must share the CUDA IPC namespace
            (Docker's ``--ipc=host``). When ``None`` (default), GPU arrays take
            the same host round-trip as CPU arrays. This is an experimental
            tesseract-core feature.
        check_static_outputs: Whether to compare the non-array outputs ``apply``
            returns against the ones ``abstract_eval`` reported, and warn on any
            that differ. The value the caller gets is the one from
            ``abstract_eval`` either way, since static outputs are read at trace
            time. ``None`` (default) reads ``TESSERACT_JAX_CHECK_STATIC_OUTPUTS``,
            which is on unless set to a false value. Pass ``False`` to skip the
            comparison for one call. Skipping it also skips building the keypaths
            the warning needs; the caller gets the same values either way.

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

    # Every call is dispatched through the primitive (see the def_impl note
    # above), which needs abstract_eval to report the output shapes, so a
    # Tesseract without that endpoint cannot be used here at all.
    if "abstract_eval" not in tesseract_client.available_endpoints:
        raise ValueError(
            "Tesseract object does not support abstract_eval, "
            "tesseract-jax requires it to determine the output shapes of a call. "
            "Add an abstract_eval endpoint to the Tesseract object, or call it "
            "directly through the Tesseract client instead of apply_tesseract."
        )

    client = Jaxeract(tesseract_client, device_transport=device_transport)

    flat_args, input_pytreedef = jax.tree.flatten(inputs)
    # Arrays -- concrete or traced -- are operands of the primitive; only genuine
    # non-array leaves (a str, an int, a bool) are static. Treating a concrete
    # array as static would close it over as a bind parameter, diverging from the
    # traced path and forcing a recompile whenever its value changes.
    static_input_mask = tuple(
        not isinstance(arg, (jax.Array, np.ndarray)) for arg in flat_args
    )
    array_args, static_args = split_args(flat_args, static_input_mask)
    has_tangent = (True,) * len(array_args)

    # abstract_eval's output structure tells us how to unflatten the arrays the
    # primitive returns.
    avals = client.abstract_eval(inputs)

    if check_static_outputs is None:
        check_static_outputs = _env_flag(CHECK_STATIC_OUTPUTS_ENV_VAR, True)

    is_aval = lambda x: isinstance(x, dict) and "dtype" in x and "shape" in x
    avals_with_path, output_pytreedef = jax.tree_util.tree_flatten_with_path(
        avals, is_leaf=is_aval
    )
    # An OutputSchema may carry non-array fields alongside its arrays, such as a
    # backend name or a convergence flag. A JAX primitive can only return arrays,
    # so those leaves never enter the bind. They are read from abstract_eval,
    # held aside as static primitive parameters, and put back into the output
    # pytree once the bind has returned.
    static_output_mask = tuple(not is_aval(aval) for _, aval in avals_with_path)
    static_output_values = tuple(
        aval
        for (_, aval), static in zip(avals_with_path, static_output_mask, strict=True)
        if static
    )

    for _path, aval in avals_with_path:
        if is_aval(aval):
            _check_dtype(aval["dtype"])

    flat_avals = tuple(
        jax.ShapeDtypeStruct(shape=tuple(aval["shape"]), dtype=aval["dtype"])
        for (_, aval), static in zip(avals_with_path, static_output_mask, strict=True)
        if not static
    )

    # Apply the primitive
    out = tesseract_dispatch_p.bind(
        *array_args,
        params=DispatchParams(
            static_args=static_args,
            input_pytreedef=input_pytreedef,
            output_pytreedef=output_pytreedef,
            output_avals=flat_avals,
            static_output_mask=static_output_mask,
            static_output_values=static_output_values,
            check_static_outputs=check_static_outputs,
            static_input_mask=static_input_mask,
            has_tangent=has_tangent,
            client=client,
            eval_func="apply",
            vmap_method=vmap_method,
            materialize_jacobian=materialize_jacobian,
        ),
    )

    # Put the static leaves back where the schema had them.
    if any(static_output_mask):
        out = combine_args(
            tuple(out),
            static_output_values,
            static_output_mask,
        )

    return jax.tree.unflatten(output_pytreedef, out)
