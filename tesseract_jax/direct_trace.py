# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline a Tesseract's real endpoint into the jaxpr instead of a host callback.

Applies when a Tesseract runs in-process (``Tesseract.from_tesseract_api``)
and its endpoint is pure JAX: ``apply_tesseract(..., traceable=True)`` traces
the endpoint directly via ``jax.interpreters.mlir.lower_fun``, so XLA fuses it
with the rest of the program instead of seeing an opaque custom call.

``build_direct_endpoint`` dispatches on ``params.eval_func`` to ``apply`` /
``jacobian_vector_product`` / ``vector_jacobian_product`` / ``jacobian``.
``primitive.py``'s ``jvp_rule`` / ``transpose_rule`` are unchanged -- they
still decide which endpoint a derivative uses (a Tesseract's own
``jacobian_vector_product`` may be a hand-derived, numerically stabilised
formula, not what naively differentiating ``apply`` would give); this module
only changes how the already-selected endpoint is invoked.

Each endpoint's ``inputs: InputSchema`` argument is validated for
shape/dtype only, against the same ``AbstractEval_``-prefixed sibling schema
``abstract_eval`` already uses -- pydantic's ``Array[...]`` validator calls
``np.asarray``, which raises on a JAX ``Tracer``, so the real schema can't be
constructed from traced values at all. Real values are then patched in over
the shape/dtype placeholders without re-validating.

The other arguments (path sets, the (co)tangent dict) need no schema
validation here: this module calls ``api_module.<endpoint>`` directly, so
``create_gradient_schema``'s wrapper types -- and their validators -- are
never constructed. Those arguments come from tesseract-jax's own dispatch
bookkeeping (mirroring ``Jaxeract`` in ``tesseract_compat.py``), not
user-supplied text.

An endpoint's return value is under the same constraint as its input:
building it via the schema's validating constructor (``OutputSchema(...)``)
fails the same way a traced input would. It must return a plain ``dict`` (or
build via ``model_construct``) -- the existing ``apply_jit(inputs.model_dump())``
recipe pattern -- for every endpoint, not just ``apply``.
"""

import functools
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.tree
import jax.tree_util
from jax.typing import ArrayLike
from pydantic import BaseModel

from tesseract_jax.tesseract_compat import _placeholder
from tesseract_jax.tree_util import (
    _pytree_to_tesseract_flat,
    combine_args,
    dummy_output_tree,
    split_args,
    to_shape_dtype_pytree,
    unflatten_args,
    warn_on_static_output_drift,
)

if TYPE_CHECKING:
    from tesseract_jax.dispatch_params import DispatchParams
    from tesseract_jax.tesseract_compat import Jaxeract


@functools.cache
def _extract_api_module(
    client: "Jaxeract", endpoint: str = "apply"
) -> ModuleType | None:
    """The real ``tesseract_api`` module backing an in-process Tesseract, or ``None``.

    Only a ``LocalClient`` (``Tesseract.from_tesseract_api``) has one: a
    served (``HTTPClient``) Tesseract runs in another process. ``LocalClient``
    doesn't keep the module itself, only the endpoint wrapper functions
    ``tesseract_core.runtime.core.create_endpoints`` built from it -- so it's
    recovered from the closure cell those wrappers hold it in. This depends
    on ``create_endpoints`` closing over a free variable named ``api_module``;
    if a future ``tesseract-core`` release changes that shape, this returns
    ``None`` (a loud error at the call site) rather than tracing the wrong
    thing.
    """
    local_client = getattr(client.client, "_client", None)
    endpoints = getattr(local_client, "_endpoints", None)
    if not endpoints or endpoint not in endpoints:
        return None

    func = endpoints[endpoint]
    freevars = func.__code__.co_freevars
    closure = func.__closure__
    if not closure or "api_module" not in freevars:
        return None

    api_module = closure[freevars.index("api_module")].cell_contents
    if not isinstance(api_module, ModuleType):
        return None
    return api_module


def is_traceable(client: "Jaxeract", endpoint: str = "apply") -> bool:
    """Whether ``client`` has an importable Python function to trace directly."""
    return _extract_api_module(client, endpoint) is not None


@functools.cache
def _abstract_input_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> type[BaseModel]:
    """The ``AbstractEval_``-prefixed sibling of ``InputSchema``, cached per schema pair."""
    from tesseract_core.runtime.schema_generation import create_abstract_eval_schema

    AbstractInputSchema, _ = create_abstract_eval_schema(InputSchema, OutputSchema)
    return AbstractInputSchema


def _patch_with_real_values(schema_node: Any, real_node: Any) -> Any:
    """Replace each array (``ShapeDType``) leaf with the matching real value.

    A field missing from ``real_node`` (omitted from ``apply_tesseract``'s
    raw ``inputs``, relying on ``InputSchema``'s own default) keeps
    ``schema_node``'s own value instead -- the same default the callback
    path's ``InputSchema.model_validate`` would use, since pydantic doesn't
    re-validate defaults either. A present non-array leaf also keeps
    ``schema_node``'s value, since it already passed whatever validator
    applies to it during the abstract-schema validation.
    """
    from tesseract_core.runtime.schema_types import ShapeDType

    if isinstance(schema_node, ShapeDType):
        return real_node
    if isinstance(schema_node, BaseModel):
        updates = {
            name: (
                _patch_with_real_values(getattr(schema_node, name), real_node[name])
                if isinstance(real_node, dict) and name in real_node
                else getattr(schema_node, name)
            )
            for name in type(schema_node).model_fields
        }
        return schema_node.model_copy(update=updates)
    if isinstance(schema_node, dict):
        return {
            key: (
                _patch_with_real_values(value, real_node[key])
                if isinstance(real_node, dict) and key in real_node
                else value
            )
            for key, value in schema_node.items()
        }
    if isinstance(schema_node, (list, tuple)):
        patched = [
            _patch_with_real_values(value, real_node[i])
            if isinstance(real_node, (list, tuple)) and i < len(real_node)
            else value
            for i, value in enumerate(schema_node)
        ]
        return type(schema_node)(patched)
    return schema_node


def build_direct_endpoint(
    params: "DispatchParams",
) -> Callable[..., tuple[ArrayLike, ...]]:
    """Dispatch on ``params.eval_func`` to the matching ``_build_direct_*`` builder."""
    try:
        builder = _DIRECT_TRACE_BUILDERS[params.eval_func]
    except KeyError:
        raise NotImplementedError(
            f"traceable=True does not support eval_func={params.eval_func!r}; "
            f"only {sorted(_DIRECT_TRACE_BUILDERS)} is."
        ) from None
    return builder(params)


def _abstract_inputs_schema_for(client: "Jaxeract", endpoint: str) -> tuple:
    """The real ``api_module`` and its cached ``AbstractEval_`` input schema."""
    api_module = _extract_api_module(client, endpoint)
    if api_module is None:
        raise ValueError(
            f"traceable=True requires an in-process Tesseract built via "
            f"Tesseract.from_tesseract_api(...); this client has no importable "
            f"{endpoint!r} function to trace directly."
        )
    AbstractInputSchema = _abstract_input_schema(
        api_module.InputSchema, api_module.OutputSchema
    )
    return api_module, AbstractInputSchema


def _patch_inputs(AbstractInputSchema: type[BaseModel], real_inputs: Any) -> Any:
    """Validate ``real_inputs``'s shapes/dtypes, then patch real values back in.

    ``real_inputs`` is the full (static + dynamic) input pytree, as restored
    by ``unflatten_args``; its dynamic leaves may be tracers.
    """
    abstract_inputs = to_shape_dtype_pytree(real_inputs)
    abstract_instance = AbstractInputSchema.model_validate({"inputs": abstract_inputs})
    return _patch_with_real_values(abstract_instance.inputs, real_inputs)


def _call_with_patched_inputs(endpoint_fn: Callable, **kwargs: Any) -> Any:
    """Call ``endpoint_fn(**kwargs)``, suppressing the expected schema/value mismatch warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings", category=UserWarning
        )
        return endpoint_fn(**kwargs)


def _output_flat(params: "DispatchParams", client: "Jaxeract") -> dict:
    """One entry per output leaf; differentiable paths map to a path string, others to ``None``."""
    return _pytree_to_tesseract_flat(
        dummy_output_tree(
            params.output_pytreedef, len(params.output_avals), params.static_output_mask
        ),
        schema_paths=client.differentiable_output_paths,
    )


def _build_direct_apply(
    params: "DispatchParams",
) -> Callable[..., tuple[ArrayLike, ...]]:
    """Trace the Tesseract's real ``apply`` directly.

    Takes the same flat, dynamic-leaves-only argument list as the callback
    path's dispatch closure, and returns the same flat, dynamic-leaves-only
    tuple of outputs.
    """
    api_module, AbstractInputSchema = _abstract_inputs_schema_for(
        params.client, "apply"
    )

    def direct_apply(*args: ArrayLike) -> tuple[ArrayLike, ...]:
        real_inputs = unflatten_args(
            args, params.static_args, params.input_pytreedef, params.is_static_mask
        )
        patched_inputs = _patch_inputs(AbstractInputSchema, real_inputs)
        out = _call_with_patched_inputs(api_module.apply, inputs=patched_inputs)
        out_dict = out.model_dump() if isinstance(out, BaseModel) else out

        # Mirrors Jaxeract.apply's static-output-drift check.
        static_output_mask = params.static_output_mask
        checking = params.check_static_outputs and any(static_output_mask)
        if checking:
            leaves_with_path = jax.tree_util.tree_flatten_with_path(out_dict)[0]
            flat_out = tuple(leaf for _, leaf in leaves_with_path)
        else:
            flat_out = tuple(jax.tree_util.tree_leaves(out_dict))

        if any(static_output_mask):
            flat_out, returned_statics = split_args(flat_out, static_output_mask)
            if checking:
                _, static_paths = split_args(
                    tuple(path for path, _ in leaves_with_path), static_output_mask
                )
                warn_on_static_output_drift(
                    static_paths, returned_statics, params.static_output_values
                )
        return flat_out

    return direct_apply


def _build_direct_jvp(params: "DispatchParams") -> Callable[..., tuple[ArrayLike, ...]]:
    """Trace the Tesseract's real ``jacobian_vector_product`` directly.

    Mirrors ``Jaxeract.jacobian_vector_product`` in ``tesseract_compat.py``.
    """
    client = params.client
    api_module, AbstractInputSchema = _abstract_inputs_schema_for(
        client, "jacobian_vector_product"
    )

    def direct_jvp(*args: ArrayLike) -> tuple[ArrayLike, ...]:
        has_tangent = params.has_tangent
        n_primals = params.n_primals
        primals = args[:n_primals]
        all_tangents = args[n_primals:]
        tangents = tuple(t for t, h in zip(all_tangents, has_tangent, strict=True) if h)
        n_zeros = len(primals) - sum(has_tangent)
        full_tangents = combine_args([None] * n_zeros, tangents, has_tangent)

        primal_inputs = unflatten_args(
            primals, params.static_args, params.input_pytreedef, params.is_static_mask
        )
        tangent_inputs = unflatten_args(
            full_tangents,
            params.static_args,
            params.input_pytreedef,
            params.is_static_mask,
            remove_static_args=True,
        )

        flat_tangents = _pytree_to_tesseract_flat(
            tangent_inputs, schema_paths=client.differentiable_input_paths
        )
        flat_tangents = {p: v for p, v in flat_tangents.items() if v is not None}

        output_flat = _output_flat(params, client)
        jvp_outputs = [p for p, v in output_flat.items() if v is not None]

        patched_inputs = _patch_inputs(AbstractInputSchema, primal_inputs)
        out_data = _call_with_patched_inputs(
            api_module.jacobian_vector_product,
            inputs=patched_inputs,
            jvp_inputs=list(flat_tangents.keys()),
            jvp_outputs=jvp_outputs,
            tangent_vector=flat_tangents,
        )

        out = []
        for path, aval in zip(output_flat, params.output_avals, strict=False):
            if path in out_data:
                out.append(out_data[path])
            else:
                out.append(
                    jnp.asarray(_placeholder(aval.shape, aval.dtype, on_device=False))
                )
        return tuple(out)

    return direct_jvp


def _build_direct_vjp(params: "DispatchParams") -> Callable[..., tuple[ArrayLike, ...]]:
    """Trace the Tesseract's real ``vector_jacobian_product`` directly.

    Mirrors ``Jaxeract.vector_jacobian_product`` in ``tesseract_compat.py``.
    """
    client = params.client
    api_module, AbstractInputSchema = _abstract_inputs_schema_for(
        client, "vector_jacobian_product"
    )

    def direct_vjp(*args: ArrayLike) -> tuple[ArrayLike, ...]:
        has_tangent = params.has_tangent
        n_primals = params.n_primals
        primals = args[:n_primals]
        cotangents = args[n_primals:]

        primal_inputs = unflatten_args(
            primals, params.static_args, params.input_pytreedef, params.is_static_mask
        )
        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=client.differentiable_input_paths
        )

        vjp_inputs = [
            p for p, m in zip(flat_inputs, params.is_static_mask, strict=True) if not m
        ]
        vjp_inputs = [p for p, h in zip(vjp_inputs, has_tangent, strict=True) if h]

        if any(params.static_output_mask):
            cotangents = combine_args(
                tuple(cotangents),
                (None,) * sum(params.static_output_mask),
                params.static_output_mask,
            )
        cotangent_pytree = jax.tree.unflatten(params.output_pytreedef, cotangents)
        flat_cotangents = _pytree_to_tesseract_flat(
            cotangent_pytree, schema_paths=client.differentiable_output_paths
        )
        cotangents_dict = {p: v for p, v in flat_cotangents.items() if v is not None}

        patched_inputs = _patch_inputs(AbstractInputSchema, primal_inputs)
        out_data = _call_with_patched_inputs(
            api_module.vector_jacobian_product,
            inputs=patched_inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=list(cotangents_dict.keys()),
            cotangent_vector=cotangents_dict,
        )

        out = []
        array_idx = 0
        tan_idx = 0
        for all_idx, path in enumerate(flat_inputs):
            if path in out_data:
                out.append(out_data[path])
                tan_idx += 1
            elif (
                tan_idx < len(has_tangent)
                and not params.is_static_mask[all_idx]
                and not has_tangent[tan_idx]
            ):
                arg = args[array_idx]
                out.append(
                    jnp.asarray(_placeholder(arg.shape, arg.dtype, on_device=False))
                )
                tan_idx += 1
            if not params.is_static_mask[all_idx]:
                array_idx += 1
        return tuple(out)

    return direct_vjp


def _build_direct_jacobian(
    params: "DispatchParams",
) -> Callable[..., tuple[ArrayLike, ...]]:
    """Trace the Tesseract's real ``jacobian`` directly.

    Mirrors ``Jaxeract.jacobian`` in ``tesseract_compat.py``, casting each
    block's dtype with ``jnp.asarray`` (trace-safe) instead of ``np.asarray``.
    """
    client = params.client
    api_module, AbstractInputSchema = _abstract_inputs_schema_for(client, "jacobian")

    def direct_jacobian(*args: ArrayLike) -> tuple[ArrayLike, ...]:
        n_primals = params.n_primals
        primals = args[:n_primals]

        primal_inputs = unflatten_args(
            primals, params.static_args, params.input_pytreedef, params.is_static_mask
        )
        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=client.differentiable_input_paths
        )
        jac_inputs = (
            list(params.jac_input_paths)
            if params.jac_input_paths is not None
            else [p for p, v in flat_inputs.items() if v is not None]
        )

        output_flat = _output_flat(params, client)
        jac_outputs = (
            list(params.jac_output_paths)
            if params.jac_output_paths is not None
            else [p for p, v in output_flat.items() if v is not None]
        )

        patched_inputs = _patch_inputs(AbstractInputSchema, primal_inputs)
        out_data = _call_with_patched_inputs(
            api_module.jacobian,
            inputs=patched_inputs,
            jac_inputs=jac_inputs,
            jac_outputs=jac_outputs,
        )

        ip_to_dtype = {p: v.dtype for p, v in flat_inputs.items() if v is not None}
        op_to_dtype = {
            p: aval.dtype
            for (p, v), aval in zip(
                output_flat.items(), params.output_avals, strict=True
            )
            if v is not None
        }
        out = []
        for op in jac_outputs:
            for ip in jac_inputs:
                target = (
                    ip_to_dtype[ip] if params.jac_mode == "bwd" else op_to_dtype[op]
                )
                out.append(jnp.asarray(out_data[op][ip], dtype=target))
        return tuple(out)

    return direct_jacobian


_DIRECT_TRACE_BUILDERS: dict[
    str, Callable[["DispatchParams"], Callable[..., tuple[ArrayLike, ...]]]
] = {
    "apply": _build_direct_apply,
    "jacobian_vector_product": _build_direct_jvp,
    "vector_jacobian_product": _build_direct_vjp,
    "jacobian": _build_direct_jacobian,
}
