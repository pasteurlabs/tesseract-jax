# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace a Tesseract's real dispatch endpoints instead of dispatching them.

``traced_tesseract(tesseract_client)`` returns a shallow copy of
``tesseract_client`` whose ``LocalClient`` is replaced by ``TracedClient``.
``Tesseract``'s own ``apply``/``jacobian_vector_product``/
``vector_jacobian_product``/``jacobian`` methods are thin forwarders to
``self._client.run_tesseract(endpoint, payload, ...)`` -- swapping only
``_client`` means every one of them, plus ``openapi_schema`` /
``available_endpoints`` (which route through ``run_tesseract`` too), keeps
working unchanged; only the four differentiable-dispatch endpoints are
actually intercepted, everything else delegates to the real ``LocalClient``.
Feeding this into ``Jaxeract`` and lowering the resulting dispatch closure
with ``jax.interpreters.mlir.lower_fun`` instead of
``mlir.emit_python_callback`` gets ``apply_tesseract(..., traceable=True)``
for free: every endpoint's path/tangent/placeholder bookkeeping is
``Jaxeract``'s existing, already-tested code (``tesseract_compat.py``),
unchanged. ``TracedClient`` itself only does the one genuinely new thing --
validate and call the real Python endpoint directly.

An endpoint's ``inputs: InputSchema`` argument is validated for shape/dtype
only, against the same ``AbstractEval_``-prefixed sibling schema
``abstract_eval`` already uses -- pydantic's ``Array[...]`` validator calls
``np.asarray``, which raises on a JAX ``Tracer``, so the real schema can't be
constructed from traced values at all. Real values are patched in over the
shape/dtype placeholders afterwards, without re-validating.

An endpoint's return value is under the same constraint as its input:
building it via the schema's validating constructor (``OutputSchema(...)``)
fails the same way a traced input would. It must return a plain ``dict`` (or
build via ``model_construct``) -- the existing ``apply_jit(inputs.model_dump())``
recipe pattern -- for every endpoint, not just ``apply``.
"""

import copy
import functools
import warnings
from types import ModuleType
from typing import Any

from pydantic import BaseModel
from tesseract_core import Tesseract

from tesseract_jax.tree_util import to_shape_dtype_pytree


def _extract_api_module(
    tesseract_client: Tesseract, endpoint: str = "apply"
) -> ModuleType | None:
    """The real ``tesseract_api`` module backing an in-process Tesseract, or ``None``.

    Only a ``LocalClient`` (``Tesseract.from_tesseract_api``) has one: a
    served (``HTTPClient``) Tesseract runs in another process.

    Tries the public ``LocalClient.api_module`` first (added in
    pasteurlabs/tesseract-core#784). Falls back to recovering it from a
    closure cell of the schema-validating endpoint wrapper
    ``tesseract_core.runtime.core.create_endpoints`` built -- a private
    integration point, tied to that function closing over a free variable
    named ``api_module`` -- for tesseract-core versions before that PR.
    TODO: delete the fallback once the floor is bumped past it.
    """
    local_client = getattr(tesseract_client, "_client", None)

    api_module = getattr(local_client, "api_module", None)
    if isinstance(api_module, ModuleType):
        return api_module

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


def is_traceable(tesseract_client: Tesseract, endpoint: str = "apply") -> bool:
    """Whether ``tesseract_client`` has an importable Python function to trace directly."""
    return _extract_api_module(tesseract_client, endpoint) is not None


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


def _abstract_inputs_schema_for(tesseract_client: Tesseract, endpoint: str) -> tuple:
    """The real ``api_module`` and its cached ``AbstractEval_`` input schema."""
    api_module = _extract_api_module(tesseract_client, endpoint)
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

    ``real_inputs`` is the full (static + dynamic) input pytree; its dynamic
    leaves may be tracers.
    """
    abstract_inputs = to_shape_dtype_pytree(real_inputs)
    abstract_instance = AbstractInputSchema.model_validate({"inputs": abstract_inputs})
    return _patch_with_real_values(abstract_instance.inputs, real_inputs)


def _call_with_patched_inputs(endpoint_fn: Any, **kwargs: Any) -> Any:
    """Call ``endpoint_fn(**kwargs)``, suppressing the expected schema/value mismatch warning.

    One of ``kwargs`` is a ``_patch_inputs``-patched instance whose declared
    field types (``ShapeDType``) no longer match what it actually holds, so
    any ``.model_dump()`` on it -- inside the endpoint, or on its return value
    -- warns about the mismatch. Expected: the corresponding real schema's
    ``Array`` type would raise outright in the same spot (it calls
    ``np.asarray`` on serialization too).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings", category=UserWarning
        )
        return endpoint_fn(**kwargs)


class TracedClient:
    """Stands in for a Tesseract's ``LocalClient``, tracing four of its endpoints.

    Traces ``apply``/``jacobian_vector_product``/``vector_jacobian_product``/
    ``jacobian`` directly instead of dispatching them through the
    schema-validated wrapper. Every other endpoint (``openapi_schema``,
    ``abstract_eval``, ``health``, ``test``) is delegated to the real
    ``LocalClient`` unchanged -- there is nothing to trace there
    (``abstract_eval`` in particular only ever deals in shapes/dtypes, never
    a traced value, so the real, fully-validated path is strictly better,
    not just adequate).
    """

    _TRACED_ENDPOINTS = frozenset(
        {"apply", "jacobian_vector_product", "vector_jacobian_product", "jacobian"}
    )

    def __init__(self, tesseract_client: Tesseract) -> None:
        self._tesseract_client = tesseract_client

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TracedClient):
            return NotImplemented
        return self._tesseract_client == other._tesseract_client

    def __hash__(self) -> int:
        return hash((TracedClient, self._tesseract_client))

    def run_tesseract(
        self,
        endpoint: str,
        payload: dict | None = None,
        run_id: str | None = None,
        stream_logs: Any = False,
    ) -> dict:
        """Dispatch ``endpoint``, tracing it directly if it's one of the four traced ones."""
        if endpoint not in self._TRACED_ENDPOINTS:
            return self._tesseract_client._client.run_tesseract(
                endpoint, payload, run_id, stream_logs
            )
        api_module, schema = _abstract_inputs_schema_for(
            self._tesseract_client, endpoint
        )
        payload = dict(payload)
        patched_inputs = _patch_inputs(schema, payload.pop("inputs"))
        out = _call_with_patched_inputs(
            getattr(api_module, endpoint), inputs=patched_inputs, **payload
        )
        return out.model_dump() if isinstance(out, BaseModel) else out


def traced_tesseract(tesseract_client: Tesseract) -> Tesseract:
    """A copy of ``tesseract_client`` that traces its dispatch endpoints directly.

    A shallow copy with ``_client`` replaced by a ``TracedClient`` wrapping
    the original -- everything else (``._stream_logs``, etc.) is shared with
    the original unchanged.
    """
    shim = copy.copy(tesseract_client)
    shim._client = TracedClient(tesseract_client)
    return shim
