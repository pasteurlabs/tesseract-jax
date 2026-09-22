# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace a Tesseract's real dispatch endpoints instead of dispatching them.

``traced_tesseract(client)`` shallow-copies a ``Tesseract``, swapping its
``LocalClient`` for ``TracedClient``. ``Tesseract``'s own ``apply`` /
``jacobian_vector_product`` / ``vector_jacobian_product`` / ``jacobian``
methods all forward through ``self._client.run_tesseract(endpoint, payload,
...)``, so swapping just ``_client`` makes every one of them -- and
``openapi_schema`` / ``available_endpoints``, which route through
``run_tesseract`` too -- traceable, with ``TracedClient`` doing the one
genuinely new thing: validating and calling the real Python endpoint
directly. Everything else is ``Jaxeract``'s existing dispatch code
(``tesseract_compat.py``), unchanged.

Lowering the resulting dispatch closure with ``jax.interpreters.mlir.lower_fun``
(``mlir.emit_python_callback`` for the normal path) is what makes
``apply_tesseract(..., traceable=True)`` inline the endpoint into the jaxpr.

Inputs and outputs are exchanged as validation-light objects: shapes/dtypes
are checked against the same ``AbstractEval_``-prefixed sibling schema
``abstract_eval`` already uses, then real values are patched in without
re-validating, and each endpoint returns a plain ``dict`` (or builds via
``model_construct``) rather than the schema's validating constructor. Both
follow from the same fact -- pydantic's ``Array[...]`` validator calls
``np.asarray``, which rejects a Tracer -- for every endpoint, not just ``apply``.
"""

import copy
import functools
import warnings
from types import ModuleType
from typing import Any

from pydantic import BaseModel
from tesseract_core import Tesseract

from tesseract_jax.tree_util import to_shape_dtype_pytree


def _extract_api_module_from_local_client(
    local_client: Any, endpoint: str = "apply"
) -> ModuleType | None:
    """The real ``tesseract_api`` module backing a ``LocalClient``, or ``None``.

    Tries the public ``LocalClient.api_module`` first (added in
    pasteurlabs/tesseract-core#784). Falls back to recovering it from a
    closure cell of the schema-validating endpoint wrapper
    ``tesseract_core.runtime.core.create_endpoints`` built -- a private
    integration point, tied to that function closing over a free variable
    named ``api_module`` -- for tesseract-core versions before that PR.
    TODO: delete the fallback once the floor is bumped past it.
    """
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
    """Whether ``tesseract_client`` has an importable Python function to trace directly.

    Only a ``LocalClient`` (``Tesseract.from_tesseract_api``) has one: a
    served (``HTTPClient``) Tesseract runs in another process.
    """
    local_client = getattr(tesseract_client, "_client", None)
    return _extract_api_module_from_local_client(local_client, endpoint) is not None


@functools.cache
def _abstract_input_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> type[BaseModel]:
    """The ``AbstractEval_``-prefixed sibling of ``InputSchema``, cached per schema pair."""
    from tesseract_core.runtime.schema_generation import create_abstract_eval_schema

    AbstractInputSchema, _ = create_abstract_eval_schema(InputSchema, OutputSchema)
    return AbstractInputSchema


def _present(real_node: Any, key: Any) -> bool:
    """Whether ``real_node`` has a value at ``key`` (a dict key or list/tuple index)."""
    if isinstance(real_node, dict):
        return key in real_node
    if isinstance(real_node, (list, tuple)):
        return key < len(real_node)
    return False


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
                if _present(real_node, name)
                else getattr(schema_node, name)
            )
            for name in type(schema_node).model_fields
        }
        return schema_node.model_copy(update=updates)
    if isinstance(schema_node, dict):
        return {
            key: (
                _patch_with_real_values(value, real_node[key])
                if _present(real_node, key)
                else value
            )
            for key, value in schema_node.items()
        }
    if isinstance(schema_node, (list, tuple)):
        patched = [
            _patch_with_real_values(value, real_node[i])
            if _present(real_node, i)
            else value
            for i, value in enumerate(schema_node)
        ]
        return type(schema_node)(patched)
    return schema_node


def _abstract_inputs_schema_for_local_client(local_client: Any, endpoint: str) -> tuple:
    """The real ``api_module`` and its cached ``AbstractEval_`` input schema."""
    api_module = _extract_api_module_from_local_client(local_client, endpoint)
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

    The patched instance's ``ShapeDType``-typed fields don't match its real
    values, so any ``.model_dump()`` -- inside the endpoint, or on its return
    value -- warns about the mismatch; suppressed because it's expected.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Pydantic serializer warnings", category=UserWarning
        )
        return endpoint_fn(**kwargs)


class TracedClient:
    """Stands in for a Tesseract's ``LocalClient``, tracing endpoints that take ``inputs``.

    Traces any endpoint whose payload has an ``"inputs"`` key (``apply`` /
    ``jacobian_vector_product`` / ``vector_jacobian_product`` / ``jacobian``
    today; a new differentiable-dispatch endpoint traces automatically once
    added, since the patch/call logic below is endpoint-name-agnostic).
    ``abstract_eval`` is excluded by name, since its payload also happens to
    be ``{"inputs": ...}``. Every excluded endpoint (``abstract_eval``,
    ``openapi_schema``, ``health``, ``test``) delegates to the real
    ``LocalClient``.
    """

    def __init__(self, local_client: Any) -> None:
        self._local_client = local_client

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TracedClient):
            return NotImplemented
        return self._local_client == other._local_client

    def __hash__(self) -> int:
        return hash((TracedClient, self._local_client))

    def run_tesseract(
        self,
        endpoint: str,
        payload: dict | None = None,
        run_id: str | None = None,
        stream_logs: Any = False,
    ) -> dict:
        """Dispatch ``endpoint``, tracing it directly unless excluded (see class docstring)."""
        if endpoint == "abstract_eval" or not payload or "inputs" not in payload:
            return self._local_client.run_tesseract(
                endpoint, payload, run_id, stream_logs
            )
        api_module, schema = _abstract_inputs_schema_for_local_client(
            self._local_client, endpoint
        )
        payload = dict(payload)
        patched_inputs = _patch_inputs(schema, payload.pop("inputs"))
        out = _call_with_patched_inputs(
            getattr(api_module, endpoint), inputs=patched_inputs, **payload
        )
        return out.model_dump() if isinstance(out, BaseModel) else out


def traced_tesseract(tesseract_client: Tesseract) -> Tesseract:
    """A copy of ``tesseract_client`` that traces its dispatch endpoints directly.

    Wraps the original ``_client`` (the real ``LocalClient``) in a
    ``TracedClient``, giving a flat ``Tesseract -> TracedClient ->
    LocalClient`` chain. Everything else on the copy (``._stream_logs``,
    etc.) is shared with the original.
    """
    shim = copy.copy(tesseract_client)
    shim._client = TracedClient(tesseract_client._client)
    return shim
