# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace a Tesseract's real dispatch endpoints instead of dispatching them.

Provides ``TracedClient`` as a drop-in replacement for ``LocalCLient``,
validating and calling the real Python endpoint directly.

Inputs and outputs are exchanged as validation-light objects: shapes/dtypes
are checked against the same ``AbstractEval_``-prefixed sibling schema
``abstract_eval`` already uses, then real values are patched in without
re-validating, and each endpoint returns a plain ``dict`` (or builds via
``model_construct``) rather than the schema's validating constructor. Both
follow from the same fact -- pydantic's ``Array[...]`` validator calls
``np.asarray``, which rejects a Tracer -- for every endpoint, not just ``apply``.
"""

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
    """The real ``tesseract_api`` module backing a ``LocalClient``, or ``None``."""
    api_module = getattr(local_client, "api_module", None)
    if isinstance(api_module, ModuleType):
        return api_module

    # TODO: Remove the below when #684 lands and maybe inline this function
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

    TODO: This function is only called once it will also be inlined when #684 lands
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

        api_module = _extract_api_module_from_local_client(self._local_client, endpoint)
        if api_module is None:
            raise ValueError(
                "traceable=True requires an in-process Tesseract built via "
                "Tesseract.from_tesseract_api(...)."
            )

        schema = _abstract_input_schema(api_module.InputSchema, api_module.OutputSchema)

        orig_inputs = payload.pop("inputs")
        abstract_inputs = to_shape_dtype_pytree(orig_inputs)
        abstract_instance = schema.model_validate({"inputs": abstract_inputs})
        patched_instance = _patch_with_real_values(
            abstract_instance.inputs, orig_inputs
        )

        # If endpoint calls ``inputs.model_dump``, pydantic would raise a warning because
        # our patched inputs is still an ``AbstractInputSchema`` even though it doesn't
        # conform to this specification due to the presence of concrete arrays. This is
        # expected so we can safely suppress.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Pydantic serializer warnings", category=UserWarning
            )
            out = getattr(api_module, endpoint)(inputs=patched_instance, **payload)

        return out.model_dump() if isinstance(out, BaseModel) else out
