# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract client that traces its real endpoints instead of dispatching them.

``TracedClient`` duck-types the same interface ``Jaxeract`` (in
``tesseract_compat.py``) expects from a real ``tesseract_core.Tesseract``
client: ``.openapi_schema``, ``.available_endpoints``, and
``apply``/``jacobian_vector_product``/``vector_jacobian_product``/``jacobian``
methods taking and returning plain dicts. Feeding a ``Jaxeract(TracedClient(...))``
into the existing dispatch closure (``primitive.py``'s ``_build_dispatch_closure``)
and lowering it with ``jax.interpreters.mlir.lower_fun`` instead of
``mlir.emit_python_callback`` gets ``apply_tesseract(..., traceable=True)`` for
free: every endpoint's path/tangent/placeholder bookkeeping is ``Jaxeract``'s
existing, already-tested code, unchanged. ``TracedClient`` itself only does the
one genuinely new thing -- validate and call the real Python endpoint directly.

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
    served (``HTTPClient``) Tesseract runs in another process. ``LocalClient``
    doesn't keep the module itself, only the endpoint wrapper functions
    ``tesseract_core.runtime.core.create_endpoints`` built from it -- so it's
    recovered from the closure cell those wrappers hold it in. This depends
    on ``create_endpoints`` closing over a free variable named ``api_module``;
    if a future ``tesseract-core`` release changes that shape, this returns
    ``None`` (a loud error at the call site) rather than tracing the wrong
    thing.
    """
    local_client = getattr(tesseract_client, "_client", None)
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
    """Duck-types a ``Tesseract`` client, tracing its real endpoints directly.

    Wrap a ``Jaxeract`` around one of these instead of a real
    ``tesseract_core.Tesseract`` to make its ``apply`` /
    ``jacobian_vector_product`` / ``vector_jacobian_product`` / ``jacobian``
    calls trace the underlying Python functions in-process, rather than
    dispatching through the real client's (HTTP or validated-local) endpoint
    boundary. See the module docstring for the mechanism and its two
    preconditions (in-process Tesseract; dict-returning endpoints).
    """

    def __init__(self, tesseract_client: Tesseract) -> None:
        self._tesseract_client = tesseract_client
        self.openapi_schema = tesseract_client.openapi_schema
        self.available_endpoints = tesseract_client.available_endpoints

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TracedClient):
            return NotImplemented
        return self._tesseract_client == other._tesseract_client

    def __hash__(self) -> int:
        return hash((TracedClient, self._tesseract_client))

    def apply(self, inputs: dict) -> dict:
        """Trace the real ``apply`` directly."""
        api_module, schema = _abstract_inputs_schema_for(
            self._tesseract_client, "apply"
        )
        patched = _patch_inputs(schema, inputs)
        out = _call_with_patched_inputs(api_module.apply, inputs=patched)
        return out.model_dump() if isinstance(out, BaseModel) else out

    def jacobian_vector_product(
        self, inputs: dict, jvp_inputs: list, jvp_outputs: list, tangent_vector: dict
    ) -> dict:
        """Trace the real ``jacobian_vector_product`` directly."""
        api_module, schema = _abstract_inputs_schema_for(
            self._tesseract_client, "jacobian_vector_product"
        )
        patched = _patch_inputs(schema, inputs)
        return _call_with_patched_inputs(
            api_module.jacobian_vector_product,
            inputs=patched,
            jvp_inputs=jvp_inputs,
            jvp_outputs=jvp_outputs,
            tangent_vector=tangent_vector,
        )

    def vector_jacobian_product(
        self, inputs: dict, vjp_inputs: list, vjp_outputs: list, cotangent_vector: dict
    ) -> dict:
        """Trace the real ``vector_jacobian_product`` directly."""
        api_module, schema = _abstract_inputs_schema_for(
            self._tesseract_client, "vector_jacobian_product"
        )
        patched = _patch_inputs(schema, inputs)
        return _call_with_patched_inputs(
            api_module.vector_jacobian_product,
            inputs=patched,
            vjp_inputs=vjp_inputs,
            vjp_outputs=vjp_outputs,
            cotangent_vector=cotangent_vector,
        )

    def jacobian(self, inputs: dict, jac_inputs: list, jac_outputs: list) -> dict:
        """Trace the real ``jacobian`` directly."""
        api_module, schema = _abstract_inputs_schema_for(
            self._tesseract_client, "jacobian"
        )
        patched = _patch_inputs(schema, inputs)
        return _call_with_patched_inputs(
            api_module.jacobian,
            inputs=patched,
            jac_inputs=jac_inputs,
            jac_outputs=jac_outputs,
        )
