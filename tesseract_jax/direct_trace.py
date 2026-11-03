# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace a Tesseract's Python endpoints into the jaxpr instead of dispatching them.

``TracedClient`` stands in for a Tesseract's ``LocalClient``. It validates
inputs against the ``AbstractEval_`` schema that ``abstract_eval`` uses, which
checks shapes and dtypes only, and then patches the real (possibly traced)
values back in without re-validating them. Full validation is not possible
because pydantic's ``Array`` validator calls ``np.asarray``, which raises on a
tracer. For the same reason, every traced endpoint must return a plain
``dict`` (or build its output via ``model_construct``) instead of calling
``OutputSchema(...)``.
"""

import copy
import functools
import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import dtypes
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined
from tesseract_core import Tesseract

from tesseract_jax.tree_util import to_shape_dtype_pytree


@functools.cache
def _abstract_input_schema(
    InputSchema: type[BaseModel], OutputSchema: type[BaseModel]
) -> type[BaseModel]:
    """Return the ``AbstractEval_`` variant of ``InputSchema``, cached per schema pair."""
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


def _safe_cast_dtype(value: Any, target: str) -> Any:
    """Cast ``value`` to ``target`` unless the cast could silently corrupt it.

    Float and complex targets accept any ``same_kind`` cast, as tesseract-core
    does, and an overflowing value becomes ``inf``. Integer and bool targets
    accept only widening casts, because a JAX cast does not range-check the
    value the way tesseract-core's array validator does. A value left uncast
    later fails MLIR's output-type check, which is safer than computing on a
    wrapped value. Targets JAX cannot represent (64-bit dtypes without
    ``jax_enable_x64``) are left uncast too.
    """
    target_dtype = np.dtype(target)
    if (
        value.dtype == target_dtype
        or dtypes.canonicalize_dtype(target_dtype) != target_dtype
    ):
        return value
    if np.can_cast(value.dtype, target_dtype, casting="safe") or (
        jnp.issubdtype(target_dtype, jnp.inexact)
        and np.can_cast(value.dtype, target_dtype, casting="same_kind")
    ):
        return value.astype(target_dtype)
    return value


def _checked_default(value: Any, field_description: str) -> Any:
    """Return an omitted field's default, raising if it contains a ``ShapeDType`` placeholder.

    Fallback for when :func:`_omitted_field_value` cannot recover the real default.
    """
    from tesseract_core.runtime.schema_types import ShapeDType

    if isinstance(value, ShapeDType):
        raise ValueError(  # noqa: TRY004
            f"A traced Tesseract cannot use the schema default for "
            f"{field_description!r}: it was omitted from the inputs, and its "
            "default is an array. tesseract-core resolves that default to a "
            "shape/dtype placeholder, not a real value, so pass this field "
            "explicitly when calling a Tesseract built with to_traceable()."
        )
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _checked_default(getattr(value, name), f"{field_description}.{name}")
    elif isinstance(value, dict):
        for key, v in value.items():
            _checked_default(v, f"{field_description}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _checked_default(v, f"{field_description}[{i}]")
    return value


def _real_nested_cls(
    real_cls: type[BaseModel] | None, name: str
) -> type[BaseModel] | None:
    """Return the model class annotating field ``name`` of ``real_cls``, or ``None``.

    Only bare class annotations resolve. ``Optional[Model]``, unions, and other
    wrapped annotations return ``None``.
    """
    field_info = real_cls.model_fields.get(name) if real_cls is not None else None
    if field_info is None:
        return None
    annotation = field_info.annotation
    is_model = isinstance(annotation, type) and issubclass(annotation, BaseModel)
    return annotation if is_model else None


def _omitted_field_value(
    abstract_value: Any, real_cls: type[BaseModel] | None, name: str, path: str
) -> Any:
    """Return the value of an omitted field, recovering its real default where possible.

    tesseract-core (>=1.15) validates defaults against the abstract schema,
    which turns an array-valued default into a ``ShapeDType`` placeholder.
    Since a default is never a tracer, we validate it against the real field
    annotation instead, using the same ``TypeAdapter`` call tesseract-core
    makes in ``replace_array_default_with_shapedtype``. If ``real_cls`` is
    unknown (inside a plain ``dict``/``list``, or behind an annotation
    :func:`_real_nested_cls` does not resolve) or the default fails to
    validate, :func:`_checked_default` handles the field instead.
    """
    from tesseract_core.runtime.schema_types import ShapeDType

    if not isinstance(abstract_value, ShapeDType) or real_cls is None:
        return _checked_default(abstract_value, path)

    field_info = real_cls.model_fields.get(name)
    if field_info is None:
        return _checked_default(abstract_value, path)

    default = field_info.get_default(call_default_factory=True)
    if default is PydanticUndefined:
        return _checked_default(abstract_value, path)

    try:
        real_default = TypeAdapter(field_info.annotation).validate_python(default)
    except ValidationError:
        return _checked_default(abstract_value, path)

    return jnp.asarray(real_default)


def _patch_with_real_values(
    schema_node: Any,
    real_node: Any,
    real_cls: type[BaseModel] | None = None,
    path: str = "inputs",
) -> Any:
    """Replace each ``ShapeDType`` leaf of ``schema_node`` with the matching value in ``real_node``.

    Array values are cast to the dtype the abstract schema resolved, if
    :func:`_safe_cast_dtype` allows it, to match the coercion the callback path
    applies. Non-array leaves keep their already-validated ``schema_node``
    value. Fields omitted from ``real_node`` go through
    :func:`_omitted_field_value`, which uses ``real_cls``, the non-abstract
    model class of ``schema_node``, to recover real defaults.
    """
    from tesseract_core.runtime.schema_types import ShapeDType

    if isinstance(schema_node, ShapeDType):
        return _safe_cast_dtype(jnp.asarray(real_node), schema_node.dtype)
    if isinstance(schema_node, BaseModel):
        updates = {
            name: (
                _patch_with_real_values(
                    getattr(schema_node, name),
                    real_node[name],
                    _real_nested_cls(real_cls, name),
                    f"{path}.{name}",
                )
                if _present(real_node, name)
                else _omitted_field_value(
                    getattr(schema_node, name), real_cls, name, f"{path}.{name}"
                )
            )
            for name in type(schema_node).model_fields
        }
        return schema_node.model_copy(update=updates)
    if isinstance(schema_node, dict):
        return {
            key: _patch_container_item(key, value, real_node, path)
            for key, value in schema_node.items()
        }
    if isinstance(schema_node, (list, tuple)):
        patched = [
            _patch_container_item(i, value, real_node, path)
            for i, value in enumerate(schema_node)
        ]
        return type(schema_node)(patched)
    return schema_node


def _patch_container_item(key: Any, value: Any, real_node: Any, path: str) -> Any:
    """Patch the ``dict``/``list``/``tuple`` item at ``key``, or check its default if omitted."""
    item_path = f"{path}[{key!r}]"
    if _present(real_node, key):
        return _patch_with_real_values(value, real_node[key], None, item_path)
    return _checked_default(value, item_path)


class TracedClient:
    """Stand-in for ``LocalClient`` that traces every endpoint called with ``inputs``.

    This covers ``apply`` and the differentiation endpoints. ``abstract_eval``
    and endpoints without ``inputs`` delegate to the wrapped ``LocalClient``.
    """

    def __init__(self, local_client: Any) -> None:
        self._local_client = local_client

    def __getattr__(self, name: str) -> Any:
        """Forward other attributes, which ``Jaxeract`` and ``Tesseract`` read off the client."""
        return getattr(self._local_client, name)

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
        encoding: Any = None,
    ) -> dict:
        """Trace ``endpoint`` directly, or delegate it to the wrapped client.

        ``encoding`` is ignored, as in ``LocalClient``, since arrays stay in memory.
        """
        if endpoint == "abstract_eval" or not payload or "inputs" not in payload:
            return self._local_client.run_tesseract(
                endpoint, payload, run_id, stream_logs, encoding
            )

        # to_traceable() has already checked that api_module exists.
        api_module = self._local_client.api_module

        schema = _abstract_input_schema(api_module.InputSchema, api_module.OutputSchema)

        orig_inputs = payload.pop("inputs")
        abstract_inputs = to_shape_dtype_pytree(orig_inputs)
        abstract_instance = schema.model_validate({"inputs": abstract_inputs})
        patched_instance = _patch_with_real_values(
            abstract_instance.inputs, orig_inputs, api_module.InputSchema
        )

        # The patched inputs are an AbstractInputSchema instance holding real arrays,
        # so pydantic warns if the endpoint calls model_dump. This is expected.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Pydantic serializer warnings", category=UserWarning
            )
            out = getattr(api_module, endpoint)(inputs=patched_instance, **payload)

        return out.model_dump() if isinstance(out, BaseModel) else out


def to_traceable(tesseract_client: Tesseract) -> Tesseract:
    """Return a copy of ``tesseract_client`` whose endpoints ``apply_tesseract`` traces into the jaxpr.

    Requires an in-process Tesseract built via ``Tesseract.from_tesseract_api``,
    and raises ``ValueError`` otherwise. Pass the copy to ``apply_tesseract`` in
    place of the original. Every endpoint it calls must return a plain ``dict``,
    and inputs are only validated for shape and dtype. See the "No JAX
    operations inside ``from_tesseract_api`` endpoints" sharp edge in
    :doc:`/content/get-started`.
    """
    api_module = getattr(tesseract_client._client, "api_module", None)
    if api_module is None:
        raise ValueError(
            "to_traceable requires an in-process Tesseract built via "
            "Tesseract.from_tesseract_api(...); "
            f"{tesseract_client!r} has no importable apply function to trace "
            "directly."
        )
    traced = copy.copy(tesseract_client)
    traced._client = TracedClient(tesseract_client._client)
    return traced
