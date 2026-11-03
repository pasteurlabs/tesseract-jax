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
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import dtypes
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined

from tesseract_jax.tree_util import to_shape_dtype_pytree


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


def _safe_cast_dtype(value: Any, target: str) -> Any:
    """Cast ``value`` to ``target`` only when it is an AD-safe, checkable cast.

    A float/complex ``target`` accepts any cast tesseract-core accepts for it
    (an overflowing value becomes ``inf``, as any JAX cast would, not an
    error); an integer/bool ``target`` only accepts a widening cast, since
    narrowing or a sign change can silently wrap a value and a JAX cast does
    not range-check it the way tesseract-core's real array validator does on
    the callback path. Left uncast, a real mismatch here still surfaces as
    the MLIR output-type error this function was built to avoid, rather than
    silently computing on a wrapped value. A ``target`` JAX cannot represent
    (a 64-bit dtype with ``jax_enable_x64`` off) is left uncast too.

    This policy isn't shared with anything else in tesseract-jax today (a
    matching input-side cast in ``primitive.py``, ``_cast_to_schema_dtype``,
    exists only on the not-yet-merged #307 -- if it lands, extract one shared
    "is this cast AD-safe" predicate instead of keeping two copies in sync).
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
    """An omitted field's schema default, or raise if it hides an unresolved array placeholder.

    Used once :func:`_omitted_field_value` and :func:`_real_nested_cls` have
    already failed to recover a real value for this leaf -- a field whose
    real sibling class isn't known (inside a plain ``dict``/``list``, or
    behind an ``Optional``/``Union`` annotation we don't resolve; see
    :func:`_real_nested_cls`) or whose default a ``TypeAdapter`` rejects.
    tesseract-core (>=1.15) validates a field's default against the abstract
    schema too, so an array-valued default (e.g. ``np.float32(1.0)``) comes
    back as a ``ShapeDType`` placeholder with no real value to patch in --
    unlike on the callback path, where ``InputSchema.model_validate`` keeps
    the real default since pydantic doesn't re-validate defaults there.
    """
    from tesseract_core.runtime.schema_types import ShapeDType

    if isinstance(value, ShapeDType):
        raise ValueError(  # noqa: TRY004
            f"traceable=True cannot use the schema default for {field_description!r}: "
            "it was omitted from the inputs, and its default is an array. "
            "tesseract-core resolves that default to a shape/dtype placeholder, not "
            "a real value, so pass this field explicitly when using traceable=True."
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
    """The real (non-abstract) nested model class backing field ``name``, or ``None``.

    ``None`` means a nested default under this field can't be recovered by
    :func:`_omitted_field_value` -- either because ``real_cls`` itself was
    already unrecoverable, or because the field's real annotation isn't a
    bare class reference (``Optional[Model]``, ``Model | Other``, and similar
    wrapped/union annotations all fall back to :func:`_checked_default`
    instead of being resolved here).
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
    """The real value for an omitted field: its real default, or :func:`_checked_default`'s fallback.

    tesseract-core (>=1.15) validates a field's default against the abstract
    schema too, so an array-valued default comes back as a ``ShapeDType``
    placeholder on ``abstract_value`` with no real value left on it. But a
    default is never a tracer -- only a caller-supplied value can be -- so
    there's nothing stopping us from validating it through the *real* field
    directly instead: the exact same ``TypeAdapter(field.annotation)
    .validate_python(default)`` tesseract-core's own schema generation runs
    before discarding the result down to a ``ShapeDType`` (see
    ``replace_array_default_with_shapedtype`` in tesseract-core's
    ``schema_generation.py``) -- we just keep the value instead of throwing
    it away. Falls back to :func:`_checked_default` whenever ``real_cls``
    doesn't have this field (not a bare class reference; see
    :func:`_real_nested_cls`) or its default doesn't validate.
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
    """Replace each array (``ShapeDType``) leaf with the matching real value.

    A field missing from ``real_node`` (omitted from ``apply_tesseract``'s
    raw ``inputs``, relying on ``InputSchema``'s own default) is resolved by
    :func:`_omitted_field_value` -- the real default when ``real_cls`` (the
    non-abstract sibling of ``schema_node``'s class) makes that possible, else
    :func:`_checked_default`'s fallback. A present non-array leaf keeps
    ``schema_node``'s value, since it already passed whatever validator
    applies to it during the abstract-schema validation.

    A present array leaf is cast to ``schema_node.dtype`` -- the dtype the
    abstract-schema validation already resolved the field to (including any
    expected_dtype coercion ``Array[...]`` declares) -- when
    :func:`_safe_cast_dtype` allows it, so a tracer gets the same dtype
    coercion ``python_to_array`` would apply to a real value on the callback
    path, instead of flowing through unconverted and failing MLIR's
    output-type check later.
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
    """A ``dict``/``list``/``tuple`` item at ``key``, patched or defaulted like any other leaf."""
    item_path = f"{path}[{key!r}]"
    if _present(real_node, key):
        return _patch_with_real_values(value, real_node[key], None, item_path)
    return _checked_default(value, item_path)


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

    def __getattr__(self, name: str) -> Any:
        """Forward anything not overridden here to the wrapped real client.

        ``Jaxeract`` (and ``Tesseract`` properties like ``openapi_schema``,
        ``server_capabilities``, ``current_encoding``) read these directly off
        the client, not through ``run_tesseract``, so a shim that only
        overrides ``run_tesseract`` is not actually a drop-in replacement
        without this.
        """
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
        """Dispatch ``endpoint``, tracing it directly unless excluded (see class docstring).

        ``encoding`` is ignored, like ``LocalClient``'s: arrays are passed in
        memory either way, so there is nothing to encode.
        """
        if endpoint == "abstract_eval" or not payload or "inputs" not in payload:
            return self._local_client.run_tesseract(
                endpoint, payload, run_id, stream_logs, encoding
            )

        api_module = getattr(self._local_client, "api_module", None)
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
            abstract_instance.inputs, orig_inputs, api_module.InputSchema
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
