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
from pydantic import BaseModel

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

    Mirrors the policy ``apply_tesseract`` itself applies to inputs before
    dispatch (``_cast_to_schema_dtype`` in ``primitive.py``): a float/complex
    ``target`` accepts any cast tesseract-core accepts for it (an overflowing
    value becomes ``inf``, as any JAX cast would, not an error); an
    integer/bool ``target`` only accepts a widening cast, since narrowing or a
    sign change can silently wrap a value and a JAX cast does not range-check
    it the way tesseract-core's real array validator does on the callback
    path. Left uncast, a real mismatch here still surfaces as the MLIR
    output-type error this function was built to avoid, rather than silently
    computing on a wrapped value. A ``target`` JAX cannot represent (a 64-bit
    dtype with ``jax_enable_x64`` off) is left uncast too.
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
    """An omitted field's schema default, or raise if it hides an array placeholder.

    tesseract-core (>=1.15) validates a field's default against the abstract
    schema too, so an array-valued default (e.g. ``np.float32(1.0)``) comes
    back as a ``ShapeDType`` placeholder with no real value to patch in --
    unlike on the callback path, where ``InputSchema.model_validate`` keeps
    the real default since pydantic doesn't re-validate defaults there.
    traceable=True has no way to recover the original default from the
    abstract schema alone, so this raises instead of silently handing the
    real endpoint a placeholder where it expects a real array.
    """
    from tesseract_core.runtime.schema_types import ShapeDType

    if isinstance(value, ShapeDType):
        # ValueError, not TypeError: value is of the expected type (ShapeDType) --
        # it's its *meaning* (no real default survived validation) that's wrong,
        # not its type.
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


def _patch_with_real_values(
    schema_node: Any, real_node: Any, path: str = "inputs"
) -> Any:
    """Replace each array (``ShapeDType``) leaf with the matching real value.

    A field missing from ``real_node`` (omitted from ``apply_tesseract``'s
    raw ``inputs``, relying on ``InputSchema``'s own default) keeps
    ``schema_node``'s own value instead -- the same default the callback
    path's ``InputSchema.model_validate`` would use, since pydantic doesn't
    re-validate defaults either -- except an array-valued default, which
    :func:`_checked_default` rejects (see its docstring). A present non-array
    leaf also keeps ``schema_node``'s value, since it already passed whatever
    validator applies to it during the abstract-schema validation.

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
                    getattr(schema_node, name), real_node[name], f"{path}.{name}"
                )
                if _present(real_node, name)
                else _checked_default(getattr(schema_node, name), f"{path}.{name}")
            )
            for name in type(schema_node).model_fields
        }
        return schema_node.model_copy(update=updates)
    if isinstance(schema_node, dict):
        return {
            key: (
                _patch_with_real_values(value, real_node[key], f"{path}[{key!r}]")
                if _present(real_node, key)
                else _checked_default(value, f"{path}[{key!r}]")
            )
            for key, value in schema_node.items()
        }
    if isinstance(schema_node, (list, tuple)):
        patched = [
            _patch_with_real_values(value, real_node[i], f"{path}[{i}]")
            if _present(real_node, i)
            else _checked_default(value, f"{path}[{i}]")
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
