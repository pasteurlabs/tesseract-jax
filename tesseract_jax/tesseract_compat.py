# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import jax.tree
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax.tree_util import (
    PyTree,
    _pytree_to_tesseract_flat,
    combine_args,
    unflatten_args,
)

if TYPE_CHECKING:
    from tesseract_jax.dispatch_params import DispatchParams

# WARNING: Do NOT use jax.numpy within Jaxeract methods, as they are executed from within FFI callbacks
# and cannot safely allocate JAX arrays. Use vanilla numpy instead.

# The endpoint methods are transport-agnostic: the CPU host-callback lowering
# passes real arrays (``ArrayLike``), while the GPU FFI lowering passes bare
# ``__cuda_array_interface__`` device views. ``Any`` admits both so a runtime
# type-check does not reject the duck-typed GPU views.
TransportArray = Any


def _on_device(values: "list | tuple") -> bool:
    """Whether ``values`` are cuda_ipc device arrays (vs host NumPy arrays).

    The endpoint methods are transport-agnostic; this distinguishes the GPU FFI
    lowering (bare ``__cuda_array_interface__`` device views / ``IpcDeviceArray``
    results) from the CPU host-callback lowering (real NumPy arrays).

    The ``cuda_ipc`` import is deliberately lazy, not at module scope: eagerly
    importing ``tesseract_core.runtime.cuda_ipc`` perturbs schema/typeguard state
    in the shared interpreter and breaks in-process (``LocalClient``) Tesseracts
    whose endpoints use ellipsis-shaped array schemas.
    """
    from tesseract_core.runtime.cuda_ipc import has_cuda_array_interface

    return any(has_cuda_array_interface(v) for v in values)


def _cast_return(value: TransportArray, *, dtype: np.dtype) -> TransportArray:
    """Coerce a dispatch result to the return ``dtype`` without leaving the device.

    On the cuda_ipc (GPU FFI) path ``value`` is a device array whose bytes the
    FFI handler copies straight into XLA's output buffer -- it is already the
    right dtype (the server computed it), so it is returned untouched. Wrapping
    it in ``np.asarray`` here would force a device->host copy and then hand a
    host pointer back across the FFI boundary. On the host path ``value`` is a
    NumPy array and we cast as before.
    """
    if _on_device([value]):
        return value
    return np.asarray(value, dtype=dtype)


def _placeholder(
    shape: tuple[int, ...], dtype: np.dtype, *, on_device: bool
) -> TransportArray:
    """A discarded slot in a derivative call's output tuple.

    Used for the gradient of a non-differentiable input and the tangent of a
    non-differentiable output. Such a slot exists only to satisfy the
    output-tuple-length contract; JAX's transpose machinery never consumes it for
    any user-requested derivative, so its *value* is immaterial (verified:
    substituting any value leaves every user-visible gradient unchanged).

    On the cuda_ipc path we return ``None``: the native FFI handler fills XLA's
    (already-allocated) output buffer for that slot rather than copying from a
    fabricated source array. On the host path we return an array filled with each
    dtype's ``0/0`` value (see :func:`_discarded_slot`). Either way an accidental
    consumer surfaces loudly rather than silently.
    """
    if on_device:
        return None
    return _discarded_slot(shape, dtype)


# Every dtype a Tesseract schema can carry; see the `dtype` enum in the
# generated OpenAPI schema (tesseract_core.runtime.schema_types).
_SCHEMA_DTYPES = (
    "bool",
    "complex64",
    "complex128",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


def _compute_discarded_fill(dtype: np.dtype) -> np.ndarray:
    """The value a discarded derivative slot is filled with, for one dtype.

    Whatever ``0/0`` yields there: NaN in every component for the inexact
    dtypes (so a complex slot is poisoned in its imaginary part too), and zero
    for those with no invalid value to spell.
    """
    zero = np.zeros((), dtype)
    with np.errstate(invalid="ignore"):
        fill = zero / zero if np.issubdtype(dtype, np.inexact) else zero
    # 0-d array, not the scalar that `/` returns, and read-only because it is
    # shared between calls.
    fill = np.asarray(fill, dtype=dtype)
    fill.flags.writeable = False
    return fill


# Materialised at import: the dtype domain is closed, so the table is complete
# and inspectable. Derived from the rule above so the two cannot drift.
_DISCARDED_FILL: dict[np.dtype, np.ndarray] = {
    np.dtype(name): _compute_discarded_fill(np.dtype(name)) for name in _SCHEMA_DTYPES
}


def _discarded_slot(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """A discarded slot in a derivative call's output tuple."""
    dtype = np.dtype(dtype)
    try:
        fill = _DISCARDED_FILL[dtype]
    except KeyError:
        # Not reachable through a Tesseract schema today. Raise deliberately
        # rather than let the bare KeyError out: this runs inside a host
        # callback, so whatever escapes reaches the user wrapped in an opaque
        # "INTERNAL: CpuCallback error calling callback".
        raise NotImplementedError(
            f"No discarded-slot fill defined for dtype {dtype}. Expected one "
            f"of: {', '.join(sorted(map(str, _DISCARDED_FILL)))}. This dtype "
            f"should not be reachable through a Tesseract schema, so please "
            f"report it."
        ) from None
    return np.full(shape, fill, dtype=dtype)


class Jaxeract:
    """A wrapper around a Tesseract client to make its signature compatible with JAX primitives."""

    def __init__(self, tesseract_client: Tesseract, *, cuda_ipc: bool = False) -> None:
        """Initialize the Tesseract client.

        ``cuda_ipc`` opts this call into exchanging GPU arrays with a served
        Tesseract via CUDA IPC handles instead of a host round-trip; it gates
        both the GPU FFI lowering and the :meth:`cuda_ipc` context below.
        """
        self.client = tesseract_client
        self._cuda_ipc = cuda_ipc

        self.tesseract_input_args = tuple(
            arg
            for arg in self.client.openapi_schema["components"]["schemas"][
                "Apply_InputSchema"
            ]["properties"]
        )
        # We need this to adhere to jax convention on tree flattening (sort keys alphabetically)
        # Only outermost level should be sufficient.
        self.tesseract_input_args = tuple(sorted(self.tesseract_input_args))

        self.tesseract_output_args = tuple(
            arg
            for arg in self.client.openapi_schema["components"]["schemas"][
                "Apply_OutputSchema"
            ]["properties"]
        )

        self.differentiable_input_paths = self.client.openapi_schema["components"][
            "schemas"
        ]["ApplyInputSchema"]["differentiable_arrays"]

        self.differentiable_output_paths = self.client.openapi_schema["components"][
            "schemas"
        ]["ApplyOutputSchema"]["differentiable_arrays"]

        self.available_methods = self.client.available_endpoints

    # Every attribute above is derived from ``self.client``, so two wrappers around
    # the same Tesseract are interchangeable. Saying so matters: this object is a
    # parameter of ``tesseract_dispatch_p``, and ``apply_tesseract`` builds a fresh
    # one per call, so with the inherited identity semantics two otherwise identical
    # calls lower to custom calls that XLA cannot recognise as equal and therefore
    # cannot common up. Equality is delegated rather than based on the URL or schema
    # so that distinct Tesseracts stay distinct; if ``Tesseract`` ever gains value
    # semantics of its own, this inherits them.
    def __eq__(self, other: object) -> bool:
        """Whether ``other`` wraps the same Tesseract in the same transport mode.

        ``_cuda_ipc`` participates: a cuda_ipc call and a host-transport call to
        the same Tesseract lower to different custom calls, so they must not
        compare equal or XLA would common them up.
        """
        if not isinstance(other, Jaxeract):
            return NotImplemented
        return self.client == other.client and self._cuda_ipc == other._cuda_ipc

    def __hash__(self) -> int:
        """Hash consistently with ``__eq__``."""
        return hash((Jaxeract, self.client, self._cuda_ipc))

    @contextlib.contextmanager
    def cuda_ipc(self) -> Generator[None]:
        """Temporarily make the underlying HTTP client use ``cuda_ipc`` encoding.

        Used by the GPU (FFI) lowering so that, for the duration of one dispatch,
        the client exports GPU array *inputs* via CUDA IPC handles and decodes
        ``cuda_ipc`` *outputs* back to GPU arrays -- no host round-trip. Two
        things must change and be restored:

        * ``_output_format`` (drives both the request encoder and response
          decoder), and
        * an ``Accept: application/json+cuda_ipc`` header, since the response
          format is otherwise the server's default and the client never sends
          Accept on its own.

        Scoped so the shared client is not permanently mutated (which would leak
        cuda_ipc behavior onto host-callback / CPU uses of the same client). A
        no-op when this call did not opt into ``cuda_ipc`` (:attr:`_cuda_ipc`),
        or for non-HTTP clients (e.g. the in-process ``LocalClient``).
        """
        client = getattr(self.client, "_client", None)
        if (
            not self._cuda_ipc
            or client is None
            or not hasattr(client, "_output_format")
        ):
            yield
            return
        prev_fmt = client._output_format
        session = getattr(client, "_session", None)
        had_accept = session is not None and "Accept" in session.headers
        prev_accept = session.headers.get("Accept") if session is not None else None

        client._output_format = "json+cuda_ipc"
        if session is not None:
            session.headers["Accept"] = "application/json+cuda_ipc"
        try:
            yield
        finally:
            client._output_format = prev_fmt
            if session is not None:
                if had_accept:
                    session.headers["Accept"] = prev_accept
                else:
                    session.headers.pop("Accept", None)

    # The abstract_eval method is never called from a dispatch function,
    # hence its signature does not need to be identical to the one of apply,
    # vjp and vjp.
    def abstract_eval(
        self,
        inputs: PyTree,
    ) -> PyTree:
        """Run an abstract evaluation on a Tesseract.

        This used in order to get output shapes given input shapes.
        """
        abstract_inputs = jax.tree.map(
            lambda x: (
                {"shape": x.shape, "dtype": x.dtype.name} if hasattr(x, "shape") else x
            ),
            inputs,
        )

        out_data = self.client.abstract_eval(abstract_inputs)
        return out_data

    def apply(
        self,
        array_args: tuple[TransportArray, ...],
        params: "DispatchParams",
    ) -> PyTree:
        """Call the Tesseract's apply endpoint with the given arguments."""
        inputs = unflatten_args(
            array_args,
            params.static_args,
            params.input_pytreedef,
            params.is_static_mask,
        )

        out_data = self.client.apply(inputs)

        if params.output_avals is None:
            return out_data

        out_data = tuple(jax.tree.flatten(out_data)[0])
        return out_data

    def jacobian_vector_product(
        self,
        array_args: tuple[TransportArray, ...],
        params: "DispatchParams",
    ) -> PyTree:
        """Call the Tesseract's jvp endpoint with the given arguments."""
        has_tangent = params.has_tangent
        n_primals = params.n_primals
        primals = array_args[:n_primals]
        # array_args[n_primals:] contains ALL tangents (zeroed for has_tangent=False).
        # Filter to only the has_tangent=True ones before calling combine_args, which
        # expects exactly sum(has_tangent) tangents.
        all_tangents = array_args[n_primals:]
        tangents = tuple(t for t, h in zip(all_tangents, has_tangent, strict=True) if h)

        # Expand filtered tangents back to full length using has_tangent:
        # positions where has_tangent=True get the tangent, False gets None
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
            tangent_inputs, schema_paths=self.differentiable_input_paths
        )
        flat_tangents = {p: v for p, v in flat_tangents.items() if v is not None}

        output_flat = _pytree_to_tesseract_flat(
            jax.tree.unflatten(
                params.output_pytreedef, range(len(params.output_avals))
            ),
            schema_paths=self.differentiable_output_paths,
        )

        jvp_outputs = [p for p, v in output_flat.items() if v is not None]

        out_data = self.client.jacobian_vector_product(
            inputs=primal_inputs,
            jvp_inputs=list(flat_tangents.keys()),
            jvp_outputs=jvp_outputs,
            tangent_vector=flat_tangents,
        )

        on_device = _on_device(array_args)
        out = []
        for path, aval in zip(output_flat, params.output_avals, strict=False):
            if path in out_data:
                out.append(out_data[path])
            else:
                out.append(_placeholder(aval.shape, aval.dtype, on_device=on_device))

        return tuple(out)

    def jacobian(
        self,
        array_args: tuple[TransportArray, ...],
        params: "DispatchParams",
    ) -> PyTree:
        """Call the Tesseract's jacobian endpoint with the given arguments.

        Returns one ndarray per (requested_output, requested_input) pair, in
        row-major order. Each array has shape ``out_shape + in_shape`` and
        dtype determined by ``params.jac_mode`` (``"bwd"`` matches ``jax.jacrev``
        and uses input dtype; ``"fwd"`` matches ``jax.jacfwd`` and uses
        output dtype). Mirrors lineax's mode names.
        """
        n_primals = params.n_primals
        primals = array_args[:n_primals]

        primal_inputs = unflatten_args(
            primals, params.static_args, params.input_pytreedef, params.is_static_mask
        )

        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=self.differentiable_input_paths
        )
        if params.jac_input_paths is None:
            jac_inputs = [p for p, v in flat_inputs.items() if v is not None]
        else:
            jac_inputs = list(params.jac_input_paths)

        output_flat = _pytree_to_tesseract_flat(
            jax.tree.unflatten(
                params.output_pytreedef, range(len(params.output_avals))
            ),
            schema_paths=self.differentiable_output_paths,
        )
        if params.jac_output_paths is None:
            jac_outputs = [p for p, v in output_flat.items() if v is not None]
        else:
            jac_outputs = list(params.jac_output_paths)

        out_data = self.client.jacobian(
            inputs=primal_inputs,
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
                out.append(_cast_return(out_data[op][ip], dtype=target))
        return tuple(out)

    def vector_jacobian_product(
        self,
        array_args: tuple[TransportArray, ...],
        params: "DispatchParams",
    ) -> PyTree:
        """Call the Tesseract's vjp endpoint with the given arguments."""
        has_tangent = params.has_tangent
        n_primals = params.n_primals
        primals = array_args[:n_primals]
        cotangents = array_args[n_primals:]

        primal_inputs = unflatten_args(
            primals, params.static_args, params.input_pytreedef, params.is_static_mask
        )

        flat_inputs = _pytree_to_tesseract_flat(
            primal_inputs, schema_paths=self.differentiable_input_paths
        )

        vjp_inputs = [
            p for p, m in zip(flat_inputs, params.is_static_mask, strict=True) if not m
        ]

        # now we filter for tangents
        vjp_inputs = [p for p, h in zip(vjp_inputs, has_tangent, strict=True) if h]

        cotangent_pytree = jax.tree.unflatten(params.output_pytreedef, cotangents)
        flat_cotangents = _pytree_to_tesseract_flat(
            cotangent_pytree, schema_paths=self.differentiable_output_paths
        )

        cotangents_dict = {p: v for p, v in flat_cotangents.items() if v is not None}

        out_data = self.client.vector_jacobian_product(
            inputs=primal_inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=list(cotangents_dict.keys()),
            cotangent_vector=cotangents_dict,
        )

        # JAX expects gradients for all inputs, even non-differentiable ones.
        # Reconstruct the full output tuple in the same order as flat_inputs.
        out = []
        # all_idx indexes into flat_inputs, none_mask, and is_static_mask
        array_idx = 0  # Index into array_args (which excludes static inputs)
        tan_idx = 0  # Index into tangents/cotangents (which excludes non-differentiable inputs)
        for all_idx, path in enumerate(flat_inputs):
            if path in out_data:
                # Path has a gradient from the server
                out.append(out_data[path])
                tan_idx += 1
            elif (
                tan_idx < len(has_tangent)
                and not params.is_static_mask[all_idx]
                and not has_tangent[tan_idx]
            ):
                # Non-differentiable but non-static input: emit a placeholder of
                # the same shape/dtype as the corresponding input array. The slot
                # exists for the tuple-length contract; JAX's transpose machinery
                # doesn't consume it for any user-requested derivative.
                arg = array_args[array_idx]
                out.append(
                    _placeholder(arg.shape, arg.dtype, on_device=_on_device([arg]))
                )
                tan_idx += 1

            # Increment array_idx only for non-static inputs (which appear in array_args)
            if not params.is_static_mask[all_idx]:
                array_idx += 1

        return tuple(out)
