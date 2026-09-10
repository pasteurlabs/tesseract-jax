# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING

import jax.tree
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax.tree_util import (
    PyTree,
    TransportArray,
    _pytree_to_tesseract_flat,
    combine_args,
    unflatten_args,
)

if TYPE_CHECKING:
    from tesseract_jax.dispatch_params import DispatchParams

# WARNING: Do NOT use jax.numpy within Jaxeract methods, as they are executed from within FFI callbacks
# and cannot safely allocate JAX arrays. Use vanilla numpy instead.


def _on_device(values: "list | tuple") -> bool:
    """Whether ``values`` are cuda_ipc device arrays (vs host NumPy arrays).

    The endpoint methods are transport-agnostic; this distinguishes the GPU FFI
    lowering (bare ``__cuda_array_interface__`` device views / ``IpcDeviceArray``
    results) from the CPU host-callback lowering (real NumPy arrays).

    The ``cuda.ipc`` import is deliberately lazy, not at module scope: eagerly
    importing ``tesseract_core.runtime.cuda.ipc`` perturbs schema/typeguard state
    in the shared interpreter and breaks in-process (``LocalClient``) Tesseracts
    whose endpoints use ellipsis-shaped array schemas.
    """
    from tesseract_core.runtime.cuda.ipc import has_cuda_array_interface

    return any(has_cuda_array_interface(v) for v in values)


def _cast_return(value: TransportArray, *, dtype: np.dtype) -> TransportArray:
    """Coerce a dispatch result to the return ``dtype`` without leaving the device.

    On the host path ``value`` is a NumPy array and we cast it to ``dtype`` here.

    On the cuda_ipc (GPU FFI) path ``value`` is a device array whose bytes the
    FFI handler copies straight into XLA's output buffer, so we cannot cast it
    here (that would force a device->host round-trip) and return it untouched.
    This is *not* a guarantee that the device array already has ``dtype``:
    tesseract-core is deliberately not prescriptive about a jacobian endpoint's
    output dtype, so a Tesseract may return e.g. float64 where float32 was
    declared. The native shim closes that gap -- it compares each returned
    array's dtype and shape against the XLA output buffer and raises on a
    mismatch rather than reinterpreting bytes (see ``_cuda_shim.cc``).
    """
    if _on_device([value]):
        return value
    return np.asarray(value, dtype=dtype)


def _placeholder(
    shape: tuple[int, ...], dtype: np.dtype, *, on_device: bool
) -> TransportArray | None:
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


# Device transports the GPU (FFI) lowering supports end-to-end. cuda_ipc is the
# only one wired through the native shim today; add names here as the FFI path
# learns to drive them.
_SUPPORTED_TRANSPORTS = frozenset({"cuda_ipc"})


class Jaxeract:
    """A wrapper around a Tesseract client to make its signature compatible with JAX primitives."""

    def __init__(
        self,
        tesseract_client: Tesseract,
        *,
        cuda_ipc: bool = False,
        device_transport: str | None = None,
    ) -> None:
        """Initialize the Tesseract client.

        ``device_transport`` names the on-device transport used to exchange GPU
        arrays with a served Tesseract instead of a host round-trip (e.g.
        ``"cuda_ipc"``). It selects one of the runtime's registered device
        transports (see :mod:`tesseract_core.runtime.device_transport`) and gates
        both the GPU FFI lowering and the :meth:`device_transport_encoding`
        context below. ``cuda_ipc=True`` is the back-compatible spelling of
        ``device_transport="cuda_ipc"``.
        """
        if cuda_ipc and device_transport not in (None, "cuda_ipc"):
            raise ValueError(
                "Pass either cuda_ipc=True or device_transport=..., not both "
                f"conflicting values (got cuda_ipc=True, "
                f"device_transport={device_transport!r})."
            )
        if cuda_ipc and device_transport is None:
            device_transport = "cuda_ipc"

        # Only transports the GPU (FFI) lowering actually implements end-to-end
        # are accepted. The lowering is currently cuda_ipc-specific, so an
        # unsupported name would otherwise route silently into that path and send
        # an ``Accept: application/json+<name>`` the server has no backend for.
        # Extend this set as further transports are wired through the FFI path.
        if (
            device_transport is not None
            and device_transport not in _SUPPORTED_TRANSPORTS
        ):
            raise ValueError(
                f"Unsupported device_transport {device_transport!r}; "
                f"supported: {sorted(_SUPPORTED_TRANSPORTS)}."
            )

        self.client = tesseract_client
        # The transport name (None = host round-trip). ``_cuda_ipc`` is kept as a
        # bool alias so existing call sites and equality/hash keep working.
        self._device_transport = device_transport
        self._cuda_ipc = device_transport is not None

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
        return (
            self.client == other.client
            and self._device_transport == other._device_transport
        )

    def __hash__(self) -> int:
        """Hash consistently with ``__eq__``."""
        return hash((Jaxeract, self.client, self._device_transport))

    @contextlib.contextmanager
    def device_transport_encoding(self) -> Generator[None]:
        """Temporarily make the HTTP client use this call's device transport.

        Used by the GPU (FFI) lowering so that, for the duration of one dispatch,
        the client exports GPU array *inputs* by reference through the negotiated
        device transport and the served Tesseract hands the *outputs* back the
        same way -- no host round-trip. tesseract-core keeps these two axes
        orthogonal: ``_output_format`` governs how CPU arrays are serialized and
        ``_gpu_transport`` selects how GPU arrays leave the process. So two things
        must change and be restored:

        * ``_gpu_transport`` (drives the request encoder's per-leaf GPU export),
          and
        * an ``Accept: application/<output_format>; gpu_transport=<transport>``
          header, which negotiates the server's GPU output transport per request.
          The client never sends Accept on its own and the served Tesseract may
          default to ``gpu_transport="none"``, so without the header the response
          would come back host-copied.

        ``_output_format`` is left untouched: it still names the CPU-array
        encoding, and the ``Accept`` media type reuses it verbatim so the CPU
        leaves of a mixed response are unaffected.

        Scoped so the shared client is not permanently mutated (which would leak
        the on-device transport onto host-callback / CPU uses of the same client).
        A no-op when this call did not opt into a device transport
        (:attr:`_device_transport`), or for non-HTTP clients (e.g. the in-process
        ``LocalClient``).
        """
        client = getattr(self.client, "_client", None)
        if (
            self._device_transport is None
            or client is None
            or not hasattr(client, "_gpu_transport")
        ):
            yield
            return
        prev_transport = client._gpu_transport
        output_format = getattr(client, "_output_format", "json+base64")
        session = getattr(client, "_session", None)
        had_accept = session is not None and "Accept" in session.headers
        prev_accept = session.headers.get("Accept") if session is not None else None

        client._gpu_transport = self._device_transport
        if session is not None:
            session.headers["Accept"] = (
                f"application/{output_format}; gpu_transport={self._device_transport}"
            )
        try:
            yield
        finally:
            client._gpu_transport = prev_transport
            if session is not None:
                if had_accept:
                    session.headers["Accept"] = prev_accept
                else:
                    session.headers.pop("Accept", None)

    # Back-compat alias: the GPU lowering historically calls ``client.cuda_ipc()``.
    cuda_ipc = device_transport_encoding

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
