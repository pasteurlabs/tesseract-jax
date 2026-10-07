# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import TYPE_CHECKING

import jax.tree
import numpy as np
from tesseract_core import Tesseract

from tesseract_jax.dce import live_jvp_output_positions
from tesseract_jax.tree_util import (
    PyTree,
    TransportArray,
    combine_args,
    dummy_output_tree,
    pytree_to_path_dict,
    split_args,
    unflatten_args,
    warn_on_static_output_drift,
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

    The ``cuda`` import is deliberately lazy, not at module scope: eagerly
    importing ``tesseract_core.runtime.cuda.ipc`` perturbs schema/typeguard state
    in the shared interpreter and breaks in-process (``LocalClient``) Tesseracts
    whose endpoints use ellipsis-shaped array schemas.
    """
    from tesseract_core.runtime.cuda.ipc import has_cuda_array_interface

    return any(has_cuda_array_interface(v) for v in values)


def _to_host(value: TransportArray) -> TransportArray:
    """Copy a GPU array to a host NumPy array, and return anything else as is.

    Lazy import for the same reason as in :func:`_on_device`.
    """
    from tesseract_core.runtime.cuda.ipc import (
        cuda_array_to_host,
        has_cuda_array_interface,
    )

    if has_cuda_array_interface(value):
        return cuda_array_to_host(value)
    return value


def _cast_return(
    value: TransportArray, *, dtype: np.dtype, ffi_path: bool
) -> TransportArray:
    """Coerce a dispatch result to the return ``dtype``.

    On the GPU FFI path (``ffi_path``) a device ``value`` is returned as is,
    since casting it would force a device->host round-trip. tesseract-core does
    not pin a jacobian endpoint's output dtype, so the native shim checks the
    result's dtype and shape against XLA's output buffer instead and raises on a
    mismatch (see ``_cuda_shim.cc``). Every other ``value`` is copied to the host
    if needed and cast there.
    """
    if ffi_path and _on_device([value]):
        return value
    return np.asarray(_to_host(value), dtype=dtype)


def _placeholder(
    shape: tuple[int, ...], dtype: np.dtype, *, ffi_path: bool
) -> TransportArray | None:
    """A discarded slot in a derivative call's output tuple.

    Used for the gradient of a non-differentiable input and the tangent of a
    non-differentiable output. Such a slot exists only to satisfy the
    output-tuple-length contract; JAX's transpose machinery never consumes it for
    any user-requested derivative, so its value is immaterial.

    On the GPU FFI path (``ffi_path``) this returns ``None`` and the native FFI
    handler fills XLA's output buffer for that slot directly. On the host path it
    returns an array filled with each dtype's ``0/0`` value (see
    :func:`_discarded_slot`), so an accidental consumer surfaces loudly rather
    than silently.
    """
    if ffi_path:
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

    def __init__(self, tesseract_client: Tesseract) -> None:
        """Initialize the Tesseract client."""
        self.client = tesseract_client

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
    # semantics of its own, this inherits them. The GPU transport needs no part in
    # it, since it follows from the Tesseract (see ``resolve_gpu_transport``).
    def __eq__(self, other: object) -> bool:
        """Whether ``other`` wraps the same Tesseract."""
        if not isinstance(other, Jaxeract):
            return NotImplemented
        return self.client == other.client

    def __hash__(self) -> int:
        """Hash consistently with ``__eq__``."""
        return hash((Jaxeract, self.client))

    def resolve_gpu_transport(self) -> str:
        """The GPU transport a call lowered for a CUDA device uses, or ``"none"``.

        Whatever ``Tesseract.resolve_gpu_transport`` picks: the transport the
        Tesseract requests, else ``cuda_ipc`` if the Tesseract offers it and it
        works from this process.
        """
        gpu_transport = self.client.resolve_gpu_transport()
        if gpu_transport != "none" and gpu_transport not in _SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"The Tesseract requests gpu_transport={gpu_transport!r}, which "
                f"tesseract-jax cannot drive (supported: "
                f"{['none', *sorted(_SUPPORTED_TRANSPORTS)]}). Pass "
                "tesseract.with_encoding(gpu_transport='none') to apply_tesseract "
                "to copy GPU arrays to the host instead."
            )
        return gpu_transport

    def with_gpu_transport(self, gpu_transport: str) -> "Jaxeract":
        """A wrapper whose calls exchange GPU arrays over ``gpu_transport``.

        A device transport sends GPU array inputs by reference and asks for
        outputs the same way; ``"none"`` keeps both on the host. This wrapper
        itself if its Tesseract already requests that, else one around a view
        of it. Built per call from the current Tesseract, so a compiled function
        keeps working after the Tesseract is served again.
        """
        if self.client.server_capabilities is None:
            # In-process, arrays are passed as they are, so there is no
            # encoding to change.
            return self
        if (self.client.current_encoding.gpu_transport or "none") == gpu_transport:
            return self
        view = copy.copy(self)
        view.client = self.client.with_encoding(gpu_transport=gpu_transport)
        return view

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
        static_output_mask = params.static_output_mask
        inputs = unflatten_args(
            array_args,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
        )

        out_data = self.client.apply(inputs)

        # Keypaths are only needed to name a field in the drift warning, so build
        # them only when that warning can fire.
        checking = params.check_static_outputs and any(static_output_mask)
        if checking:
            leaves_with_path = jax.tree_util.tree_flatten_with_path(out_data)[0]
            out_data = tuple(leaf for _, leaf in leaves_with_path)
        else:
            out_data = tuple(jax.tree.leaves(out_data))

        if any(static_output_mask):
            # A JAX primitive can only return arrays, so drop the response's
            # static leaves here; apply_tesseract puts back the values
            # abstract_eval reported once the bind has returned.
            out_data, returned_statics = split_args(out_data, static_output_mask)
            if checking:
                _, static_paths = split_args(
                    tuple(path for path, _ in leaves_with_path), static_output_mask
                )
                warn_on_static_output_drift(
                    static_paths, returned_statics, params.static_output_values
                )
        return out_data

    def jacobian_vector_product(
        self,
        array_args: tuple[TransportArray, ...],
        params: "DispatchParams",
    ) -> PyTree:
        """Call the Tesseract's jvp endpoint with the given arguments.

        ``params.live_output_paths`` (set by the DCE rule) restricts the request
        to the output tangents that survive dead-code elimination. ``None``
        requests all differentiable outputs (the un-pruned default). The returned
        tuple is aligned to the live output leaves in ``output_avals`` order — see
        :func:`tesseract_jax.dce.live_jvp_output_positions`.
        """
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
            primals,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
        )
        tangent_inputs = unflatten_args(
            full_tangents,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
            remove_static_args=True,
        )

        flat_tangents = pytree_to_path_dict(
            tangent_inputs, schema_paths=self.differentiable_input_paths
        )
        flat_tangents = {p: v for p, v in flat_tangents.items() if v is not None}

        output_flat = pytree_to_path_dict(
            dummy_output_tree(
                params.output_pytreedef,
                len(params.output_avals),
                params.static_output_mask,
            ),
            schema_paths=self.differentiable_output_paths,
        )

        # Emit only the output tangents that survived DCE (``live_output_paths``).
        # ``live_jvp_output_positions`` is the single source of truth for which
        # leaves we return and in what order; abstract_eval sizes its result the
        # same way.
        live_positions = live_jvp_output_positions(
            params.output_pytreedef,
            len(params.output_avals),
            self.differentiable_output_paths,
            params.live_output_paths,
            params.static_output_mask,
        )
        flat_items = list(output_flat.items())

        # Only differentiable live leaves are requested from the Tesseract;
        # non-differentiable leaves (if any) are NaN-padded below.
        jvp_outputs = [
            flat_items[pos][0]
            for pos in live_positions
            if flat_items[pos][1] is not None
        ]

        out_data = self.client.jacobian_vector_product(
            inputs=primal_inputs,
            jvp_inputs=list(flat_tangents.keys()),
            jvp_outputs=jvp_outputs,
            tangent_vector=flat_tangents,
        )

        # Emit exactly the live leaves, in ``live_positions`` order, so the tuple
        # lines up with what abstract_eval declared. A non-differentiable live leaf
        # (never requested from the Tesseract) gets a placeholder.
        ffi_path = _on_device(array_args)
        out = []
        for pos in live_positions:
            path = flat_items[pos][0]
            if path in out_data:
                out.append(out_data[path])
            else:
                aval = params.output_avals[pos]
                out.append(_placeholder(aval.shape, aval.dtype, ffi_path=ffi_path))

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
            primals,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
        )

        flat_inputs = pytree_to_path_dict(
            primal_inputs, schema_paths=self.differentiable_input_paths
        )
        if params.live_input_paths is None:
            jac_inputs = [p for p, v in flat_inputs.items() if v is not None]
        else:
            jac_inputs = list(params.live_input_paths)

        output_flat = pytree_to_path_dict(
            dummy_output_tree(
                params.output_pytreedef,
                len(params.output_avals),
                params.static_output_mask,
            ),
            schema_paths=self.differentiable_output_paths,
        )
        if params.live_output_paths is None:
            jac_outputs = [p for p, v in output_flat.items() if v is not None]
        else:
            jac_outputs = list(params.live_output_paths)

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
        ffi_path = _on_device(array_args)
        out = []
        for op in jac_outputs:
            for ip in jac_inputs:
                target = (
                    ip_to_dtype[ip] if params.jac_mode == "bwd" else op_to_dtype[op]
                )
                out.append(
                    _cast_return(out_data[op][ip], dtype=target, ffi_path=ffi_path)
                )
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
            primals,
            params.static_args,
            params.input_pytreedef,
            params.static_input_mask,
        )

        flat_inputs = pytree_to_path_dict(
            primal_inputs, schema_paths=self.differentiable_input_paths
        )

        vjp_inputs = [
            p
            for p, m in zip(flat_inputs, params.static_input_mask, strict=True)
            if not m
        ]

        # now we filter for tangents
        vjp_inputs = [p for p, h in zip(vjp_inputs, has_tangent, strict=True) if h]

        # Scatter cotangents back to full non-static-output width, inserting
        # ``None`` where the cotangent was a symbolic zero.
        if params.has_cotangent:
            assert len(cotangents) == sum(params.has_cotangent)
            cotan_iter = iter(cotangents)
            cotangents = tuple(
                next(cotan_iter) if h else None for h in params.has_cotangent
            )

        # A static output leaf carries no cotangent, so fill its slot with None.
        # None is an empty pytree node and drops back out when the tree is
        # flattened into schema paths, keeping static outputs out of them.
        if any(params.static_output_mask):
            cotangents = combine_args(
                tuple(cotangents),
                (None,) * sum(params.static_output_mask),
                params.static_output_mask,
            )
        cotangent_pytree = jax.tree.unflatten(params.output_pytreedef, cotangents)
        flat_cotangents = pytree_to_path_dict(
            cotangent_pytree, schema_paths=self.differentiable_output_paths
        )

        cotangents_dict = {p: v for p, v in flat_cotangents.items() if v is not None}

        out_data = self.client.vector_jacobian_product(
            inputs=primal_inputs,
            vjp_inputs=vjp_inputs,
            vjp_outputs=list(cotangents_dict.keys()),
            cotangent_vector=cotangents_dict,
        )

        # Only differentiated inputs carry a cotangent back. A non-differentiated
        # input's slot is never consumed by JAX's transpose, so we omit it here;
        # abstract_eval declares the matching (shorter) output arity and the
        # transpose rule scatters these back into full primal order.
        return tuple(out_data[path] for path in flat_inputs if path in out_data)
