# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared JAX front end for GPU-direct Tesseract dispatch.

This module owns everything both strategies share:

* the descriptor registry that maps an integer FFI ``token`` to a live dispatch
  descriptor (the client, the flattening metadata, the chosen strategy);
* the single native dispatch callback the FFI handler invokes, which routes to
  the strategy behind the descriptor;
* ``apply_gpu_direct``, the ``apply``-only entry point that lowers to one
  ``jax.ffi.ffi_call`` into the native handler.

The two strategies (A: Python does the ``cuda_ipc`` handshake, B: native code
does it) are the *only* thing that differs between runs, so a fair comparison
holds everything here fixed and swaps ``impl``.

Scope: this is milestone-1 scope from the spec -- ``apply`` only, GPU-direct
outputs, explicit opt-in. Derivatives / batching stay on the existing
host-callback path. The goal is to measure A vs B, not to re-home all of
Tesseract-JAX onto FFI.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import numpy as np

from tesseract_jax_gpu import ensure_registered, set_dispatch_callback

# Strategy B lives in a separate Rust/PyO3 module with its own FFI target.
_B_TARGET_NAME = "tesseract_jax_cuda_ipc_dispatch_b"
_b_registered = False


def _ensure_b_registered() -> str:
    """Register the Rust (Strategy B) FFI target with XLA (idempotent)."""
    global _b_registered
    if not _b_registered:
        import jax

        import tesseract_jax_gpu_b as b

        jax.ffi.register_ffi_target(
            _B_TARGET_NAME, b.handler_capsule(), platform="CUDA"
        )
        _b_registered = True
    return _B_TARGET_NAME

# ---------------------------------------------------------------------------
# Descriptor registry (token -> descriptor)
# ---------------------------------------------------------------------------


@dataclass
class _Descriptor:
    """Everything the native callback needs to run one dispatch shape."""

    # Callable[[list[tuple[ptr, typestr, shape]]], list[cuda-array]]
    run: Callable[[list[tuple[int, str, tuple[int, ...]]]], list[Any]]
    # Kept for debugging / introspection.
    eval_func: str
    impl: str
    out_avals: tuple[Any, ...] = field(default=())


_registry: dict[int, _Descriptor] = {}
_registry_lock = threading.Lock()
_next_token = 0


def _register_descriptor(desc: _Descriptor) -> int:
    global _next_token
    with _registry_lock:
        token = _next_token
        _next_token += 1
        _registry[token] = desc
    return token


def _release_descriptor(token: int) -> None:
    with _registry_lock:
        _registry.pop(token, None)


# ---------------------------------------------------------------------------
# The single native dispatch callback
# ---------------------------------------------------------------------------


def _native_dispatch(token: int, inputs: list[tuple[int, str, tuple[int, ...]]]):
    """Invoked by the native FFI handler under the GIL.

    ``inputs`` is a list of (device_ptr, numpy_typestr, shape) describing the
    XLA input buffers (still on device). Returns a list of arrays exposing
    ``__cuda_array_interface__`` whose bytes the handler copies into the XLA
    output buffers.
    """
    desc = _registry.get(token)
    if desc is None:
        raise RuntimeError(f"tesseract_jax_gpu: unknown dispatch token {token}")
    return desc.run(inputs)


_callback_installed = False


def _ensure_callback() -> None:
    global _callback_installed
    if not _callback_installed:
        set_dispatch_callback(_native_dispatch)
        _callback_installed = True


# ---------------------------------------------------------------------------
# Reusable (jittable) GPU-direct apply
# ---------------------------------------------------------------------------


def make_gpu_apply(tesseract_client, inputs_template: dict, *, impl: str = "A"):
    """Build a reusable GPU-direct ``apply`` callable for a fixed input signature.

    Unlike :func:`apply_gpu_direct` (which registers a descriptor and traces an
    ``ffi_call`` on every call), this resolves the output structure once, installs
    a persistent descriptor with a stable token, and returns a callable that just
    issues the ``ffi_call``. The returned callable is ``jax.jit``-friendly: under
    ``jit`` the FFI lowering is traced/compiled once and reused, so steady-state
    per-call latency reflects only the actual dispatch (not JAX tracing).

    This is the shape a real optimization/sampling loop uses, and the one the
    benchmark measures. Returns ``(apply_fn, close)`` where ``apply_fn(inputs)``
    returns the output dict and ``close()`` releases the descriptor.
    """
    _, in_tree = jax.tree.flatten(inputs_template)
    abstract_inputs = jax.tree.map(
        lambda x: {"shape": tuple(x.shape), "dtype": x.dtype.name}
        if hasattr(x, "shape")
        else x,
        inputs_template,
    )
    out_abstract = tesseract_client.abstract_eval(abstract_inputs)

    def _is_aval(x):
        return isinstance(x, dict) and "shape" in x and "dtype" in x

    flat_out_avals, out_tree = jax.tree.flatten(out_abstract, is_leaf=_is_aval)
    out_sds = tuple(
        jax.ShapeDtypeStruct(tuple(a["shape"]), np.dtype(a["dtype"]))
        for a in flat_out_avals
    )

    if impl == "B":
        import tesseract_jax_gpu_b as b

        target = _ensure_b_registered()
        http = getattr(tesseract_client, "_client", None)
        if http is None or not hasattr(http, "url"):
            raise RuntimeError("Strategy B requires an HTTP-served Tesseract.")
        input_keys = sorted(inputs_template.keys())
        output_keys = _flat_output_keys(out_tree, out_sds)
        global _next_token
        with _registry_lock:
            token = _next_token
            _next_token += 1
        b.register_descriptor(token, f"{http.url}/apply", input_keys, output_keys)

        def close():
            b.release_descriptor(token)
    else:
        from tesseract_jax_gpu.strategies import make_runner

        target = ensure_registered()
        _ensure_callback()
        runner = make_runner(
            "A",
            tesseract_client=tesseract_client,
            inputs_template=inputs_template,
            in_tree=in_tree,
            out_tree=out_tree,
            out_avals=out_sds,
        )
        desc = _Descriptor(run=runner, eval_func="apply", impl="A", out_avals=out_sds)
        token = _register_descriptor(desc)

        def close():
            _release_descriptor(token)

    token_arr = np.int64(token)

    def apply_fn(inputs: dict) -> dict:
        flat_inputs, _ = jax.tree.flatten(inputs)
        outs = jax.ffi.ffi_call(target, list(out_sds), has_side_effect=True)(
            *flat_inputs, token=token_arr
        )
        return jax.tree.unflatten(out_tree, outs)

    return apply_fn, close


# ---------------------------------------------------------------------------
# apply-only GPU-direct entry point (one-shot convenience)
# ---------------------------------------------------------------------------


def apply_gpu_direct(
    tesseract_client,
    inputs: dict,
    *,
    impl: str = "A",
) -> dict:
    """Apply a Tesseract to ``inputs`` keeping data on the GPU.

    Eager (non-traced) ``apply`` only. ``inputs`` is a dict of JAX arrays on the
    GPU. ``impl`` selects the dispatch strategy ("A" or "B"). Returns a dict of
    JAX arrays on the GPU.

    This lowers to a single ``jax.ffi.ffi_call`` into the native handler. The
    handler hands the input device pointers to the strategy's dispatch function,
    which returns result arrays on the GPU; the handler copies them into the XLA
    output buffers.
    """
    # Flatten inputs in a stable (sorted-key) order matching Tesseract's schema.
    flat_inputs, in_tree = jax.tree.flatten(inputs)

    # Output structure/avals via the Tesseract's abstract_eval endpoint.
    abstract_inputs = jax.tree.map(
        lambda x: {"shape": tuple(x.shape), "dtype": x.dtype.name}
        if hasattr(x, "shape")
        else x,
        inputs,
    )
    out_abstract = tesseract_client.abstract_eval(abstract_inputs)

    def _is_aval(x):
        return isinstance(x, dict) and "shape" in x and "dtype" in x

    flat_out_avals, out_tree = jax.tree.flatten(out_abstract, is_leaf=_is_aval)
    out_sds = tuple(
        jax.ShapeDtypeStruct(tuple(a["shape"]), np.dtype(a["dtype"]))
        for a in flat_out_avals
    )

    if impl == "B":
        return _apply_b(
            tesseract_client, inputs, flat_inputs, out_tree, out_sds
        )

    return _apply_a(
        tesseract_client, inputs, flat_inputs, in_tree, out_tree, out_sds
    )


def _apply_a(tesseract_client, inputs, flat_inputs, in_tree, out_tree, out_sds):
    """Strategy A: C++ FFI handler + Python cuda_ipc dispatch callback."""
    from tesseract_jax_gpu.strategies import make_runner

    target = ensure_registered()
    _ensure_callback()

    runner = make_runner(
        "A",
        tesseract_client=tesseract_client,
        inputs_template=inputs,
        in_tree=in_tree,
        out_tree=out_tree,
        out_avals=out_sds,
    )
    desc = _Descriptor(run=runner, eval_func="apply", impl="A", out_avals=out_sds)
    token = _register_descriptor(desc)

    try:
        outs = jax.ffi.ffi_call(
            target, list(out_sds), has_side_effect=True
        )(*flat_inputs, token=np.int64(token))
        outs = jax.tree.map(lambda x: x.block_until_ready(), outs)
    finally:
        _release_descriptor(token)

    return jax.tree.unflatten(out_tree, outs)


def _apply_b(tesseract_client, inputs, flat_inputs, out_tree, out_sds):
    """Strategy B: Rust FFI handler does the whole dispatch natively (no GIL).

    Python's only job here is to register a descriptor (URL + flat input/output
    key order) keyed by a token, then issue the ffi_call. The Rust handler reads
    the descriptor, does base64 encode + HTTP + cuda_ipc decode + device copy,
    and never re-enters Python.
    """
    import tesseract_jax_gpu_b as b

    target = _ensure_b_registered()

    # HTTP endpoint + flat input/output key order for the Rust side.
    http = getattr(tesseract_client, "_client", None)
    if http is None or not hasattr(http, "url"):
        raise RuntimeError(
            "Strategy B requires an HTTP-served Tesseract "
            "(Tesseract.from_url / from_image().serve())."
        )
    apply_url = f"{http.url}/apply"
    input_keys = sorted(inputs.keys())
    output_keys = _flat_output_keys(out_tree, out_sds)

    global _next_token
    with _registry_lock:
        token = _next_token
        _next_token += 1
    b.register_descriptor(token, apply_url, input_keys, output_keys)

    try:
        outs = jax.ffi.ffi_call(
            target, list(out_sds), has_side_effect=True
        )(*flat_inputs, token=np.int64(token))
        outs = jax.tree.map(lambda x: x.block_until_ready(), outs)
    finally:
        b.release_descriptor(token)

    return jax.tree.unflatten(out_tree, outs)


def _flat_output_keys(out_tree, out_sds) -> list[str]:
    """Flat output leaf names in the order XLA passes output buffers.

    Milestone-1 scope: a flat dict of array outputs. The Tesseract returns a
    dict; jax.tree.flatten uses sorted keys, matching the buffer order.
    """
    indices = jax.tree.unflatten(out_tree, range(len(out_sds)))
    if not isinstance(indices, dict):
        raise NotImplementedError(
            "Strategy B currently supports a flat dict of array outputs."
        )
    # Position i -> key whose leaf index is i.
    keys_by_pos: list[str] = [""] * len(out_sds)
    for k in sorted(indices.keys()):
        keys_by_pos[indices[k]] = k
    return keys_by_pos
