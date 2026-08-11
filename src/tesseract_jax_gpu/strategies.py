# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dispatch seam: the two strategies under comparison.

Both strategies keep data on the GPU and use tesseract-core's ``cuda_ipc``
encoding. They share the native FFI shim and the tested CUDA IPC primitives from
``tesseract_core.runtime.array_encoding``. They differ only in *dispatch
orchestration*:

* **Strategy A** reuses the full ``tesseract_core`` HTTP client. One call to
  ``Tesseract.apply`` runs the client's generic pytree encode (``_tree_map`` +
  per-leaf ``__cuda_array_interface__`` probing), ``orjson`` round-trip, HTTP
  POST, and generic pytree decode. Maximal reuse; the per-call cost includes the
  client's Python tree-walking regardless of payload.

* **Strategy B** does a lean bespoke dispatch. It knows the flat input arrays
  and output leaf paths up front, so it encodes inputs and decodes outputs
  directly (no generic ``_tree_map`` walk, no ``hasattr`` probing per leaf) and
  POSTs over a persistent session. The IPC handshake itself reuses the same
  ``array_encoding`` ctypes primitives as A -- reimplementing
  ``cudaIpcGetMemHandle`` natively would duplicate tested code for no
  correctness gain -- but everything around it is stripped down.

Both return a list of GPU arrays (exposing ``__cuda_array_interface__``) in the
Tesseract's flat output order, which the FFI handler copies into XLA's output
buffers.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_runner(impl: str, **kwargs) -> Callable[[list], list]:
    """Build the per-dispatch runner for the chosen strategy."""
    if impl == "A":
        return _make_runner_a(**kwargs)
    if impl == "B":
        return _make_runner_b(**kwargs)
    raise ValueError(f"unknown impl {impl!r}; expected 'A' or 'B'")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _cupy_view(ptr: int, typestr: str, shape: tuple[int, ...]):
    """Wrap a raw device pointer as an unowned cupy view (no copy)."""
    import cupy

    dtype = np.dtype(typestr)
    n = int(np.prod(shape)) if shape else 1
    nbytes = n * dtype.itemsize
    mem = cupy.cuda.UnownedMemory(ptr, nbytes, owner=None)
    memptr = cupy.cuda.MemoryPointer(mem, 0)
    return cupy.ndarray(tuple(shape), dtype=dtype, memptr=memptr)


def _input_views(inputs, input_keys):
    """Map (ptr, typestr, shape) tuples to a dict keyed by Tesseract input name.

    ``input_keys`` is the flat, sorted key order the front end used to flatten
    the input dict, which matches jax.tree.flatten on a dict (sorted keys).
    """
    views = {}
    for key, (ptr, typestr, shape) in zip(input_keys, inputs, strict=True):
        views[key] = _cupy_view(ptr, typestr, tuple(shape))
    return views


# ---------------------------------------------------------------------------
# Input direction (milestone-1 scope: base64 inputs, cuda_ipc outputs)
# ---------------------------------------------------------------------------
#
# tesseract-core's server currently rejects cuda_ipc-*encoded inputs*: the
# dynamically-built input validation model omits CudaIpcArrayData from its data
# union (array_encoding.get_array_model). Until that is fixed upstream, GPU
# inputs must be sent host-side (base64), which still leaves the output path
# fully on-device -- the larger and well-tested win. We therefore materialize
# input views to host here. When the upstream fix lands, this host copy can be
# dropped for true end-to-end GPU-direct inputs.


def _view_to_host(view) -> np.ndarray:
    """Copy a cupy device view to a host numpy array."""
    return view.get()


# ---------------------------------------------------------------------------
# Strategy A: full tesseract-core client reuse
# ---------------------------------------------------------------------------


def _make_runner_a(
    *,
    tesseract_client,
    inputs_template: dict,
    in_tree,
    out_tree,
    out_avals,
):
    input_keys = sorted(inputs_template.keys())

    # Flat output leaf paths in Tesseract order (dotted), so we can pull the
    # decoded arrays out of the returned dict in the right order.
    out_leaf_getters = _out_leaf_getters(out_tree, out_avals)

    def run(inputs: list) -> list:
        # Inputs host-side (base64); outputs come back cuda_ipc (on GPU).
        host_inputs = {
            key: _view_to_host(_cupy_view(ptr, typestr, tuple(shape)))
            for key, (ptr, typestr, shape) in zip(input_keys, inputs, strict=True)
        }
        # Request cuda_ipc *outputs* for the duration of this call only, so the
        # shared client's default output format is not permanently mutated
        # (which would break other, non-GPU-direct uses of the same client).
        with _cuda_ipc_output(tesseract_client):
            result = tesseract_client.apply(host_inputs)
        return [get(result) for get in out_leaf_getters]

    return run


# ---------------------------------------------------------------------------
# Strategy B: lean bespoke dispatch
# ---------------------------------------------------------------------------


def _make_runner_b(
    *,
    tesseract_client,
    inputs_template: dict,
    in_tree,
    out_tree,
    out_avals,
):
    import orjson
    import pybase64
    from tesseract_core.runtime.array_encoding import _load_cuda_ipc_arraydict

    input_keys = sorted(inputs_template.keys())
    url, session = _http_endpoint(tesseract_client)
    apply_url = f"{url}/apply"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json+cuda_ipc",
    }
    out_leaf_getters = _out_leaf_getters(out_tree, out_avals)

    def _encode_base64_input(view) -> dict:
        # Host-side base64 encode (see input-direction note above). Lean: we know
        # the leaf is an array, so we skip the client's generic pytree probing.
        host = view.get()
        host = np.ascontiguousarray(host)
        return {
            "object_type": "array",
            "shape": list(host.shape),
            "dtype": host.dtype.name,
            "data": {
                "buffer": pybase64.b64encode_as_string(host.data),
                "encoding": "base64",
            },
        }

    def run(inputs: list) -> list:
        # Encode inputs directly (base64), no generic _tree_map walk.
        encoded_inputs = {}
        for key, (ptr, typestr, shape) in zip(input_keys, inputs, strict=True):
            view = _cupy_view(ptr, typestr, tuple(shape))
            encoded_inputs[key] = _encode_base64_input(view)

        body = orjson.dumps({"inputs": encoded_inputs})
        resp = session.post(apply_url, data=body, headers=headers)
        if not resp.ok:
            raise RuntimeError(
                f"Tesseract apply failed ({resp.status_code}): {resp.text}"
            )
        payload = orjson.loads(resp.content)

        # Decode only the known output leaves, in Tesseract order.
        decoded = _decode_leaf(payload, _load_cuda_ipc_arraydict)
        return [get(decoded) for get in out_leaf_getters]

    return run


def _decode_leaf(payload: dict, load_cuda_ipc):
    """Decode every cuda_ipc array leaf in the response to a cupy array.

    Kept minimal: walk the top-level output dict and decode array-shaped dicts.
    """
    out = {}
    for key, val in payload.items():
        out[key] = _decode_value(val, load_cuda_ipc)
    return out


def _decode_value(val: Any, load_cuda_ipc):
    if isinstance(val, dict) and "shape" in val and "data" in val:
        enc = val["data"].get("encoding")
        if enc == "cuda_ipc":
            return load_cuda_ipc(val)
        # base64 fallback (CPU arrays); decode to numpy then to cupy on demand.
        from tesseract_core.sdk.tesseract import _decode_array

        return _decode_array(val)
    if isinstance(val, dict):
        return {k: _decode_value(v, load_cuda_ipc) for k, v in val.items()}
    return val


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


import contextlib


@contextlib.contextmanager
def _cuda_ipc_output(tesseract_client):
    """Temporarily make the client request cuda_ipc outputs, then restore.

    Two things must change for the duration of a single ``apply``:

    * ``_output_format`` so the client's request encoder emits cuda_ipc for GPU
      inputs and its response decoder handles cuda_ipc arrays, and
    * an ``Accept: application/json+cuda_ipc`` header, since the response format
      is otherwise chosen by the *server's* default and the HTTP client never
      sends Accept on its own. Without this, a base64-default server would
      return base64 even though the client is in cuda_ipc mode.

    Both are scoped so the shared client is not permanently mutated (which would
    leak cuda_ipc behavior onto other users, e.g. the host-callback baseline).
    """
    client = getattr(tesseract_client, "_client", None)
    if client is None or not hasattr(client, "_output_format"):
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


def _http_endpoint(tesseract_client):
    """Return (base_url, requests.Session) for the client's HTTP transport."""
    client = getattr(tesseract_client, "_client", None)
    if client is None or not hasattr(client, "url"):
        raise RuntimeError(
            "GPU-direct dispatch requires an HTTP-served Tesseract "
            "(Tesseract.from_url / from_image().serve())."
        )
    return client.url, client._session


def _out_leaf_getters(out_tree, out_avals):
    """Build getters that pull each flat output leaf from the decoded dict.

    The Tesseract returns a dict of outputs. jax.tree.flatten on that dict uses
    sorted keys, so the flat output order is the sorted-key order of the
    (possibly nested) output structure. For milestone-1 scope we support a flat
    dict of array outputs; nested outputs would extend this walker.
    """
    # Reconstruct the flat leaf key order from the output pytree structure by
    # unflattening indices and re-flattening with a dict-aware walk.
    import jax

    indices = jax.tree.unflatten(out_tree, range(len(out_avals)))

    # indices is the output structure with integer leaves; produce, for each
    # leaf position, the sequence of dict keys to reach it.
    paths: list[tuple] = [()] * len(out_avals)

    def _walk(node, prefix):
        if isinstance(node, dict):
            for k in sorted(node.keys()):
                _walk(node[k], prefix + (k,))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(v, prefix + (i,))
        else:
            paths[node] = prefix

    _walk(indices, ())

    def make_getter(path):
        def get(decoded):
            cur = decoded
            for k in path:
                cur = cur[k]
            return cur

        return get

    return [make_getter(p) for p in paths]
