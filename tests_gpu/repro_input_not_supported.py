# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer: cuda_ipc is accepted as OUTPUT encoding but rejected as INPUT.

Claim under test (from the findings doc): a Tesseract can *return* arrays via the
``cuda_ipc`` encoding, but it *rejects* the same encoding on the way *in*, so
GPU-direct dispatch cannot yet send GPU inputs without a host (base64) copy.

Root cause: the base ``EncodedArrayModel`` includes ``CudaIpcArrayData`` in its
``data`` union, but the *per-field* model built by ``get_array_model`` (which is
what actually validates request inputs) rebuilds that union as only
``BinrefArrayData | Base64ArrayData | JsonArrayData`` -- omitting
``CudaIpcArrayData``. So a cuda_ipc-encoded input fails discriminated-union
validation.

This reproducer needs no GPU, no CuPy, and no running server: it exercises the
exact Pydantic validation the FastAPI server applies to request bodies. It uses
a syntactically valid cuda_ipc array dict (a zero handle); validation fails on
the *encoding*, well before any handle is ever opened, which is the whole point.

Run:
    python tests_gpu/repro_input_not_supported.py

Exit code 0 means the claim reproduced (input rejected, output accepted).
"""

from __future__ import annotations

import base64
import sys

from pydantic import BaseModel, TypeAdapter, ValidationError

from tesseract_core.runtime.array_encoding import (
    EncodedArrayModel,
    get_array_model,
)


def _cuda_ipc_array_dict() -> dict:
    """A syntactically valid cuda_ipc array payload for a (3,) float32 array."""
    handle = base64.b64encode(b"\x00" * 64).decode()  # 64-byte cudaIpcMemHandle_t
    return {
        "object_type": "array",
        "shape": [3],
        "dtype": "float32",
        "data": {
            "handle": handle,
            "device": 0,
            "storage_offset": 0,
            "storage_size": 12,
            "encoding": "cuda_ipc",
        },
    }


def _base64_array_dict() -> dict:
    """The same array, base64-encoded -- the currently-supported input path."""
    import numpy as np

    arr = np.arange(3, dtype=np.float32)
    return {
        "object_type": "array",
        "shape": [3],
        "dtype": "float32",
        "data": {
            "buffer": base64.b64encode(arr.tobytes()).decode(),
            "encoding": "base64",
        },
    }


def main() -> int:
    payload = _cuda_ipc_array_dict()

    # (1) The BASE model -- which advertises cuda_ipc support -- accepts it.
    EncodedArrayModel.model_validate(payload)
    print("[ok]  base EncodedArrayModel accepts a cuda_ipc array")

    # (2) The per-field INPUT model that the server actually uses to validate
    #     request bodies REJECTS it. This is the input-direction gap.
    #     get_array_model(expected_shape, expected_dtype, flags) builds the model
    #     the ``Array[(3,), Float32]`` annotation installs for request validation.
    InputArrayModel = get_array_model((3,), "float32", flags=[])

    try:
        InputArrayModel.model_validate(payload)
    except ValidationError as e:
        # Expected: the cuda_ipc branch isn't in the input model's data union.
        assert "cuda_ipc" in str(e) or "encoding" in str(e), str(e)
        print("[repro] per-field INPUT model REJECTS the cuda_ipc array:")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            print(f"          {loc}: {err['type']}")
    else:
        print("[FAIL] input model unexpectedly ACCEPTED cuda_ipc -- claim no longer holds")
        return 1

    # (3) Sanity: the same input model accepts the base64 encoding (the path the
    #     GPU-direct strategies are forced to use for inputs today).
    InputArrayModel.model_validate(_base64_array_dict())
    print("[ok]  per-field INPUT model accepts the base64 array (the fallback in use)")

    # (4) Show it is specifically the data union that differs.
    base_encodings = _data_union_encodings(EncodedArrayModel)
    input_encodings = _data_union_encodings(InputArrayModel)
    print(f"\n  base model  data encodings : {sorted(base_encodings)}")
    print(f"  input model data encodings : {sorted(input_encodings)}")
    missing = base_encodings - input_encodings
    print(f"  missing from input model   : {sorted(missing)}")
    assert "cuda_ipc" in missing, "expected cuda_ipc to be the omitted encoding"

    print("\nCLAIM REPRODUCED: cuda_ipc is a valid OUTPUT encoding but is not a "
          "valid INPUT encoding (omitted from the per-field input model's data union).")
    return 0


def _data_union_encodings(model: type[BaseModel]) -> set[str]:
    """Extract the set of ``encoding`` literals allowed by a model's ``data`` field."""
    field = model.model_fields["data"]
    encodings: set[str] = set()
    # The data field is a discriminated union of *ArrayData models, each with an
    # ``encoding: Literal[...]`` field. Walk the union members.
    for member in _union_members(field.annotation):
        enc_field = getattr(member, "model_fields", {}).get("encoding")
        if enc_field is None:
            continue
        # Literal[...] args are the allowed encoding strings.
        adapter = TypeAdapter(enc_field.annotation)
        schema = adapter.json_schema()
        if "const" in schema:
            encodings.add(schema["const"])
        elif "enum" in schema:
            encodings.update(schema["enum"])
    return encodings


def _union_members(annotation):
    import typing

    if typing.get_origin(annotation) in (typing.Union, __import__("types").UnionType):
        return typing.get_args(annotation)
    return (annotation,)


if __name__ == "__main__":
    sys.exit(main())
