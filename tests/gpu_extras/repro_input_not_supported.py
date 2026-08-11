# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproducer: is cuda_ipc accepted as an array INPUT encoding?

History: tesseract-core initially accepted ``cuda_ipc`` as an *output* encoding
(flag-gated behind ``TESSERACT_ENABLE_EXPERIMENTAL_CUDA_IPC``) but rejected it as
an *input* encoding -- the per-field input model built by ``get_array_model``
omitted ``CudaIpcArrayData`` from its ``data`` union. That gap forced GPU-direct
dispatch to send inputs base64-over-host. It has since been fixed.

This script needs no GPU, CuPy, or server: it exercises the exact Pydantic
validation the server applies to request bodies. It turns the experimental flag
ON throughout, so it distinguishes "input rejected because the encoding is
structurally absent" from "input rejected because the flag is off".

Exit code 0  -> input ACCEPTED (the fix is in / claim no longer holds)
Exit code 1  -> input REJECTED (the input-direction gap is present)
"""

from __future__ import annotations

import base64
import sys

from pydantic import BaseModel, TypeAdapter, ValidationError

from tesseract_core.runtime.array_encoding import EncodedArrayModel, get_array_model


def _cuda_ipc_array_dict() -> dict:
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


def _data_union_encodings(model: type[BaseModel]) -> set[str]:
    import types
    import typing

    field = model.model_fields["data"]
    ann = field.annotation
    if typing.get_origin(ann) in (typing.Union, types.UnionType):
        members = typing.get_args(ann)
    else:
        members = (ann,)
    encodings: set[str] = set()
    for member in members:
        enc = getattr(member, "model_fields", {}).get("encoding")
        if enc is None:
            continue
        schema = TypeAdapter(enc.annotation).json_schema()
        if "const" in schema:
            encodings.add(schema["const"])
        elif "enum" in schema:
            encodings.update(schema["enum"])
    return encodings


def main() -> int:
    from tesseract_core.runtime.config import get_config, update_config

    update_config(enable_experimental_cuda_ipc=True)
    assert get_config().enable_experimental_cuda_ipc

    from tesseract_core.runtime.file_interactions import available_formats

    assert "json+cuda_ipc" in available_formats(), available_formats()
    print("[ok]  with the flag ON, 'json+cuda_ipc' is an available OUTPUT format")

    payload = _cuda_ipc_array_dict()
    EncodedArrayModel.model_validate(payload)
    print("[ok]  base EncodedArrayModel accepts a cuda_ipc array")

    input_model = get_array_model((3,), "float32", flags=[])
    base_enc = _data_union_encodings(EncodedArrayModel)
    input_enc = _data_union_encodings(input_model)
    print(f"  base model  data encodings : {sorted(base_enc)}")
    print(f"  input model data encodings : {sorted(input_enc)}")

    try:
        input_model.model_validate(payload)
    except ValidationError as e:
        print("[repro] per-field INPUT model REJECTS cuda_ipc (flag ON):")
        for err in e.errors():
            print(f"          {'.'.join(map(str, err['loc']))}: {err['type']}")
        print("\nINPUT-DIRECTION GAP PRESENT: cuda_ipc is a valid output encoding "
              "but not a valid input encoding; no flag enables it.")
        return 1

    print("[ok]  per-field INPUT model ACCEPTS cuda_ipc")
    print("\nINPUT DIRECTION SUPPORTED: cuda_ipc inputs validate; GPU-direct "
          "dispatch can send GPU inputs without a base64 host copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
