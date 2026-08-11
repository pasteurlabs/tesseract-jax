// Copyright 2025 Pasteur Labs. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal C++ glue for Strategy B (Rust native dispatch).
//
// This translation unit contains NO dispatch logic. Its only job is to use
// XLA's ffi.h -- which is designed for exactly this -- to decode the FFI call
// frame (inputs, outputs, the token attribute, the CUDA stream) into a flat C
// struct, then hand raw pointers to the Rust core (`tjgpu_b_dispatch`). Rust
// does 100% of the actual dispatch (HTTP, JSON, base64, CUDA IPC decode, the
// device->device copy) with no Python and no GIL.
//
// We keep this in C++ purely to avoid hand-transcribing the large XLA_FFI_Api
// function-pointer table (needed to fetch the stream) into Rust. The handler
// remains a plain XLA_FFI_Handler* exposed via a PyCapsule for
// jax.ffi.register_ffi_target.

#include <cstdint>
#include <vector>

// cudaStream_t is `struct CUstream_st*`; a void* alias is ABI-compatible and
// lets us avoid including the CUDA headers (Rust dlopens the runtime).
using cudaStream_t = void*;

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" {

// One buffer: device pointer, numpy-style dtype code, rank, and dims pointer.
struct TjgBuffer {
  void* data;
  int32_t dtype;  // XLA_FFI_DataType enum value
  int64_t rank;
  const int64_t* dims;
};

// Implemented in Rust. Returns 0 on success, non-zero on failure; on failure
// `err_buf` (capacity err_cap) is filled with a NUL-terminated message.
int tjgpu_b_dispatch(void* stream, int64_t token, const TjgBuffer* inputs,
                     size_t n_inputs, const TjgBuffer* outputs,
                     size_t n_outputs, char* err_buf, size_t err_cap);

}  // extern "C"

namespace {

TjgBuffer to_c(ffi::AnyBuffer buf) {
  TjgBuffer b;
  b.data = buf.untyped_data();
  b.dtype = static_cast<int32_t>(buf.element_type());
  auto dims = buf.dimensions();
  b.rank = static_cast<int64_t>(dims.size());
  b.dims = dims.begin();
  return b;
}

ffi::Error DispatchImpl(cudaStream_t stream, int64_t token,
                        ffi::RemainingArgs args, ffi::RemainingRets rets) {
  std::vector<TjgBuffer> in;
  in.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto a = args.get<ffi::AnyBuffer>(i);
    if (a.has_error()) return a.error();
    in.push_back(to_c(*a));
  }
  std::vector<TjgBuffer> out;
  out.reserve(rets.size());
  for (size_t i = 0; i < rets.size(); ++i) {
    auto r = rets.get<ffi::AnyBuffer>(i);
    if (r.has_error()) return r.error();
    out.push_back(to_c(**r));
  }

  char errbuf[512] = {0};
  int rc = tjgpu_b_dispatch(static_cast<void*>(stream), token, in.data(),
                            in.size(), out.data(), out.size(), errbuf,
                            sizeof(errbuf));
  if (rc != 0) {
    return ffi::Error::Internal(std::string("strategy B dispatch failed: ") +
                                errbuf);
  }
  return ffi::Error::Success();
}

}  // namespace

extern "C" {

// Build the handler and return it as a plain function pointer. Called from Rust
// (which wraps it in a PyCapsule for registration).
void* tjgpu_b_handler() {
  static auto* handler = ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()
                             .Attr<int64_t>("token")
                             .RemainingArgs()
                             .RemainingRets()
                             .To(DispatchImpl)
                             .release();
  static XLA_FFI_Handler* fn = +[](XLA_FFI_CallFrame* cf) -> XLA_FFI_Error* {
    return handler->Call(cf);
  };
  return reinterpret_cast<void*>(fn);
}

}  // extern "C"
