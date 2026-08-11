// Copyright 2025 Pasteur Labs. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native FFI shim for GPU-direct Tesseract dispatch.
//
// This module registers a single XLA custom-call (typed FFI) handler that
// bridges XLA-owned device buffers into a Python dispatch callback and copies
// the callback's device-resident results back into XLA-owned output buffers.
//
// Design notes
// ------------
// * The CUDA runtime is loaded at *runtime* via dlopen (see cuda_rt()), so this
//   extension is not linked against any specific CUDA version. It resolves the
//   handful of symbols it needs by name and works against CUDA 11/12/13, whose
//   runtime ABI for these functions is stable. This mirrors how tesseract-core
//   accesses CUDA from Python (ctypes.CDLL).
// * The handler is a plain C-ABI function pointer wrapped in a PyCapsule; XLA
//   calls it directly on its executor thread, with no Python on the stack. To
//   reach Python we acquire the GIL (pybind11 gil_scoped_acquire) and call a
//   registered callable.
// * The registered Python callback returns the result arrays (as objects
//   exposing __cuda_array_interface__) and the shim copies them device->device
//   into XLA's output buffers.

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Minimal CUDA runtime access via dlopen (no compile-time CUDA dependency)
// ---------------------------------------------------------------------------

namespace {

using cudaError_t = int;
// cudaStream_t is `struct CUstream_st*`; a void* alias is ABI-compatible and
// lets us avoid including the CUDA headers (we dlopen the runtime instead).
using cudaStream_t = void*;
constexpr int cudaSuccess = 0;
constexpr int cudaMemcpyDeviceToDevice = 3;

struct CudaRuntime {
  void* handle = nullptr;
  cudaError_t (*Memcpy)(void*, const void*, size_t, int) = nullptr;
  cudaError_t (*MemcpyAsync)(void*, const void*, size_t, int, void* /*stream*/) =
      nullptr;
  cudaError_t (*StreamSynchronize)(void* /*stream*/) = nullptr;
  const char* (*GetErrorString)(cudaError_t) = nullptr;
};

CudaRuntime& cuda_rt() {
  static CudaRuntime rt;
  static std::once_flag once;
  std::call_once(once, [] {
    const char* names[] = {"libcudart.so",    "libcudart.so.13",
                           "libcudart.so.12", "libcudart.so.11"};
    for (const char* n : names) {
      rt.handle = dlopen(n, RTLD_NOW | RTLD_GLOBAL);
      if (rt.handle) break;
    }
    if (!rt.handle) {
      throw std::runtime_error(
          "tesseract_jax: could not dlopen libcudart (is CUDA installed?)");
    }
    rt.Memcpy = reinterpret_cast<decltype(rt.Memcpy)>(
        dlsym(rt.handle, "cudaMemcpy"));
    rt.MemcpyAsync = reinterpret_cast<decltype(rt.MemcpyAsync)>(
        dlsym(rt.handle, "cudaMemcpyAsync"));
    rt.StreamSynchronize = reinterpret_cast<decltype(rt.StreamSynchronize)>(
        dlsym(rt.handle, "cudaStreamSynchronize"));
    rt.GetErrorString = reinterpret_cast<decltype(rt.GetErrorString)>(
        dlsym(rt.handle, "cudaGetErrorString"));
    if (!rt.Memcpy || !rt.MemcpyAsync || !rt.StreamSynchronize) {
      throw std::runtime_error(
          "tesseract_jax: failed to resolve required cudaMemcpy symbols");
    }
  });
  return rt;
}

std::string cuda_err(cudaError_t e) {
  auto& rt = cuda_rt();
  if (rt.GetErrorString) {
    const char* s = rt.GetErrorString(e);
    if (s) return std::string(s);
  }
  return "cuda error " + std::to_string(e);
}

// ---------------------------------------------------------------------------
// Python dispatch registry
// ---------------------------------------------------------------------------
//
// The lowering passes an integer `token` as an FFI attribute. That token keys a
// Python-side descriptor (the Jaxeract client, eval_func, pytree metadata, ...)
// held in the Python module. We call a single registered dispatch callable with
// (token, input_views) and receive back a list of result arrays.

py::object& dispatch_callable() {
  static py::object cb;  // set from Python via set_dispatch_callback
  return cb;
}

// Map an XLA FFI dtype to a numpy typestr (little-endian) so the Python side can
// build __cuda_array_interface__ views without guessing. Element size comes from
// AnyBuffer::size_bytes(); we only need the typestr here.
const char* dtype_typestr(ffi::DataType dt) {
  using DT = ffi::DataType;
  switch (dt) {
    case DT::PRED: return "|b1";
    case DT::S8:   return "|i1";
    case DT::U8:   return "|u1";
    case DT::S16:  return "<i2";
    case DT::U16:  return "<u2";
    case DT::S32:  return "<i4";
    case DT::U32:  return "<u4";
    case DT::S64:  return "<i8";
    case DT::U64:  return "<u8";
    case DT::F16:  return "<f2";
    case DT::F32:  return "<f4";
    case DT::F64:  return "<f8";
    case DT::C64:  return "<c8";
    case DT::C128: return "<c16";
    case DT::BF16: return "<V2";  // no numpy bf16; opaque 2 bytes
    default: return "";
  }
}

// A plain description of one buffer, handed to Python.
struct BufferDesc {
  uintptr_t ptr;
  std::string typestr;
  std::vector<int64_t> shape;
  size_t nbytes;
};

BufferDesc describe(ffi::AnyBuffer buf) {
  BufferDesc d;
  d.ptr = reinterpret_cast<uintptr_t>(buf.untyped_data());
  d.typestr = dtype_typestr(buf.element_type());
  auto dims = buf.dimensions();
  for (size_t i = 0; i < dims.size(); ++i) {
    d.shape.push_back(dims[i]);
  }
  d.nbytes = buf.size_bytes();
  return d;
}

// ---------------------------------------------------------------------------
// The FFI handler
// ---------------------------------------------------------------------------

ffi::Error DispatchImpl(cudaStream_t stream, int64_t token,
                        ffi::RemainingArgs args, ffi::RemainingRets rets) {
  auto& rt = cuda_rt();

  // Gather input buffer descriptors (device pointers stay on device).
  // args.get<T>() returns ErrorOr<T>; rets.get<T>() returns ErrorOr<Result<T>>.
  std::vector<BufferDesc> in_descs;
  in_descs.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    if (arg.has_error()) return arg.error();
    in_descs.push_back(describe(*arg));
  }

  // Gather output buffer descriptors so we can copy into them after dispatch.
  std::vector<BufferDesc> out_descs;
  out_descs.reserve(rets.size());
  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<ffi::AnyBuffer>(i);
    if (ret.has_error()) return ret.error();
    out_descs.push_back(describe(**ret));
  }

  // XLA's input buffers are only valid once prior stream work completes. For a
  // correct-first implementation we synchronize the stream so the Python side
  // (which operates on CUDA's default/per-thread stream via CuPy/ctypes) sees
  // ready inputs. This is the conservative ordering contract from the spec;
  // event-based ordering is a later optimization.
  if (cudaError_t e = rt.StreamSynchronize(stream); e != cudaSuccess) {
    return ffi::Error::Internal("cudaStreamSynchronize(pre) failed: " +
                                cuda_err(e));
  }

  // Call into Python under the GIL.
  std::vector<py::object> results_keepalive;
  std::vector<BufferDesc> result_descs;
  {
    py::gil_scoped_acquire gil;
    py::object& cb = dispatch_callable();
    if (cb.is_none()) {
      return ffi::Error::Internal(
          "tesseract_jax: no dispatch callback registered");
    }

    // Build the list of input views: (ptr, typestr, shape) tuples.
    py::list py_inputs;
    for (const auto& d : in_descs) {
      py_inputs.append(
          py::make_tuple(d.ptr, py::str(d.typestr), py::cast(d.shape)));
    }

    py::object out;
    try {
      out = cb(token, py_inputs);
    } catch (py::error_already_set& e) {
      return ffi::Error::Internal(std::string("dispatch callback raised: ") +
                                  e.what());
    }

    // Expect a list of objects exposing __cuda_array_interface__.
    py::sequence seq = py::reinterpret_borrow<py::sequence>(out);
    if (py::len(seq) != out_descs.size()) {
      return ffi::Error::Internal(
          "dispatch callback returned wrong number of results");
    }
    for (size_t i = 0; i < out_descs.size(); ++i) {
      py::object item = seq[i];
      results_keepalive.push_back(item);  // keep alive through the copy
      py::object cai = item.attr("__cuda_array_interface__");
      py::tuple data = cai["data"].cast<py::tuple>();
      BufferDesc rd;
      rd.ptr = data[0].cast<uintptr_t>();
      rd.nbytes = out_descs[i].nbytes;  // trust XLA's expected size
      result_descs.push_back(rd);
    }
  }  // release GIL before the device copies

  // Copy each result device->device into the XLA-owned output buffer.
  for (size_t i = 0; i < out_descs.size(); ++i) {
    if (cudaError_t e = rt.MemcpyAsync(
            reinterpret_cast<void*>(out_descs[i].ptr),
            reinterpret_cast<void*>(result_descs[i].ptr), out_descs[i].nbytes,
            cudaMemcpyDeviceToDevice, stream);
        e != cudaSuccess) {
      return ffi::Error::Internal("cudaMemcpyAsync(result) failed: " +
                                  cuda_err(e));
    }
  }
  // Ensure the copies complete before the Python result buffers (held only by
  // results_keepalive, about to be dropped) can be freed/recycled.
  if (cudaError_t e = rt.StreamSynchronize(stream); e != cudaSuccess) {
    return ffi::Error::Internal("cudaStreamSynchronize(post) failed: " +
                                cuda_err(e));
  }
  {
    py::gil_scoped_acquire gil;
    results_keepalive.clear();
  }
  return ffi::Error::Success();
}

// Build the binding as a named value so the commas inside the template
// arguments don't confuse the variadic XLA_FFI_DEFINE_HANDLER macro's
// argument counting. Then hand the ready-made XLA_FFI_Handler* to
// register_ffi_target via a PyCapsule.
XLA_FFI_Handler* MakeDispatchHandler() {
  static auto* handler = ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()
                             .Attr<int64_t>("token")
                             .RemainingArgs()
                             .RemainingRets()
                             .To(DispatchImpl)
                             .release();
  return +[](XLA_FFI_CallFrame* call_frame) -> XLA_FFI_Error* {
    return handler->Call(call_frame);
  };
}

}  // namespace

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_cuda_shim, m) {
  m.doc() = "Native FFI shim for GPU-direct Tesseract dispatch";

  m.def("set_dispatch_callback", [](py::object cb) {
    dispatch_callable() = std::move(cb);
  });

  // Expose the handler as a PyCapsule for jax.ffi.register_ffi_target.
  m.def("handler_capsule", []() {
    return py::capsule(reinterpret_cast<void*>(MakeDispatchHandler()),
                       "xla._CUSTOM_CALL_TARGET");
  });
}
