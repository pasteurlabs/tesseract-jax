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
#include <cstdio>
#include <cstdlib>
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

// cudaMemoryType values (stable across CUDA 10-13). Device and Managed memory
// are dereferenceable on-device; Host/Unregistered means the pointer is (or may
// be) host memory -- the signature of an accidental host round-trip.
constexpr int cudaMemoryTypeUnregistered = 0;
constexpr int cudaMemoryTypeHost = 1;
constexpr int cudaMemoryTypeDevice = 2;
constexpr int cudaMemoryTypeManaged = 3;

// Mirror of `struct cudaPointerAttributes` for CUDA 11/12/13. Only the leading
// `type` field is read here; the rest is present to size the struct correctly
// for the ABI (cudaPointerGetAttributes writes all of it). Layout:
//   enum cudaMemoryType type;  int device;  void* devicePointer;
//   void* hostPointer;
struct CudaPointerAttributes {
  int type;
  int device;
  void* devicePointer;
  void* hostPointer;
};

struct CudaRuntime {
  void* handle = nullptr;
  cudaError_t (*Memcpy)(void*, const void*, size_t, int) = nullptr;
  cudaError_t (*MemcpyAsync)(void*, const void*, size_t, int, void* /*stream*/) =
      nullptr;
  cudaError_t (*MemsetAsync)(void*, int, size_t, void* /*stream*/) = nullptr;
  cudaError_t (*StreamSynchronize)(void* /*stream*/) = nullptr;
  const char* (*GetErrorString)(cudaError_t) = nullptr;
  // Optional: only used by the debug pointer-residency check. May be null if the
  // symbol is unavailable; the check degrades to a no-op in that case.
  cudaError_t (*PointerGetAttributes)(void* /*attrs*/, const void*) = nullptr;
  cudaError_t (*GetLastError)() = nullptr;
  // Driver API (from libcuda), used to save/restore the current CUDA context
  // around the dispatch. The Python decode path calls runtime APIs
  // (cudaSetDevice/cudaIpcOpenMemHandle) that bind the runtime *primary* context
  // onto this thread, displacing the separate driver context XLA runs kernels
  // in. We snapshot and restore XLA's context so later launches are unaffected.
  // Null if libcuda is unavailable; the save/restore then degrades to a no-op.
  int (*CtxGetCurrent)(void** /*CUcontext*/) = nullptr;
  int (*CtxSetCurrent)(void* /*CUcontext*/) = nullptr;
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
    rt.MemsetAsync = reinterpret_cast<decltype(rt.MemsetAsync)>(
        dlsym(rt.handle, "cudaMemsetAsync"));
    rt.StreamSynchronize = reinterpret_cast<decltype(rt.StreamSynchronize)>(
        dlsym(rt.handle, "cudaStreamSynchronize"));
    rt.GetErrorString = reinterpret_cast<decltype(rt.GetErrorString)>(
        dlsym(rt.handle, "cudaGetErrorString"));
    // Optional symbols for the debug residency check; tolerated if missing.
    rt.PointerGetAttributes =
        reinterpret_cast<decltype(rt.PointerGetAttributes)>(
            dlsym(rt.handle, "cudaPointerGetAttributes"));
    rt.GetLastError = reinterpret_cast<decltype(rt.GetLastError)>(
        dlsym(rt.handle, "cudaGetLastError"));
    if (!rt.Memcpy || !rt.MemcpyAsync || !rt.MemsetAsync ||
        !rt.StreamSynchronize) {
      throw std::runtime_error(
          "tesseract_jax: failed to resolve required cudaMemcpy/Memset symbols");
    }

    // Driver API for context save/restore. libcuda is the NVIDIA driver stub,
    // separate from the runtime (libcudart); it is normally already loaded in a
    // GPU-enabled process (XLA depends on it). Resolving the symbols from the
    // already-loaded image via RTLD_DEFAULT avoids guessing its soname; we fall
    // back to dlopen for robustness. If unavailable, the pointers stay null and
    // the save/restore becomes a no-op (see ContextGuard).
    rt.CtxGetCurrent = reinterpret_cast<decltype(rt.CtxGetCurrent)>(
        dlsym(RTLD_DEFAULT, "cuCtxGetCurrent"));
    rt.CtxSetCurrent = reinterpret_cast<decltype(rt.CtxSetCurrent)>(
        dlsym(RTLD_DEFAULT, "cuCtxSetCurrent"));
    if (!rt.CtxGetCurrent || !rt.CtxSetCurrent) {
      const char* driver_names[] = {"libcuda.so.1", "libcuda.so"};
      for (const char* n : driver_names) {
        if (void* h = dlopen(n, RTLD_NOW | RTLD_GLOBAL)) {
          rt.CtxGetCurrent = reinterpret_cast<decltype(rt.CtxGetCurrent)>(
              dlsym(h, "cuCtxGetCurrent"));
          rt.CtxSetCurrent = reinterpret_cast<decltype(rt.CtxSetCurrent)>(
              dlsym(h, "cuCtxSetCurrent"));
          break;
        }
      }
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

// Whether the debug pointer-residency check is enabled. Gated on the
// TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS env var, read once. When on, every
// buffer that crosses the FFI boundary (XLA inputs and the Python callback's
// results) is asserted to live in device (or managed) memory, so an accidental
// host round-trip -- e.g. a np.asarray/np.full slipping onto a dispatch return
// path -- fails loudly at the boundary instead of silently copying through host.
bool debug_check_device_ptrs() {
  static const bool on = [] {
    const char* v = std::getenv("TESSERACT_JAX_DEBUG_CHECK_DEVICE_PTRS");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
  }();
  return on;
}

const char* memory_type_name(int t) {
  switch (t) {
    case cudaMemoryTypeUnregistered: return "unregistered (host)";
    case cudaMemoryTypeHost:         return "host";
    case cudaMemoryTypeDevice:       return "device";
    case cudaMemoryTypeManaged:      return "managed";
    default:                         return "unknown";
  }
}

// Assert `ptr` is device-resident. Returns an ffi::Error (so the caller can
// propagate it) when the pointer is host/unregistered memory; success otherwise.
// `what` labels the buffer in the message (e.g. "input 0", "result 2").
ffi::Error assert_device_ptr(const void* ptr, const std::string& what) {
  auto& rt = cuda_rt();
  if (!rt.PointerGetAttributes) {
    // Symbol unavailable: cannot check, so do not block. (Should be rare.)
    return ffi::Error::Success();
  }
  CudaPointerAttributes attrs{};
  cudaError_t e = rt.PointerGetAttributes(&attrs, ptr);
  if (e != cudaSuccess) {
    // A plain host pointer makes older drivers return cudaErrorInvalidValue.
    // Clear the sticky error so we don't poison a later real check, then treat
    // it as a residency failure -- an unrecognized pointer is not device memory.
    if (rt.GetLastError) rt.GetLastError();
    return ffi::Error::Internal(
        "tesseract_jax [debug]: " + what +
        " pointer is not device-resident (cudaPointerGetAttributes failed: " +
        cuda_err(e) + "). An accidental host copy likely reached the FFI "
        "boundary.");
  }
  if (attrs.type != cudaMemoryTypeDevice &&
      attrs.type != cudaMemoryTypeManaged) {
    return ffi::Error::Internal(
        "tesseract_jax [debug]: " + what + " pointer is in " +
        std::string(memory_type_name(attrs.type)) +
        " memory, expected device. An accidental host copy reached the FFI "
        "boundary (e.g. a np.asarray/np.full on a GPU dispatch return path).");
  }
  return ffi::Error::Success();
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

  // Snapshot the current CUDA context and restore it when this handler returns.
  // The Python dispatch decodes cuda_ipc results via runtime APIs
  // (cudaSetDevice/cudaMalloc/cudaIpcOpenMemHandle) that bind the runtime
  // *primary* context onto this thread. XLA runs its kernels in a *different*
  // (driver) context, so without this the primary context stays current and the
  // next kernel launch fails with cudaErrorInvalidValue before cuModuleGetFunction.
  // The guard restores XLA's context on *every* exit path (like KeepaliveGuard).
  // A no-op if the driver symbols are unavailable or nothing was current.
  struct ContextGuard {
    CudaRuntime& rt;
    void* saved = nullptr;
    bool active = false;
    ContextGuard(CudaRuntime& r) : rt(r) {
      if (rt.CtxGetCurrent && rt.CtxSetCurrent &&
          rt.CtxGetCurrent(&saved) == 0 && saved != nullptr) {
        active = true;
      }
    }
    ~ContextGuard() {
      if (active) rt.CtxSetCurrent(saved);
    }
  } context_guard{rt};

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

  // Debug: XLA inputs must be device memory. They always should be (they are
  // XLA device buffers); checking them validates the check itself and would
  // catch a platform/lowering mishap.
  if (debug_check_device_ptrs()) {
    for (size_t i = 0; i < in_descs.size(); ++i) {
      if (auto e = assert_device_ptr(reinterpret_cast<void*>(in_descs[i].ptr),
                                     "input " + std::to_string(i));
          e.failure()) {
        return e;
      }
    }
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
  // The keepalive vector holds Python objects, so it must be emptied while the
  // GIL is held -- otherwise the py::object destructors call dec_ref() with no
  // GIL and abort the process. This guard clears it under the GIL on *every*
  // exit path (including early error returns), so a residency-check failure
  // surfaces as a clean ffi::Error instead of a crash.
  struct KeepaliveGuard {
    std::vector<py::object>& v;
    ~KeepaliveGuard() {
      if (v.empty()) return;
      py::gil_scoped_acquire gil;
      v.clear();
    }
  } keepalive_guard{results_keepalive};
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
    // TEMP DIAGNOSTIC: log the driver context around the decode to confirm
    // whether/what the cuda_ipc decode changes. Gated on TESSERACT_JAX_DEBUG_CTX.
    const bool dbg_ctx = [] {
      const char* v = std::getenv("TESSERACT_JAX_DEBUG_CTX");
      return v != nullptr && v[0] != '\0' && std::string(v) != "0";
    }();
    void* ctx_before = nullptr;
    if (dbg_ctx && rt.CtxGetCurrent) {
      int rc = rt.CtxGetCurrent(&ctx_before);
      std::fprintf(stderr,
                   "[tj-ctx] before decode: rc=%d ctx=%p CtxGet=%p CtxSet=%p\n",
                   rc, ctx_before, (void*)rt.CtxGetCurrent, (void*)rt.CtxSetCurrent);
      std::fflush(stderr);
    }
    try {
      out = cb(token, py_inputs);
    } catch (py::error_already_set& e) {
      return ffi::Error::Internal(std::string("dispatch callback raised: ") +
                                  e.what());
    }
    if (dbg_ctx && rt.CtxGetCurrent) {
      void* ctx_after = nullptr;
      int rc = rt.CtxGetCurrent(&ctx_after);
      // cudaGetLastError both reads AND clears the sticky per-thread error. If
      // the decode left a sticky error, this reveals it (and clearing it is
      // itself a candidate fix to test).
      int last_err = rt.GetLastError ? rt.GetLastError() : -999;
      std::fprintf(stderr,
                   "[tj-ctx] after decode:  rc=%d ctx=%p (changed=%d) "
                   "cudaGetLastError=%d (%s)\n",
                   rc, ctx_after, ctx_after != ctx_before, last_err,
                   cuda_err(last_err).c_str());
      std::fflush(stderr);
    }

    // Expect a list of objects exposing __cuda_array_interface__.
    py::sequence seq = py::reinterpret_borrow<py::sequence>(out);
    if (py::len(seq) != out_descs.size()) {
      return ffi::Error::Internal(
          "dispatch callback returned wrong number of results");
    }
    const bool check_ptrs = debug_check_device_ptrs();
    for (size_t i = 0; i < out_descs.size(); ++i) {
      py::object item = seq[i];
      results_keepalive.push_back(item);  // keep alive through the copy
      BufferDesc rd;
      // A ``None`` result marks a placeholder slot: a discarded gradient/tangent
      // for a non-differentiable input or output that no consumer reads. There is
      // no source array to copy; rd.ptr == 0 flags it for a NaN-fill (rather than
      // a copy) of XLA's output buffer below -- no source buffer is fabricated.
      if (item.is_none()) {
        rd.ptr = 0;
        rd.nbytes = 0;
        result_descs.push_back(rd);
        continue;
      }
      py::object cai = item.attr("__cuda_array_interface__");
      py::tuple data = cai["data"].cast<py::tuple>();
      rd.ptr = data[0].cast<uintptr_t>();
      rd.nbytes = out_descs[i].nbytes;  // trust XLA's expected size
      result_descs.push_back(rd);
      // Debug: the dispatch's returned buffers must be device-resident.
      // This is the check that matters: a host copy on a derivative return path
      // (np.asarray/np.full materializing a host array) surfaces here as a host
      // pointer, and we fail instead of silently copying host->"device".
      if (check_ptrs) {
        if (auto e = assert_device_ptr(reinterpret_cast<void*>(rd.ptr),
                                       "result " + std::to_string(i));
            e.failure()) {
          return e;
        }
      }
    }
  }  // release GIL before the device copies

  // Fill each XLA-owned output buffer. A null pointer marks a placeholder slot
  // (the dispatch returned ``None``): its value is a discarded gradient/tangent
  // that no consumer reads, so instead of copying we fill it with the byte
  // pattern 0xff -- which is a (quiet) NaN for every IEEE float width. This keeps
  // the device path's poison semantics identical to the host path (np.full(nan))
  // and leaves no output buffer undefined, all without allocating a source
  // buffer. Every other slot is a real device result we copy device->device.
  for (size_t i = 0; i < out_descs.size(); ++i) {
    if (result_descs[i].ptr == 0) {
      if (cudaError_t e = rt.MemsetAsync(reinterpret_cast<void*>(out_descs[i].ptr),
                                         0xff, out_descs[i].nbytes, stream);
          e != cudaSuccess) {
        return ffi::Error::Internal("cudaMemsetAsync(placeholder) failed: " +
                                    cuda_err(e));
      }
      continue;
    }
    if (cudaError_t e = rt.MemcpyAsync(
            reinterpret_cast<void*>(out_descs[i].ptr),
            reinterpret_cast<void*>(result_descs[i].ptr), out_descs[i].nbytes,
            cudaMemcpyDeviceToDevice, stream);
        e != cudaSuccess) {
      return ffi::Error::Internal("cudaMemcpyAsync(result) failed: " +
                                  cuda_err(e));
    }
  }
  // Ensure the copies complete before the Python result buffers (held by
  // results_keepalive) can be freed/recycled. The keepalive is then cleared
  // under the GIL by keepalive_guard as this function returns.
  if (cudaError_t e = rt.StreamSynchronize(stream); e != cudaSuccess) {
    return ffi::Error::Internal("cudaStreamSynchronize(post) failed: " +
                                cuda_err(e));
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
