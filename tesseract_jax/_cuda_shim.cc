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
//   reach Python we acquire the GIL (nanobind gil_scoped_acquire) and call a
//   registered callable.
// * The Python bindings use nanobind, built against the CPython stable ABI
//   (Py_LIMITED_API): the module's entire Python surface is three entry points
//   marshalling ints/strings/lists/tuples/objects, well inside the Limited API,
//   so one abi3 wheel per platform serves every supported CPython version.
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
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
  // Reads and clears the thread's sticky last-error slot. Used after the Python
  // decode to drop the non-fatal error it leaves behind (see DispatchImpl).
  cudaError_t (*GetLastError)() = nullptr;
};

// libcudart names/paths to dlopen, most-preferred first. Set from Python (see
// set_cudart_candidates) before the first dispatch, so the shim resolves the
// *same* runtime tesseract-core's cuda_ipc codec does (wheel dirs first). The
// list is the single source of truth for discovery: there is deliberately no
// hardcoded soname fallback here, because the C++ handler is only ever reached
// through gpu_ffi.ensure_registered(), which primes this list on the same line
// it registers the FFI target. A guessed fallback would risk loading a
// *different* libcudart than the codec -- the exact mismatch this indirection
// exists to prevent -- so an unprimed list is a hard error instead.
std::vector<std::string>& cudart_candidates() {
  // Leaked on purpose (see dispatch_callable): a function-local static would run
  // its destructor during C++ static teardown, after the interpreter is gone.
  static auto* names = new std::vector<std::string>();
  return *names;
}

CudaRuntime& cuda_rt() {
  static CudaRuntime rt;
  static std::once_flag once;
  std::call_once(once, [] {
    const auto& primed = cudart_candidates();
    if (primed.empty()) {
      throw std::runtime_error(
          "tesseract_jax: libcudart candidates not set; the FFI shim must be "
          "primed via gpu_ffi.ensure_registered() before dispatch");
    }
    for (const std::string& n : primed) {
      rt.handle = dlopen(n.c_str(), RTLD_NOW | RTLD_GLOBAL);
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

nb::object& dispatch_callable() {
  // Heap-allocated and intentionally never freed. A function-local
  // ``static nb::object`` would run ~object() during C++ static destruction at
  // process exit -- which happens *after* the Python interpreter is finalized --
  // so the Py_DECREF it performs dereferences a dead interpreter and segfaults
  // (observed as an exit-139 teardown crash in gdb: ~object() from this module).
  // Leaking the reference is the standard fix: the process is exiting, so the
  // holdout costs nothing and no destructor touches Python after finalization.
  static nb::object* cb = new nb::object();  // set via set_dispatch_callback
  return *cb;
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

// Mirrors `np.issubdtype(arr.dtype, np.inexact)`: the float and complex dtypes,
// i.e. the ones with a NaN bit pattern to spell. Used to fill a discarded
// derivative slot the same way the host path does -- NaN for inexact dtypes,
// zero for everything else (see the placeholder fill in DispatchImpl).
bool dtype_is_inexact(ffi::DataType dt) {
  using DT = ffi::DataType;
  switch (dt) {
    case DT::F16:
    case DT::F32:
    case DT::F64:
    case DT::BF16:
    case DT::C64:
    case DT::C128:
      return true;
    default:
      return false;
  }
}

// Render a shape as "(d0, d1, ...)" for mismatch diagnostics.
std::string shape_str(const std::vector<int64_t>& shape) {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  s += ")";
  return s;
}

// A plain description of one buffer, handed to Python.
struct BufferDesc {
  uintptr_t ptr;
  std::string typestr;
  std::vector<int64_t> shape;
  size_t nbytes;
  bool inexact = false;  // dtype has a NaN to spell (F16/F32/F64/BF16/C64/C128)
};

BufferDesc describe(ffi::AnyBuffer buf) {
  BufferDesc d;
  d.ptr = reinterpret_cast<uintptr_t>(buf.untyped_data());
  d.typestr = dtype_typestr(buf.element_type());
  d.inexact = dtype_is_inexact(buf.element_type());
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
  std::vector<nb::object> results_keepalive;
  std::vector<BufferDesc> result_descs;
  // The keepalive vector holds Python objects, so it must be emptied while the
  // GIL is held -- otherwise the nb::object destructors call dec_ref() with no
  // GIL and abort the process. This guard clears it under the GIL on *every*
  // exit path (including early error returns), so a residency-check failure
  // surfaces as a clean ffi::Error instead of a crash.
  struct KeepaliveGuard {
    std::vector<nb::object>& v;
    ~KeepaliveGuard() {
      if (v.empty()) return;
      nb::gil_scoped_acquire gil;
      v.clear();
    }
  } keepalive_guard{results_keepalive};
  {
    nb::gil_scoped_acquire gil;
    nb::object& cb = dispatch_callable();
    if (cb.is_none()) {
      return ffi::Error::Internal(
          "tesseract_jax: no dispatch callback registered");
    }

    // Build the list of input views: (ptr, typestr, shape) tuples.
    nb::list py_inputs;
    for (const auto& d : in_descs) {
      py_inputs.append(
          nb::make_tuple(d.ptr, nb::str(d.typestr.c_str()), nb::cast(d.shape)));
    }

    nb::object out;
    try {
      out = cb(token, py_inputs);
    } catch (nb::python_error& e) {
      return ffi::Error::Internal(std::string("dispatch callback raised: ") +
                                  e.what());
    }

    // Clear any sticky CUDA runtime error left by the Python decode. The
    // cuda_ipc decode drives the runtime API by ctypes (cudaSetDevice /
    // cudaMalloc / cudaIpcOpenMemHandle / cudaIpcCloseMemHandle); on this path
    // it leaves a non-sticky cudaErrorInvalidValue (code 1) in the thread's
    // last-error slot without failing the decode. cudaGetLastError both reads
    // and *clears* that slot. If we don't clear it here, JAX's next kernel
    // launch calls cudaGetLastError first, sees the stale error, and aborts
    // with "error before calling cuModuleGetFunction (1): cudaErrorInvalidValue"
    // -- so the first dispatch works but every subsequent JAX op fails. The
    // context is unaffected (verified: it is unchanged across the decode); only
    // the last-error slot needs resetting.
    if (rt.GetLastError) {
      rt.GetLastError();
    }

    // Expect a list of objects exposing __cuda_array_interface__. Reading their
    // pointer/dtype/shape drives arbitrary Python (attribute access, casts,
    // __getitem__), any of which may raise -- and a nanobind exception must not
    // unwind across the C-ABI FFI boundary into XLA. Convert any throw into an
    // ffi::Error so a malformed dispatch return fails cleanly instead of
    // crashing the process.
    try {
      nb::sequence seq = nb::borrow<nb::sequence>(out);
      if (nb::len(seq) != out_descs.size()) {
        return ffi::Error::Internal(
            "dispatch callback returned wrong number of results");
      }
      const bool check_ptrs = debug_check_device_ptrs();
      for (size_t i = 0; i < out_descs.size(); ++i) {
        nb::object item = seq[i];
        results_keepalive.push_back(item);  // keep alive through the copy
        BufferDesc rd;
        // A ``None`` result marks a placeholder slot: a discarded
        // gradient/tangent for a non-differentiable input or output that no
        // consumer reads. There is no source array to copy; rd.ptr == 0 flags it
        // for a fill (rather than a copy) of XLA's output buffer below -- no
        // source buffer is fabricated.
        if (item.is_none()) {
          rd.ptr = 0;
          rd.nbytes = 0;
          result_descs.push_back(rd);
          continue;
        }
        nb::object cai = item.attr("__cuda_array_interface__");
        nb::tuple data = nb::cast<nb::tuple>(cai["data"]);
        rd.ptr = nb::cast<uintptr_t>(data[0]);
        // XLA sized this output buffer from tesseract-jax's declared avals, but
        // nothing forces the Tesseract to return that dtype or shape: the
        // jacobian response schema permits any dtype, and unconstrained output
        // shapes are only validated by the server when the schema fully pins
        // them. Unlike the host path we cannot cast here, so compare before
        // copying -- otherwise a same-itemsize dtype swap (int32 vs float32)
        // would be silently reinterpreted, and a narrower source would drive an
        // out-of-bounds device read.
        rd.typestr = nb::cast<std::string>(cai["typestr"]);
        rd.shape = nb::cast<std::vector<int64_t>>(cai["shape"]);
        if (rd.typestr != out_descs[i].typestr || rd.shape != out_descs[i].shape) {
          return ffi::Error::InvalidArgument(
              "tesseract_jax result " + std::to_string(i) +
              ": Tesseract returned " + rd.typestr + shape_str(rd.shape) +
              ", expected " + out_descs[i].typestr +
              shape_str(out_descs[i].shape));
        }
        rd.nbytes = out_descs[i].nbytes;  // now known to agree
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
    } catch (nb::python_error& e) {
      return ffi::Error::Internal(
          std::string("tesseract_jax: decoding dispatch results failed: ") +
          e.what());
    } catch (const std::exception& e) {
      return ffi::Error::Internal(
          std::string("tesseract_jax: decoding dispatch results failed: ") +
          e.what());
    }
  }  // release GIL before the device copies

  // Fill each XLA-owned output buffer. A null pointer marks a placeholder slot
  // (the dispatch returned ``None``): its value is a discarded gradient/tangent
  // that no consumer reads, so instead of copying we fill it with each dtype's
  // 0/0 pattern, matching the host path. 0xff is a quiet NaN at every IEEE float
  // width (and, repeated, a NaN in each half of a complex value); for the exact
  // dtypes -- ints, unsigned, bool -- 0xff would instead be a plausible in-range
  // value (or, for bool/PRED, the out-of-range byte 0xff), so those are
  // zero-filled to stay consistent with the host's 0/0 result. This leaves no
  // output buffer undefined without allocating a source buffer. Every other slot
  // is a real device result we copy device->device.
  for (size_t i = 0; i < out_descs.size(); ++i) {
    if (result_descs[i].ptr == 0) {
      const int pattern = out_descs[i].inexact ? 0xff : 0x00;
      if (cudaError_t e = rt.MemsetAsync(reinterpret_cast<void*>(out_descs[i].ptr),
                                         pattern, out_descs[i].nbytes, stream);
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

NB_MODULE(_cuda_shim, m) {
  m.doc() = "Native FFI shim for GPU-direct Tesseract dispatch";

  m.def("set_dispatch_callback", [](nb::object cb) {
    dispatch_callable() = std::move(cb);
  });

  // Prime the libcudart search list (see cudart_candidates). Must be called
  // before the first dispatch: cuda_rt() reads it once, under std::call_once, on
  // the first dlopen and ignores later changes. gpu_ffi.ensure_registered()
  // calls this ahead of registering the FFI target.
  m.def("set_cudart_candidates", [](std::vector<std::string> names) {
    cudart_candidates() = std::move(names);
  });

  // Expose the handler as a PyCapsule for jax.ffi.register_ffi_target.
  m.def("handler_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(MakeDispatchHandler()),
                       "xla._CUSTOM_CALL_TARGET");
  });
}
