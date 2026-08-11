// Copyright 2025 Pasteur Labs. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
//! Strategy B: native, GIL-free GPU-direct Tesseract dispatch.
//!
//! The XLA FFI handler (built in glue.cc) calls [`tjgpu_b_dispatch`] with raw
//! device pointers and the CUDA stream. This module does the *entire* dispatch
//! natively -- base64-encode inputs, HTTP POST to the Tesseract, parse the JSON
//! response, and for each cuda_ipc output open the IPC handle and copy the data
//! device->device into the XLA-owned output buffer -- with no Python and no GIL
//! on the hot path. Python is used only once, up front, to register a
//! descriptor (URL + input/output metadata) keyed by an integer token.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use base64::Engine;
use pyo3::prelude::*;

mod cuda;

// ---------------------------------------------------------------------------
// Descriptor registry (token -> dispatch descriptor), set from Python
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Descriptor {
    apply_url: String,
    /// Tesseract input names in the flat (sorted) order XLA passes buffers.
    input_keys: Vec<String>,
    /// Tesseract output leaf names in the flat order XLA expects buffers.
    output_keys: Vec<String>,
}

fn registry() -> &'static Mutex<HashMap<i64, Descriptor>> {
    static R: OnceLock<Mutex<HashMap<i64, Descriptor>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// The C ABI the C++ glue calls into
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct TjgBuffer {
    data: *mut c_void,
    dtype: i32, // XLA_FFI_DataType enum value
    rank: i64,
    dims: *const i64,
}

impl TjgBuffer {
    fn shape(&self) -> Vec<i64> {
        if self.dims.is_null() || self.rank == 0 {
            return Vec::new();
        }
        unsafe { std::slice::from_raw_parts(self.dims, self.rank as usize).to_vec() }
    }
}

/// numpy dtype name for an XLA FFI DataType enum value (subset we support).
fn dtype_name(dt: i32) -> Option<&'static str> {
    // Values mirror XLA_FFI_DataType in c_api.h.
    Some(match dt {
        1 => "bool",     // PRED
        2 => "int8",     // S8
        3 => "int16",    // S16
        4 => "int32",    // S32
        5 => "int64",    // S64
        6 => "uint8",    // U8
        7 => "uint16",   // U16
        8 => "uint32",   // U32
        9 => "uint64",   // U64
        10 => "float16", // F16
        11 => "float32", // F32
        12 => "float64", // F64
        15 => "complex64",
        16 => "complex128",
        _ => return None,
    })
}

fn dtype_itemsize(name: &str) -> usize {
    match name {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" | "float16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" | "complex64" => 8,
        "complex128" => 16,
        _ => 0,
    }
}

fn nbytes(shape: &[i64], itemsize: usize) -> usize {
    let n: i64 = shape.iter().product::<i64>().max(if shape.is_empty() { 1 } else { 0 });
    n as usize * itemsize
}

/// Entry point invoked by the C++ FFI glue. Returns 0 on success.
///
/// # Safety
/// All pointers come from XLA's call frame and are valid for the call.
#[no_mangle]
pub unsafe extern "C" fn tjgpu_b_dispatch(
    stream: *mut c_void,
    token: i64,
    inputs: *const TjgBuffer,
    n_inputs: usize,
    outputs: *const TjgBuffer,
    n_outputs: usize,
    err_buf: *mut u8,
    err_cap: usize,
) -> i32 {
    let inputs = std::slice::from_raw_parts(inputs, n_inputs);
    let outputs = std::slice::from_raw_parts(outputs, n_outputs);

    match dispatch(stream, token, inputs, outputs) {
        Ok(()) => 0,
        Err(msg) => {
            write_err(err_buf, err_cap, &msg);
            1
        }
    }
}

unsafe fn write_err(err_buf: *mut u8, err_cap: usize, msg: &str) {
    if err_buf.is_null() || err_cap == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(err_cap - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), err_buf, n);
    *err_buf.add(n) = 0;
}

// ---------------------------------------------------------------------------
// The dispatch itself
// ---------------------------------------------------------------------------

unsafe fn dispatch(
    stream: *mut c_void,
    token: i64,
    inputs: &[TjgBuffer],
    outputs: &[TjgBuffer],
) -> Result<(), String> {
    let desc = {
        let reg = registry().lock().map_err(|e| e.to_string())?;
        reg.get(&token)
            .cloned()
            .ok_or_else(|| format!("unknown dispatch token {token}"))?
    };

    let rt = cuda::runtime()?;

    // Inputs must be ready before we read them; XLA gives us its stream.
    rt.stream_synchronize(stream)?;

    // ---- Build the JSON request: export each GPU input via CUDA IPC ----
    //
    // GPU-direct inputs (no host copy). XLA's input pointers are VMM/pool-backed,
    // which the legacy cudaIpcGetMemHandle API rejects, so we stage each input's
    // bytes into a fresh cudaMalloc buffer (one on-GPU copy) and export a handle
    // to that -- mirroring tesseract-core's staging fallback, in native code.
    //
    // The staging buffers must stay alive until the server has copied the inputs
    // out of them. The server does that synchronously while handling this request
    // (decode copies on open), so it is safe to free them right after the HTTP
    // response returns. `staging` holds them until then.
    let mut input_objs = serde_json::Map::new();
    let mut staging: Vec<*mut c_void> = Vec::with_capacity(inputs.len());
    // Ensure staging buffers are freed on every exit path (including errors).
    let cleanup = |rt: &cuda::Runtime, staging: &[*mut c_void]| {
        for &p in staging {
            let _ = rt.free(p);
        }
    };

    for (buf, key) in inputs.iter().zip(desc.input_keys.iter()) {
        let name = match dtype_name(buf.dtype) {
            Some(n) => n,
            None => {
                cleanup(rt, &staging);
                return Err(format!("unsupported input dtype code {}", buf.dtype));
            }
        };
        let shape = buf.shape();
        let nb = nbytes(&shape, dtype_itemsize(name));

        let res: Result<[u8; 64], String> = (|| {
            let stage = rt.malloc(nb)?;
            staging.push(stage);
            rt.memcpy_dtod_sync(stage, buf.data as *const c_void, nb)?;
            rt.ipc_get_handle(stage)
        })();
        let handle = match res {
            Ok(h) => h,
            Err(e) => {
                cleanup(rt, &staging);
                return Err(e);
            }
        };

        let handle_b64 = base64::engine::general_purpose::STANDARD.encode(handle);
        input_objs.insert(
            key.clone(),
            serde_json::json!({
                "object_type": "array",
                "shape": shape,
                "dtype": name,
                "data": {
                    "handle": handle_b64,
                    "device": 0,
                    "storage_offset": 0,
                    "storage_size": nb,
                    "encoding": "cuda_ipc",
                },
            }),
        );
    }
    let body = serde_json::json!({"inputs": input_objs});
    let body = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => {
            cleanup(rt, &staging);
            return Err(e.to_string());
        }
    };

    // ---- HTTP POST ----
    let resp = match ureq::post(&desc.apply_url)
        .set("Content-Type", "application/json")
        .set("Accept", "application/json+cuda_ipc")
        .send_bytes(&body)
    {
        Ok(r) => r,
        Err(e) => {
            cleanup(rt, &staging);
            return Err(format!("HTTP apply failed: {e}"));
        }
    };
    let payload: serde_json::Value = match serde_json::from_reader(resp.into_reader()) {
        Ok(p) => p,
        Err(e) => {
            cleanup(rt, &staging);
            return Err(e.to_string());
        }
    };

    // ---- For each output, decode cuda_ipc and copy into XLA's buffer ----
    let mut result = Ok(());
    for (buf, key) in outputs.iter().zip(desc.output_keys.iter()) {
        match payload.get(key) {
            Some(arr) => {
                if let Err(e) = decode_cuda_ipc_into(&rt, arr, buf, stream) {
                    result = Err(e);
                    break;
                }
            }
            None => {
                result = Err(format!("output '{key}' missing from response"));
                break;
            }
        }
    }

    // Ensure copies complete before we return (XLA reuses these buffers).
    if result.is_ok() {
        result = rt.stream_synchronize(stream);
    }

    // The server has copied the inputs out during the request; free staging now.
    cleanup(rt, &staging);
    result
}

/// Open a cuda_ipc array handle and copy its bytes into `out` (device->device).
unsafe fn decode_cuda_ipc_into(
    rt: &cuda::Runtime,
    arr: &serde_json::Value,
    out: &TjgBuffer,
    stream: *mut c_void,
) -> Result<(), String> {
    let data = arr.get("data").ok_or("array missing 'data'")?;
    let encoding = data.get("encoding").and_then(|v| v.as_str()).unwrap_or("");
    if encoding != "cuda_ipc" {
        return Err(format!(
            "expected cuda_ipc output encoding, got '{encoding}'"
        ));
    }
    let handle_b64 = data.get("handle").and_then(|v| v.as_str()).ok_or("no handle")?;
    let handle = base64::engine::general_purpose::STANDARD
        .decode(handle_b64)
        .map_err(|e| e.to_string())?;
    if handle.len() != 64 {
        return Err(format!("bad IPC handle length {}", handle.len()));
    }
    let device = data.get("device").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let storage_offset = data
        .get("storage_offset")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as usize;

    let out_shape = out.shape();
    let out_name = dtype_name(out.dtype)
        .ok_or_else(|| format!("unsupported output dtype code {}", out.dtype))?;
    let nb = nbytes(&out_shape, dtype_itemsize(out_name));

    rt.set_device(device)?;
    let base = rt.ipc_open(&handle, device)?;
    let src = (base as usize + storage_offset) as *const c_void;
    let copy_res = rt.memcpy_dtod(out.data, src, nb, stream);
    // Always close the mapping, even on copy failure.
    let close_res = rt.ipc_close(base);
    copy_res?;
    close_res?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 module
// ---------------------------------------------------------------------------

extern "C" {
    fn tjgpu_b_handler() -> *mut c_void;
}

/// Register a dispatch descriptor for `token`. Called from Python before the
/// ffi_call is issued.
#[pyfunction]
fn register_descriptor(
    token: i64,
    apply_url: String,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
) -> PyResult<()> {
    let mut reg = registry()
        .lock()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    reg.insert(
        token,
        Descriptor {
            apply_url,
            input_keys,
            output_keys,
        },
    );
    Ok(())
}

#[pyfunction]
fn release_descriptor(token: i64) {
    if let Ok(mut reg) = registry().lock() {
        reg.remove(&token);
    }
}

/// Return the XLA FFI handler as a PyCapsule for jax.ffi.register_ffi_target.
///
/// We build the capsule via the CPython C-API directly (pyo3-ffi) because we
/// carry a raw function pointer, not a Rust-owned value. The capsule name must
/// be "xla._CUSTOM_CALL_TARGET" and there is no destructor (the handler is a
/// 'static function).
#[pyfunction]
fn handler_capsule(py: Python<'_>) -> PyResult<PyObject> {
    // Keep the name string alive for the capsule's lifetime. PyCapsule stores
    // the name pointer without copying, so it must not be freed. We leak a
    // single CString (once per process) to guarantee that.
    static CAPSULE_NAME: OnceLock<std::ffi::CString> = OnceLock::new();
    let name = CAPSULE_NAME
        .get_or_init(|| std::ffi::CString::new("xla._CUSTOM_CALL_TARGET").unwrap());

    let ptr = unsafe { tjgpu_b_handler() };
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(ptr, name.as_ptr(), None)
    };
    if capsule.is_null() {
        return Err(pyo3::PyErr::fetch(py));
    }
    Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
}

#[pymodule]
fn tesseract_jax_gpu_b(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(release_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(handler_capsule, m)?)?;
    Ok(())
}
