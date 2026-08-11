// Copyright 2025 Pasteur Labs. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
//! Minimal CUDA runtime access via `libloading` (no compile-time CUDA link).
//!
//! Mirrors the handful of symbols tesseract-core reaches through ctypes. The
//! runtime library is dlopen'd by name at first use, so the crate carries no
//! CUDA-version build dependency (works against libcudart .11/.12/.13).

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::sync::OnceLock;

use libloading::{Library, Symbol};

const CUDA_SUCCESS: c_int = 0;
const CUDA_MEMCPY_HTOD: c_int = 1;
const CUDA_MEMCPY_DTOH: c_int = 2;
const CUDA_MEMCPY_DTOD: c_int = 3;
const CUDA_IPC_LAZY_ENABLE_PEER_ACCESS: u32 = 0x01;

/// ctypes mirror of `cudaIpcMemHandle_t` -- an opaque 64-byte blob that MUST be
/// passed by value (as a struct), matching the C ABI. Passing a pointer makes
/// cudaIpcOpenMemHandle fail with cudaErrorInvalidValue.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaIpcMemHandle {
    reserved: [c_char; 64],
}

// Function-pointer signatures for the runtime symbols we use.
type SetDevice = unsafe extern "C" fn(c_int) -> c_int;
type Memcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type MemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> c_int;
type StreamSync = unsafe extern "C" fn(*mut c_void) -> c_int;
type IpcOpen = unsafe extern "C" fn(*mut *mut c_void, CudaIpcMemHandle, u32) -> c_int;
type IpcClose = unsafe extern "C" fn(*mut c_void) -> c_int;
type IpcGetHandle = unsafe extern "C" fn(*mut CudaIpcMemHandle, *mut c_void) -> c_int;
type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type Free = unsafe extern "C" fn(*mut c_void) -> c_int;
type GetErrorString = unsafe extern "C" fn(c_int) -> *const c_char;

pub struct Runtime {
    _lib: Library,
    set_device: RawSymbol<SetDevice>,
    memcpy: RawSymbol<Memcpy>,
    memcpy_async: RawSymbol<MemcpyAsync>,
    stream_sync: RawSymbol<StreamSync>,
    ipc_open: RawSymbol<IpcOpen>,
    ipc_close: RawSymbol<IpcClose>,
    ipc_get_handle: RawSymbol<IpcGetHandle>,
    malloc: RawSymbol<Malloc>,
    free: RawSymbol<Free>,
    get_error_string: Option<RawSymbol<GetErrorString>>,
}

// We keep raw symbols (function pointers) alongside the Library that owns them.
// The Library outlives them (same struct), so this is sound.
struct RawSymbol<T>(T);
unsafe impl<T> Send for RawSymbol<T> {}
unsafe impl<T> Sync for RawSymbol<T> {}

unsafe fn load<T: Copy>(lib: &Library, name: &[u8]) -> Result<RawSymbol<T>, String> {
    let sym: Symbol<T> = lib
        .get(name)
        .map_err(|e| format!("missing CUDA symbol {}: {e}", String::from_utf8_lossy(name)))?;
    Ok(RawSymbol(*sym))
}

fn open_libcudart() -> Result<Library, String> {
    let names = [
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11",
    ];
    for n in names {
        if let Ok(lib) = unsafe { Library::new(n) } {
            return Ok(lib);
        }
    }
    Err("could not dlopen libcudart (is CUDA installed?)".to_string())
}

pub fn runtime() -> Result<&'static Runtime, String> {
    static RT: OnceLock<Result<Runtime, String>> = OnceLock::new();
    RT.get_or_init(|| {
        let lib = open_libcudart()?;
        unsafe {
            let set_device = load(&lib, b"cudaSetDevice")?;
            let memcpy = load(&lib, b"cudaMemcpy")?;
            let memcpy_async = load(&lib, b"cudaMemcpyAsync")?;
            let stream_sync = load(&lib, b"cudaStreamSynchronize")?;
            let ipc_open = load(&lib, b"cudaIpcOpenMemHandle")?;
            let ipc_close = load(&lib, b"cudaIpcCloseMemHandle")?;
            let ipc_get_handle = load(&lib, b"cudaIpcGetMemHandle")?;
            let malloc = load(&lib, b"cudaMalloc")?;
            let free = load(&lib, b"cudaFree")?;
            let get_error_string = load(&lib, b"cudaGetErrorString").ok();
            Ok(Runtime {
                _lib: lib,
                set_device,
                memcpy,
                memcpy_async,
                stream_sync,
                ipc_open,
                ipc_close,
                ipc_get_handle,
                malloc,
                free,
                get_error_string,
            })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

impl Runtime {
    fn err(&self, code: c_int) -> String {
        if let Some(ges) = &self.get_error_string {
            unsafe {
                let s = (ges.0)(code);
                if !s.is_null() {
                    return std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned();
                }
            }
        }
        format!("cuda error {code}")
    }

    pub fn set_device(&self, device: i32) -> Result<(), String> {
        let rc = unsafe { (self.set_device.0)(device) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaSetDevice({device}) failed: {}", self.err(rc)));
        }
        Ok(())
    }

    pub fn stream_synchronize(&self, stream: *mut c_void) -> Result<(), String> {
        let rc = unsafe { (self.stream_sync.0)(stream) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaStreamSynchronize failed: {}", self.err(rc)));
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn memcpy_dtoh(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> Result<(), String> {
        let rc = unsafe { (self.memcpy.0)(dst, src, n, CUDA_MEMCPY_DTOH) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaMemcpy(DtoH) failed: {}", self.err(rc)));
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn memcpy_htod(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> Result<(), String> {
        let rc = unsafe { (self.memcpy.0)(dst, src, n, CUDA_MEMCPY_HTOD) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaMemcpy(HtoD) failed: {}", self.err(rc)));
        }
        Ok(())
    }

    pub fn memcpy_dtod(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let rc = unsafe { (self.memcpy_async.0)(dst, src, n, CUDA_MEMCPY_DTOD, stream) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaMemcpyAsync(DtoD) failed: {}", self.err(rc)));
        }
        Ok(())
    }

    pub fn ipc_open(&self, handle: &[u8], _device: i32) -> Result<*mut c_void, String> {
        let mut h = CudaIpcMemHandle { reserved: [0; 64] };
        for (i, b) in handle.iter().enumerate().take(64) {
            h.reserved[i] = *b as c_char;
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (self.ipc_open.0)(&mut ptr, h, CUDA_IPC_LAZY_ENABLE_PEER_ACCESS)
        };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaIpcOpenMemHandle failed: {}", self.err(rc)));
        }
        if ptr.is_null() {
            return Err("cudaIpcOpenMemHandle returned null".to_string());
        }
        Ok(ptr)
    }

    pub fn ipc_close(&self, base: *mut c_void) -> Result<(), String> {
        let rc = unsafe { (self.ipc_close.0)(base) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaIpcCloseMemHandle failed: {}", self.err(rc)));
        }
        Ok(())
    }

    pub fn malloc(&self, n: usize) -> Result<*mut c_void, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { (self.malloc.0)(&mut ptr, n) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaMalloc({n}) failed: {}", self.err(rc)));
        }
        Ok(ptr)
    }

    pub fn free(&self, ptr: *mut c_void) -> Result<(), String> {
        let rc = unsafe { (self.free.0)(ptr) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaFree failed: {}", self.err(rc)));
        }
        Ok(())
    }

    /// Synchronous device->device copy (default stream). Used to stage an input
    /// into a fresh cudaMalloc buffer before exporting it.
    pub fn memcpy_dtod_sync(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
    ) -> Result<(), String> {
        let rc = unsafe { (self.memcpy.0)(dst, src, n, CUDA_MEMCPY_DTOD) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaMemcpy(DtoD) failed: {}", self.err(rc)));
        }
        Ok(())
    }

    /// cudaIpcGetMemHandle on a device pointer. Returns the 64 raw handle bytes.
    /// The pointer must be a plain cudaMalloc allocation (the legacy IPC API
    /// rejects VMM/pool-backed memory, which is why callers stage first).
    pub fn ipc_get_handle(&self, ptr: *mut c_void) -> Result<[u8; 64], String> {
        let mut h = CudaIpcMemHandle { reserved: [0; 64] };
        let rc = unsafe { (self.ipc_get_handle.0)(&mut h, ptr) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cudaIpcGetMemHandle failed: {}", self.err(rc)));
        }
        let mut out = [0u8; 64];
        for (i, b) in h.reserved.iter().enumerate() {
            out[i] = *b as u8;
        }
        Ok(out)
    }
}
