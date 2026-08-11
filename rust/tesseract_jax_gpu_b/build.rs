// Compile the minimal C++ FFI glue (glue.cc) and link it into the crate.
//
// The glue needs XLA's header-only FFI API, which ships inside jaxlib. We ask
// the configured Python for its location so the build works against whatever
// jaxlib is installed. No CUDA headers are needed (the glue only forward-uses
// cudaStream_t as an opaque pointer via ffi.h's PlatformStream).

use std::process::Command;

fn main() {
    let python = std::env::var("TJGPU_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let jaxlib_inc = Command::new(&python)
        .args([
            "-c",
            "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))",
        ])
        .output()
        .expect("failed to run python to locate jaxlib include dir");
    let jaxlib_inc = String::from_utf8(jaxlib_inc.stdout)
        .expect("non-utf8 jaxlib include path")
        .trim()
        .to_string();
    assert!(
        !jaxlib_inc.is_empty(),
        "could not locate jaxlib include dir (is jaxlib installed for {python}?)"
    );

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-fvisibility=hidden")
        .include(&jaxlib_inc)
        .file("glue.cc")
        .compile("tjgpu_b_glue");

    println!("cargo:rerun-if-changed=glue.cc");
    println!("cargo:rerun-if-env-changed=TJGPU_PYTHON");
}
