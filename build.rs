// Two kernel backends, selected by feature:
//
// - default (modern): embed the cubin produced ahead of time by
//   ./build-cuda-oxide-kernels.sh (cuda-oxide rustc backend + LLVM 21 tools
//   + CUDA 13 libnvvm/nvJitLink). No LLVM 7 anywhere.
// - `legacy`: compile kernels/ to PTX at build time via cuda_builder /
//   rustc_codegen_nvvm, which links llvm-sys and therefore needs the custom
//   LLVM 7 build (LLVM_CONFIG, LLVM_LINK_STATIC=1; see
//   container/ubuntu24-cuda13/Dockerfile).
#[cfg(feature = "legacy")]
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    let mut builder = cuda_builder::CudaBuilder::new(manifest_dir.join("kernels"))
        .fast_div(false)
        .fast_sqrt(false)
        .fma_contraction(false)
        .ftz(false)
        .arch(cuda_builder::NvvmArch::Compute80)
        .release(true)
        .use_constant_memory_space(false);

    if cfg!(feature = "galpy-kepler-reference") {
        builder = builder.build_args(&["--features", "galpy-kepler-reference"]);
    }

    builder
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}

#[cfg(not(feature = "legacy"))]
fn main() {
    // The kernels crate is never built by cargo here: ./build-cuda-oxide-kernels.sh
    // compiles kernels/ + shared/ in a shadow workspace via the cuda-oxide rustc
    // backend and writes target/cuda-oxide/kernels.cubin, which is embedded below.
    println!("cargo::rerun-if-changed=build.rs");
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let cubin = if cfg!(feature = "galpy-kepler-reference") {
        manifest_dir.join("target/cuda-oxide/galpy-kepler-reference/kernels.cubin")
    } else {
        manifest_dir.join("target/cuda-oxide/kernels.cubin")
    };

    println!("cargo::rerun-if-changed={}", cubin.display());

    std::fs::copy(&cubin, out_path.join("kernels.cubin")).unwrap_or_else(|err| {
        panic!(
            "failed to copy cuda-oxide cubin from {}: {err}. Run ./build-cuda-oxide-kernels.sh with matching features first",
            cubin.display()
        )
    });
}
