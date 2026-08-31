#[cfg(all(not(feature = "cuda-oxide-kernel"), not(feature = "nvvm-kernel")))]
compile_error!(
    "no kernel backend selected: use default (nvvm-kernel, needs LLVM 7) or --features cuda-oxide-kernel"
);

#[cfg(all(not(feature = "cuda-oxide-kernel"), feature = "nvvm-kernel"))]
use cuda_builder::CudaBuilder;
#[cfg(all(not(feature = "cuda-oxide-kernel"), feature = "nvvm-kernel"))]
use cuda_builder::NvvmArch;

#[cfg(all(not(feature = "cuda-oxide-kernel"), feature = "nvvm-kernel"))]
fn main() {
    // if std::env::var_os("DOCS_RS").is_some() {
    //     // println!("cargo:warning=build.rs skipped for docs build");
    //     return;
    // }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    let mut builder = CudaBuilder::new(manifest_dir.join("kernels"))
        .fast_div(false)
        .fast_sqrt(false)
        .fma_contraction(false)
        .ftz(false)
        .arch(NvvmArch::Compute80)
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

#[cfg(feature = "cuda-oxide-kernel")]
fn main() {
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
