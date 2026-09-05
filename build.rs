#[cfg(feature = "rust-cuda")]
use cuda_builder::CudaBuilder;
#[cfg(feature = "rust-cuda")]
use cuda_builder::NvvmArch;

#[cfg(all(feature = "rust-cuda", feature = "cuda-oxide"))]
compile_error!("features `rust-cuda` and `cuda-oxide` are mutually exclusive");

#[cfg(not(any(feature = "rust-cuda", feature = "cuda-oxide")))]
compile_error!("enable exactly one CUDA backend feature: `rust-cuda` or `cuda-oxide`");

#[cfg(all(feature = "rust-cuda", not(feature = "cuda-oxide")))]
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

    let kernel_features = if cfg!(feature = "galpy-kepler-reference") {
        "rust-cuda,galpy-kepler-reference"
    } else {
        "rust-cuda"
    };
    builder = builder.build_args(&["--features", kernel_features]);

    builder
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}

#[cfg(all(feature = "cuda-oxide", not(feature = "rust-cuda")))]
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
}
