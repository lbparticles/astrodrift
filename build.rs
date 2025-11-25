use std::env;
use std::path;

use cuda_builder::CudaBuilder;
use cuda_builder::NvvmArch;

fn main() {
    if std::env::var_os("RUSTDOC").is_some() || std::env::var_os("DOCS_RS").is_some() {
        println!("cargo:warning=build.rs skipped for docs build");
        return;
    }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");
    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    CudaBuilder::new(manifest_dir.join("kernels"))
        .fast_div(false)
        .fast_sqrt(false)
        .fma_contraction(false)
        .ftz(false)
        .arch(NvvmArch::Compute80)
        .release(false)
        .use_constant_memory_space(false)
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}
