use std::env;
use std::path;

use cuda_builder::CudaBuilder;
use cuda_builder::NvvmArch;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");

    let galpy_dir = "/data/astrodrift/galpy";

    println!("cargo:rustc-link-search=native={galpy_dir}");

    println!("cargo:rustc-link-lib=dylib=galpy.cpython-313-x86_64-linux-gnu");

    println!("cargo:rustc-link-arg=-Wl,-rpath,{galpy_dir}");

    let out_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    CudaBuilder::new(manifest_dir.join("kernels"))
        .fast_div(false)
        .fast_sqrt(false)
        .fma_contraction(false)
        .ftz(false)
        .arch(NvvmArch::Compute80)
        .use_constant_memory_space(false)
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}
