// The kernels crate is never built by cargo here: ./build-cuda-oxide-kernels.sh
// compiles kernels/ + shared/ in a shadow workspace via the cuda-oxide rustc
// backend and writes target/cuda-oxide/kernels.cubin, which is embedded below.
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
