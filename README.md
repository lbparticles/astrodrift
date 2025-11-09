```
          ###      ######## ########### #########   ########    
       ### ###   ###    ###    ###     ###    ### ###    ###    
     ###   ###  ###           ###     ###    ### ###    ###     
   ########### ##########    ###     #########  ###    ###      
  ###     ###        ###    ###     ###    ### ###    ###       
 ###     ### ###    ###    ###     ###    ### ###    ###        
###     ###  ########     ###     ###    ###  ########          
      #########  #########  ########### ########## ###########  
     ###    ### ###    ###     ###     ###            ###       
    ###    ### ###    ###     ###     ###            ###        
   ###    ### #########      ###     ########       ###         
  ###    ### ###    ###     ###     ###            ###          
 ###    ### ###    ###     ###     ###            ###           
#########  ###    ### ########### ###            ###           
```

`astrodrift` is a python library that provides numerical integrator for arbitrary potentials specialising in galatic dynamics simulations. The library provides both cpu compiled and gpu accelerated integration methods utilising a rust-based backend. The library focuses on large quantities of non-interacting test particle integrations particularly useful for tidal stream dynamics. The library also provides interpolation of moving potential increased performance.

# INSTALLATION

```python -m pip install astrodrift```

```uv add astrodrift```

# CONTRIBUTION

To install from source 

```git clone https://github.com/lbparticles/astrodrift```

To use the gpu functions calls you must have an nvidia gpu and drivers installed on your system, download them from the [official website]( https://www.nvidia.com/en-us/drivers/) or use your os package manager.

Inside the containers/, there are two provided Dockerfile to build ubuntu22 and ubuntu24 versions of an apptainer that has the necessary nvidia toolkits installed. make sure that you have installed docker and apptainer installed. There are also example shell scripts that can be modified to create a docker container in the root of project -- WORKDIR. There is a translation script from docker to apptainer. There is a run.sh script that opens the apptainer with the project bound to /data/astrodrift and sets it to the current working directory.

# CONTRIBUTORS

Jack Patterson
Angus Forrest
John Forbes

# DEPENDENCIES

## Python
<!-- PYTHON_DEPS_START -->
```toml
dependencies = [
    "numpy>=1.24.4",
    "pandas>=2.0.3",
    "patchelf>=0.17.2.4",
]
```

<details>
<summary>Python full dependency tree</summary>


astrodrift
├── numpy v2.3.4
├── pandas v2.3.3
│   ├── numpy v2.3.4
│   ├── python-dateutil v2.9.0.post0
│   │   └── six v1.17.0
│   ├── pytz v2025.2
│   └── tzdata v2025.2
└── patchelf v0.17.2.4

</details>
<!-- PYTHON_DEPS_END -->

## Rust
<!-- RUST_DEPS_START -->
```toml
[dependencies]
pyo3 = "0.25.1"
blastoff = { git = "https://github.com/Rust-GPU/Rust-CUDA", package = "blastoff", branch = "main" }
cuda_std = { git = "https://github.com/Rust-GPU/Rust-CUDA", package = "cuda_std", branch = "main" }
cust = { git = "https://github.com/Rust-GPU/Rust-CUDA", package = "cust", branch = "main" }
cust_raw = { git = "https://github.com/Rust-GPU/Rust-CUDA", package = "cust_raw", branch = "main", features = ["driver"] }
ndarray = { version = "0.16", features = ["approx"] }
ndarray-rand = "0.15.0"
rand = "0.9.2"
rayon = "1.10.0"
libm = "0.2.11"
statrs = "0.18.0"
numpy = "0.25.0"

[build-dependencies]
cuda_builder = { git = "https://github.com/Rust-GPU/Rust-CUDA", package = "cuda_builder", branch = "main" }
```

<details>
    <summary>Rust full dependency tree</summary>


drift v0.1.0 (/data/astrodrift)
├── blastoff v0.1.0 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   ├── bitflags v2.9.1
│   ├── cust v0.3.2 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   │   ├── bitflags v2.9.1
│   │   ├── bytemuck v1.23.1
│   │   │   └── bytemuck_derive v1.10.0 (proc-macro)
│   │   │       ├── proc-macro2 v1.0.95
│   │   │       │   └── unicode-ident v1.0.18
│   │   │       ├── quote v1.0.40
│   │   │       │   └── proc-macro2 v1.0.95 (*)
│   │   │       └── syn v2.0.104
│   │   │           ├── proc-macro2 v1.0.95 (*)
│   │   │           ├── quote v1.0.40 (*)
│   │   │           └── unicode-ident v1.0.18
│   │   ├── cust_core v0.1.1 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   │   │   ├── cust_derive v0.2.0 (proc-macro) (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   │   │   │   ├── proc-macro2 v1.0.95 (*)
│   │   │   │   ├── quote v1.0.40 (*)
│   │   │   │   └── syn v2.0.104 (*)
│   │   │   ├── glam v0.30.5
│   │   │   │   ├── bytemuck v1.23.1 (*)
│   │   │   │   └── libm v0.2.15
│   │   │   ├── mint v0.5.9
│   │   │   ├── num-complex v0.4.6
│   │   │   │   └── num-traits v0.2.19
│   │   │   │       └── libm v0.2.15
│   │   │   │       [build-dependencies]
│   │   │   │       └── autocfg v1.5.0
│   │   │   └── vek v0.17.1
│   │   │       ├── approx v0.5.1
│   │   │       │   └── num-traits v0.2.19 (*)
│   │   │       ├── num-integer v0.1.46
│   │   │       │   └── num-traits v0.2.19 (*)
│   │   │       └── num-traits v0.2.19 (*)
│   │   │       [build-dependencies]
│   │   │       └── rustc_version v0.4.1
│   │   │           └── semver v1.0.26
│   │   ├── cust_derive v0.2.0 (proc-macro) (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68) (*)
│   │   ├── cust_raw v0.11.3 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   │   │   [build-dependencies]
│   │   │   ├── bimap v0.6.3
│   │   │   ├── bindgen v0.71.1
│   │   │   │   ├── bitflags v2.9.1
│   │   │   │   ├── cexpr v0.6.0
│   │   │   │   │   └── nom v7.1.3
│   │   │   │   │       ├── memchr v2.7.5
│   │   │   │   │       └── minimal-lexical v0.2.1
│   │   │   │   ├── clang-sys v1.8.1
│   │   │   │   │   ├── glob v0.3.2
│   │   │   │   │   ├── libc v0.2.174
│   │   │   │   │   └── libloading v0.8.8
│   │   │   │   │       └── cfg-if v1.0.1
│   │   │   │   │   [build-dependencies]
│   │   │   │   │   └── glob v0.3.2
│   │   │   │   ├── itertools v0.13.0
│   │   │   │   │   └── either v1.15.0
│   │   │   │   ├── log v0.4.27
│   │   │   │   ├── prettyplease v0.2.36
│   │   │   │   │   ├── proc-macro2 v1.0.95 (*)
│   │   │   │   │   └── syn v2.0.104 (*)
│   │   │   │   ├── proc-macro2 v1.0.95 (*)
│   │   │   │   ├── quote v1.0.40 (*)
│   │   │   │   ├── regex v1.11.1
│   │   │   │   │   ├── regex-automata v0.4.9
│   │   │   │   │   │   └── regex-syntax v0.8.5
│   │   │   │   │   └── regex-syntax v0.8.5
│   │   │   │   ├── rustc-hash v2.1.1
│   │   │   │   ├── shlex v1.3.0
│   │   │   │   └── syn v2.0.104 (*)
│   │   │   ├── cc v1.2.30
│   │   │   │   ├── jobserver v0.1.33
│   │   │   │   │   └── libc v0.2.174
│   │   │   │   ├── libc v0.2.174
│   │   │   │   └── shlex v1.3.0
│   │   │   └── doxygen-bindgen v0.1.3
│   │   │       └── yap v0.12.0
│   │   ├── glam v0.30.5 (*)
│   │   ├── mint v0.5.9
│   │   ├── num-complex v0.4.6 (*)
│   │   └── vek v0.17.1 (*)
│   │   [build-dependencies]
│   │   └── serde_json v1.0.141
│   │       ├── itoa v1.0.15
│   │       ├── memchr v2.7.5
│   │       ├── ryu v1.0.20
│   │       └── serde v1.0.219
│   │           └── serde_derive v1.0.219 (proc-macro)
│   │               ├── proc-macro2 v1.0.95 (*)
│   │               ├── quote v1.0.40 (*)
│   │               └── syn v2.0.104 (*)
│   ├── cust_raw v0.11.3 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68) (*)
│   └── num-complex v0.4.6 (*)
├── cuda_std v0.2.2 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   ├── bitflags v2.9.1
│   ├── cuda_std_macros v0.2.0 (proc-macro) (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
│   │   ├── proc-macro2 v1.0.95 (*)
│   │   ├── quote v1.0.40 (*)
│   │   └── syn v2.0.104 (*)
│   ├── glam v0.30.5 (*)
│   ├── half v2.6.0
│   │   └── cfg-if v1.0.1
│   ├── paste v1.0.15 (proc-macro)
│   └── vek v0.17.1 (*)
├── cust v0.3.2 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68) (*)
├── cust_raw v0.11.3 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68) (*)
├── libm v0.2.15
├── ndarray v0.16.1
│   ├── approx v0.5.1 (*)
│   ├── matrixmultiply v0.3.10
│   │   └── rawpointer v0.2.1
│   │   [build-dependencies]
│   │   └── autocfg v1.5.0
│   ├── num-complex v0.4.6 (*)
│   ├── num-integer v0.1.46 (*)
│   ├── num-traits v0.2.19 (*)
│   └── rawpointer v0.2.1
├── ndarray-rand v0.15.0
│   ├── ndarray v0.16.1 (*)
│   ├── rand v0.8.5
│   │   ├── libc v0.2.174
│   │   ├── rand_chacha v0.3.1
│   │   │   ├── ppv-lite86 v0.2.21
│   │   │   │   └── zerocopy v0.8.26
│   │   │   └── rand_core v0.6.4
│   │   │       └── getrandom v0.2.16
│   │   │           ├── cfg-if v1.0.1
│   │   │           └── libc v0.2.174
│   │   └── rand_core v0.6.4 (*)
│   └── rand_distr v0.4.3
│       ├── num-traits v0.2.19 (*)
│       └── rand v0.8.5 (*)
├── numpy v0.25.0
│   ├── libc v0.2.174
│   ├── ndarray v0.16.1 (*)
│   ├── num-complex v0.4.6 (*)
│   ├── num-integer v0.1.46 (*)
│   ├── num-traits v0.2.19 (*)
│   ├── pyo3 v0.25.1
│   │   ├── indoc v2.0.6 (proc-macro)
│   │   ├── libc v0.2.174
│   │   ├── memoffset v0.9.1
│   │   │   [build-dependencies]
│   │   │   └── autocfg v1.5.0
│   │   ├── once_cell v1.21.3
│   │   ├── pyo3-ffi v0.25.1
│   │   │   └── libc v0.2.174
│   │   │   [build-dependencies]
│   │   │   └── pyo3-build-config v0.25.1
│   │   │       ├── once_cell v1.21.3
│   │   │       └── target-lexicon v0.13.2
│   │   │       [build-dependencies]
│   │   │       └── target-lexicon v0.13.2
│   │   ├── pyo3-macros v0.25.1 (proc-macro)
│   │   │   ├── proc-macro2 v1.0.95 (*)
│   │   │   ├── pyo3-macros-backend v0.25.1
│   │   │   │   ├── heck v0.5.0
│   │   │   │   ├── proc-macro2 v1.0.95 (*)
│   │   │   │   ├── pyo3-build-config v0.25.1 (*)
│   │   │   │   ├── quote v1.0.40 (*)
│   │   │   │   └── syn v2.0.104 (*)
│   │   │   │   [build-dependencies]
│   │   │   │   └── pyo3-build-config v0.25.1 (*)
│   │   │   ├── quote v1.0.40 (*)
│   │   │   └── syn v2.0.104 (*)
│   │   └── unindent v0.2.4
│   │   [build-dependencies]
│   │   └── pyo3-build-config v0.25.1 (*)
│   └── rustc-hash v2.1.1
│   [build-dependencies]
│   └── pyo3-build-config v0.25.1 (*)
├── pyo3 v0.25.1 (*)
├── rand v0.9.2
│   ├── rand_chacha v0.9.0
│   │   ├── ppv-lite86 v0.2.21 (*)
│   │   └── rand_core v0.9.3
│   │       └── getrandom v0.3.3
│   │           ├── cfg-if v1.0.1
│   │           └── libc v0.2.174
│   └── rand_core v0.9.3 (*)
├── rayon v1.10.0
│   ├── either v1.15.0
│   └── rayon-core v1.12.1
│       ├── crossbeam-deque v0.8.6
│       │   ├── crossbeam-epoch v0.9.18
│       │   │   └── crossbeam-utils v0.8.21
│       │   └── crossbeam-utils v0.8.21
│       └── crossbeam-utils v0.8.21
└── statrs v0.18.0
    ├── approx v0.5.1 (*)
    ├── nalgebra v0.33.2
    │   ├── approx v0.5.1 (*)
    │   ├── matrixmultiply v0.3.10 (*)
    │   ├── num-complex v0.4.6 (*)
    │   ├── num-rational v0.4.2
    │   │   ├── num-integer v0.1.46 (*)
    │   │   └── num-traits v0.2.19 (*)
    │   ├── num-traits v0.2.19 (*)
    │   ├── rand v0.8.5 (*)
    │   ├── rand_distr v0.4.3 (*)
    │   ├── simba v0.9.0
    │   │   ├── approx v0.5.1 (*)
    │   │   ├── num-complex v0.4.6 (*)
    │   │   ├── num-traits v0.2.19 (*)
    │   │   ├── paste v1.0.15 (proc-macro)
    │   │   └── wide v0.7.33
    │   │       ├── bytemuck v1.23.1 (*)
    │   │       └── safe_arch v0.7.4
    │   │           └── bytemuck v1.23.1 (*)
    │   └── typenum v1.18.0
    ├── num-traits v0.2.19 (*)
    └── rand v0.8.5 (*)
[build-dependencies]
└── cuda_builder v0.3.0 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
    ├── nvvm v0.1.1 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
    │   └── cust_raw v0.11.3 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
    │       [build-dependencies]
    │       ├── bimap v0.6.3
    │       ├── bindgen v0.71.1 (*)
    │       ├── cc v1.2.30 (*)
    │       └── doxygen-bindgen v0.1.3 (*)
    ├── rustc_codegen_nvvm v0.3.0 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
    │   ├── bitflags v2.9.1
    │   ├── gimli v0.30.0
    │   │   ├── fallible-iterator v0.3.0
    │   │   ├── indexmap v2.10.0
    │   │   │   ├── equivalent v1.0.2
    │   │   │   └── hashbrown v0.15.4
    │   │   └── stable_deref_trait v1.2.0
    │   ├── itertools v0.14.0
    │   │   └── either v1.15.0
    │   ├── libc v0.2.174
    │   ├── libloading v0.8.8 (*)
    │   ├── nvvm v0.1.1 (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68) (*)
    │   ├── object v0.36.7
    │   │   ├── flate2 v1.1.2
    │   │   │   ├── crc32fast v1.5.0
    │   │   │   │   └── cfg-if v1.0.1
    │   │   │   └── miniz_oxide v0.8.9
    │   │   │       └── adler2 v2.0.1
    │   │   ├── memchr v2.7.5
    │   │   └── ruzstd v0.7.3
    │   │       └── twox-hash v1.6.3
    │   │           ├── cfg-if v1.0.1
    │   │           └── static_assertions v1.1.0
    │   ├── rustc-demangle v0.1.25
    │   ├── rustc_codegen_nvvm_macros v0.1.0 (proc-macro) (https://github.com/Rust-GPU/Rust-CUDA?branch=main#3b646e68)
    │   │   ├── proc-macro2 v1.0.95 (*)
    │   │   ├── quote v1.0.40 (*)
    │   │   └── syn v2.0.104 (*)
    │   ├── smallvec v1.15.1
    │   ├── tar v0.4.44
    │   │   ├── filetime v0.2.25
    │   │   │   ├── cfg-if v1.0.1
    │   │   │   └── libc v0.2.174
    │   │   ├── libc v0.2.174
    │   │   └── xattr v1.5.1
    │   │       └── rustix v1.0.8
    │   │           ├── bitflags v2.9.1
    │   │           └── linux-raw-sys v0.9.4
    │   ├── tracing v0.1.41
    │   │   ├── pin-project-lite v0.2.16
    │   │   ├── tracing-attributes v0.1.30 (proc-macro)
    │   │   │   ├── proc-macro2 v1.0.95 (*)
    │   │   │   ├── quote v1.0.40 (*)
    │   │   │   └── syn v2.0.104 (*)
    │   │   └── tracing-core v0.1.34
    │   │       └── once_cell v1.21.3
    │   └── tracing-subscriber v0.3.19
    │       ├── matchers v0.1.0
    │       │   └── regex-automata v0.1.10
    │       │       └── regex-syntax v0.6.29
    │       ├── nu-ansi-term v0.46.0
    │       │   └── overload v0.1.1
    │       ├── once_cell v1.21.3
    │       ├── regex v1.11.1 (*)
    │       ├── sharded-slab v0.1.7
    │       │   └── lazy_static v1.5.0
    │       ├── smallvec v1.15.1
    │       ├── thread_local v1.1.9
    │       │   └── cfg-if v1.0.1
    │       ├── tracing v0.1.41 (*)
    │       ├── tracing-core v0.1.34 (*)
    │       └── tracing-log v0.2.0
    │           ├── log v0.4.27
    │           ├── once_cell v1.21.3
    │           └── tracing-core v0.1.34 (*)
    │   [build-dependencies]
    │   ├── build-helper v0.1.1
    │   │   └── semver v0.6.0
    │   │       └── semver-parser v0.7.0
    │   ├── cc v1.2.30 (*)
    │   ├── curl v0.4.48
    │   │   ├── curl-sys v0.4.82+curl-8.14.1
    │   │   │   ├── libc v0.2.174
    │   │   │   ├── libz-sys v1.1.22
    │   │   │   │   └── libc v0.2.174
    │   │   │   │   [build-dependencies]
    │   │   │   │   ├── cc v1.2.30 (*)
    │   │   │   │   ├── pkg-config v0.3.32
    │   │   │   │   └── vcpkg v0.2.15
    │   │   │   └── openssl-sys v0.9.109
    │   │   │       └── libc v0.2.174
    │   │   │       [build-dependencies]
    │   │   │       ├── cc v1.2.30 (*)
    │   │   │       ├── pkg-config v0.3.32
    │   │   │       └── vcpkg v0.2.15
    │   │   │   [build-dependencies]
    │   │   │   ├── cc v1.2.30 (*)
    │   │   │   └── pkg-config v0.3.32
    │   │   ├── libc v0.2.174
    │   │   ├── openssl-probe v0.1.6
    │   │   ├── openssl-sys v0.9.109 (*)
    │   │   └── socket2 v0.5.10
    │   │       └── libc v0.2.174
    │   ├── tar v0.4.44 (*)
    │   └── xz v0.1.0
    │       └── xz2 v0.1.7
    │           └── lzma-sys v0.1.20
    │               └── libc v0.2.174
    │               [build-dependencies]
    │               ├── cc v1.2.30 (*)
    │               └── pkg-config v0.3.32
    ├── serde v1.0.219 (*)
    └── serde_json v1.0.141 (*)

</details>
<!-- RUST_DEPS_END -->

## Container



