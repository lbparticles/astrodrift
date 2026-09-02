use cuda_core::{launch_kernel_on_stream, CudaContext, CudaModule, DeviceBuffer};
// use crate::tables::build_sphericalcutoff_force_table;
use pyo3::prelude::*;
use shared::{
    Config, Linspace, Method, Model, ModelComponent, ModernFlags, Real, Tolerance, MAX_STATES,
};
use std::array;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::os::raw::{c_double, c_int};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use thiserror::Error;

use crate::state::{InputFrame, InputState, OutputFrame, OutputState};

// cuda-oxide kernel image, embedded at build time from OUT_DIR (copied there
// by build.rs; produced by ./build-cuda-oxide-kernels.sh).
static CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

const BLOCK_SIZE: u32 = 128;
const NF64: usize = 6;

fn grid_size(n: usize, block: u32) -> (u32, u32) {
    let blocks = ((n as u32) + block - 1) / block;
    (blocks, block)
}

pub fn gather_states(
    src: &[f64],
    indices: &[usize],
    n_particles: usize,
    n_divisions: usize,
) -> Vec<f64> {
    // Each state has exactly 6 floats
    const STATE_LEN: usize = 6;

    // The total length of the result array
    let total_len = n_particles * n_divisions * STATE_LEN;
    let mut dst = Vec::with_capacity(total_len);

    for &idx in indices {
        let start = idx * STATE_LEN;
        let end = start + STATE_LEN;
        // Safety: assumes src has at least `end` elements
        dst.extend_from_slice(&src[start..end]);
    }

    dst
}

pub fn gather_states_nested_extended(
    src: &[f64],
    indices: &[Vec<isize>],
    n_particles: usize,
    n_divisions: usize,
) -> Vec<Vec<f64>> {
    const STATE_LEN_IN: usize = 6; // from the source
    const STATE_LEN_OUT: usize = 9; // desired output length per state

    let mut all = Vec::with_capacity(indices.len());

    for particle_indices in indices {
        let mut states = Vec::with_capacity(particle_indices.len() * STATE_LEN_OUT);

        for &i in particle_indices {
            let idx = i as usize;

            // Copy the 6 source floats
            states.extend_from_slice(&src[idx * STATE_LEN_IN..idx * STATE_LEN_IN + STATE_LEN_IN]);

            // Extend with 3 additional values (0.0 placeholders here)
            states.extend_from_slice(&[0.0; STATE_LEN_OUT - STATE_LEN_IN]);
        }

        all.push(states);
    }

    all
}
struct DumpData {
    dim: c_int,
    nt: c_int,
    dt_one: c_double,
    rtol: c_double,
    atol: c_double,
    t: Vec<c_double>,
    yo: Vec<c_double>,
    nargs: c_int,
    args: Vec<c_double>, // may be empty
}

fn parse_dopr54_dump<P: AsRef<Path>>(path: P) -> io::Result<DumpData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut dim: Option<c_int> = None;
    let mut nt: Option<c_int> = None;
    let mut dt_one: Option<c_double> = None;
    let mut rtol: Option<c_double> = None;
    let mut atol: Option<c_double> = None;
    let mut t: Vec<c_double> = Vec::new();
    let mut yo: Vec<c_double> = Vec::new();
    let mut nargs: c_int = 0;
    let mut args: Vec<c_double> = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        let mut parts = line.split_whitespace();
        let key = match parts.next() {
            Some(k) => k,
            None => continue,
        };
        match key {
            "dim" => {
                if let Some(v) = parts.next() {
                    dim = Some(v.parse().unwrap());
                }
            }
            "nt" => {
                if let Some(v) = parts.next() {
                    nt = Some(v.parse().unwrap());
                }
            }
            "dt_one" => {
                if let Some(v) = parts.next() {
                    dt_one = Some(v.parse().unwrap());
                }
            }
            "rtol" => {
                if let Some(v) = parts.next() {
                    rtol = Some(v.parse().unwrap());
                }
            }
            "atol" => {
                if let Some(v) = parts.next() {
                    atol = Some(v.parse().unwrap());
                }
            }
            "t" => {
                for v in parts {
                    t.push(v.parse().unwrap());
                }
            }
            "yo" => {
                for v in parts {
                    yo.push(v.parse().unwrap());
                }
            }
            "nargs" => {
                if let Some(v) = parts.next() {
                    nargs = v.parse().unwrap();
                }
            }
            "args" => {
                for v in parts {
                    args.push(v.parse().unwrap());
                }
            }
            _ => {
                // ignore unknown lines
            }
        }
    }

    Ok(DumpData {
        dim: dim.expect("dim missing in dump"),
        nt: nt.expect("nt missing in dump"),
        dt_one: dt_one.unwrap_or(-9999.99), // match typical galpy default
        rtol: rtol.expect("rtol missing in dump"),
        atol: atol.expect("atol missing in dump"),
        t,
        yo,
        nargs,
        args,
    })
}

#[derive(Debug, Error)]
pub enum GPUDispatchError {
    #[error("CUDA error: {0:?}")]
    Cuda(#[from] cuda_core::DriverError),

    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
}

pub fn launch_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        "dopr54_cpu_port",
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
    )
}

pub fn launch_dop853_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        "dop853_cpu_port",
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
    )
}

fn launch_kernel_named(
    kernel_name: &str,
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
) -> Result<OutputState, GPUDispatchError> {
    // Creating a context and loading the cubin on every call cost ~0.2 s of
    // fixed overhead per integration, which drowned out the actual
    // per-particle scaling. Cache both (the module keeps the context alive).
    static CACHED_CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();
    static CACHED_MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();
    let ctx = match CACHED_CTX.get() {
        Some(ctx) => ctx.clone(),
        None => {
            let ctx: Arc<CudaContext> = CudaContext::new(0)?;
            let _ = CACHED_CTX.set(ctx.clone());
            ctx
        }
    };
    ctx.bind_to_thread()?;
    let module = match CACHED_MODULE.get() {
        Some(module) => module.clone(),
        None => {
            let module = ctx.load_module_from_image(CUBIN)?;
            let _ = CACHED_MODULE.set(module);
            CACHED_MODULE.get().unwrap().clone()
        }
    };
    let stream = ctx.default_stream();

    let times: Vec<Real> = times.unwrap_or_else(|| {
        // Match numpy/galpy linspace semantics: the endpoint is INCLUDED, so
        // the divisor is steps-1 (dividing by `steps` silently compressed the
        // whole time grid by (steps-1)/steps and desynchronised drift's time
        // axis from galpy's).
        let denom = if linspace.steps > 1 {
            (linspace.steps - 1) as Real
        } else {
            1.0
        };
        (0..linspace.steps)
            .map(|i| {
                linspace.start
                    + (linspace.end - linspace.start) * (i as Real) / denom
            })
            .collect()
    });

    // The kernel writes one DIM-length state per particle per output time:
    // state_out is [nt * n * DIM], so size the buffer to match instead of the
    // fixed OUTPUT_LENGTH cap (which caused illegal memory accesses for
    // larger nt*n and made any N>MAX_PARTICLES run impossible).
    let output_len = times.len() * input_state.num_particles * NF64;
    let mut output_state = OutputState { data: vec![0.0; output_len] };

    let kernel = module.load_function(kernel_name)?;
    let dev_state0 = DeviceBuffer::<f64>::from_host(&stream, &input_state.data)?;
    let dev_times = DeviceBuffer::<f64>::from_host(&stream, &times)?;
    let dev_state_out = DeviceBuffer::<f64>::zeroed(&stream, output_state.data.len())?;

    let (grid, block) = grid_size(input_state.num_particles, BLOCK_SIZE);

    // FIXME: hmmmm
    let dt_one_init = -9999.99f64;

    // Kernel signature:
    //   (state0: *const f64, times: *const f64, state_out: *mut f64,
    //    n: usize, nt: usize, rtol: f64, atol: f64, dt_one_init: f64)
    let mut state0_arg = dev_state0.cu_deviceptr() as *const f64;
    let mut times_arg = dev_times.cu_deviceptr() as *const f64;
    let mut state_out_arg = dev_state_out.cu_deviceptr() as *mut f64;
    let mut n_arg = input_state.num_particles;
    let mut nt_arg = times.len();
    let mut rtol_arg = tolerance.rtol;
    let mut atol_arg = tolerance.atol;
    let mut dt_one_arg = dt_one_init;

    let mut params: [*mut std::os::raw::c_void; 8] = [
        (&mut state0_arg as *mut *const f64).cast(),
        (&mut times_arg as *mut *const f64).cast(),
        (&mut state_out_arg as *mut *mut f64).cast(),
        (&mut n_arg as *mut usize).cast(),
        (&mut nt_arg as *mut usize).cast(),
        (&mut rtol_arg as *mut f64).cast(),
        (&mut atol_arg as *mut f64).cast(),
        (&mut dt_one_arg as *mut f64).cast(),
    ];

    unsafe {
        launch_kernel_on_stream(&kernel, (grid, 1, 1), (block, 1, 1), 0, &stream, &mut params)?;
    }
    stream.synchronize()?;

    output_state.data = dev_state_out.to_host_vec(&stream)?;

    Ok(output_state)
    // let mut f = File::create("dopr54_rust_out_gpu_yay.txt")?;
    // let err: c_int = 0;
    // writeln!(f, "err {}", err)?;
    // writeln!(f, "dim {}", dump.dim)?;
    // writeln!(f, "nt {}", dump.nt)?;
    // writeln!(f, "t")?;
    // for ti in &dump.t {
    //     writeln!(f, "{:.16e}", ti)?;
    // }
    // writeln!(f, "states")?;

    // // (step, particle, dim) = ((s * n) + tid) * DIM
    // // here n=1, tid=0, so index = s*DIM + j
    // for step in 0..nt {
    //     write!(f, "step {}", step)?;
    //     let base = step * n * DIM;
    //     for j in 0..dim {
    //         let v = state_out[base + j];
    //         write!(f, " {:.16e}", v)?;
    //     }
    //     writeln!(f)?;
    // }
}

pub fn gpu_dispatch(
    config: Config,
    model: Model,
    input_frame: InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    let mut output_frame = OutputFrame(core::array::from_fn(|_| None));
    for (stage, (model_component_opt, input_state_opt)) in
        model.into_iter().zip(input_frame.into_iter()).enumerate()
    {
        if let (Some(model_component), Some(input_state)) = (model_component_opt, input_state_opt) {
            let out_state = match config.method {
                Method::DOPR54 => launch_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                ),
                Method::DOP853 => launch_dop853_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                ),
            }?;
            output_frame.0[stage] = Some(out_state);
        }
    }

    Ok(output_frame)
}
