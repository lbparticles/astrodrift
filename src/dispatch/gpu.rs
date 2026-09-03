use cuda_core::{launch_kernel_on_stream, CudaContext, CudaStream, DeviceBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
// use crate::tables::build_sphericalcutoff_force_table;
use shared::{
    Config, Linspace, Method, Model, ModelComponent, ModernFlags, Real, Tolerance, INPUT_STATE_DIM,
    MAX_PARTICLES,
};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::os::raw::{c_double, c_int};
use std::path::Path;
use thiserror::Error;

use crate::state::{InputFrame, InputState, OutputFrame, OutputState};

// Kernel image embedded at build time, selected by feature:
// - default (modern): cuda-oxide cubin from ./build-cuda-oxide-kernels.sh
// - legacy: PTX compiled by cuda_builder/rustc_codegen_nvvm (needs LLVM 7)
// Both are loaded through cuda-core's CudaContext::load_module_from_image:
// cuModuleLoadData accepts PTX text and cubin ELF alike, so the dispatch
// code below is identical for the two flavors.
#[cfg(feature = "legacy")]
static KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
#[cfg(not(feature = "legacy"))]
static KERNEL_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.cubin"));

const BLOCK_SIZE: u32 = 128;
/// Hard on-stack trajectory limit inside the device kernels
/// (`out_steps: [[f64; DIM]; 1024]` in kernels/src/dopr54.rs).
const MAX_KERNEL_STEPS: usize = 1024;

fn grid_size(n: usize, block: u32) -> (u32, u32) {
    let blocks = n.div_ceil(block as usize);
    (blocks as u32, block)
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
    _n_particles: usize,
    _n_divisions: usize,
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
// Diagnostics helper: parses the text dump format galpy's integrators emit
// (used when comparing GPU results against the reference implementation).
// Currently only referenced by debugging code kept commented out below.
#[allow(dead_code)]
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

/// Parses `raw` as a dump field value, mapping malformed text to an
/// `InvalidData` I/O error instead of panicking (failure modes propagate).
#[allow(dead_code)]
fn parse_field<T: std::str::FromStr>(raw: &str) -> io::Result<T> {
    raw.parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("malformed dump field {raw:?}")))
}

/// See [`DumpData`].
#[allow(dead_code)]
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
        let Some(key) = parts.next() else { continue };
        match key {
            "dim" => {
                if let Some(v) = parts.next() {
                    dim = Some(parse_field(v)?);
                }
            }
            "nt" => {
                if let Some(v) = parts.next() {
                    nt = Some(parse_field(v)?);
                }
            }
            "dt_one" => {
                if let Some(v) = parts.next() {
                    dt_one = Some(parse_field(v)?);
                }
            }
            "rtol" => {
                if let Some(v) = parts.next() {
                    rtol = Some(parse_field(v)?);
                }
            }
            "atol" => {
                if let Some(v) = parts.next() {
                    atol = Some(parse_field(v)?);
                }
            }
            "t" => {
                for v in parts {
                    t.push(parse_field(v)?);
                }
            }
            "yo" => {
                for v in parts {
                    yo.push(parse_field(v)?);
                }
            }
            "nargs" => {
                if let Some(v) = parts.next() {
                    nargs = parse_field(v)?;
                }
            }
            "args" => {
                for v in parts {
                    args.push(parse_field(v)?);
                }
            }
            _ => {
                // ignore unknown lines
            }
        }
    }

    let missing = |field: &str| {
        io::Error::new(io::ErrorKind::InvalidData, format!("{field} missing in dump"))
    };
    Ok(DumpData {
        dim: dim.ok_or_else(|| missing("dim"))?,
        nt: nt.ok_or_else(|| missing("nt"))?,
        dt_one: dt_one.unwrap_or(-9999.99), // match typical galpy default
        rtol: rtol.ok_or_else(|| missing("rtol"))?,
        atol: atol.ok_or_else(|| missing("atol"))?,
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

    #[error("too many particles: {requested} requested, {max} supported")]
    TooManyParticles { requested: usize, max: usize },

    #[error("too many output steps: {requested} requested, {max} supported by the kernel")]
    TooManySteps { requested: usize, max: usize },

    #[error("unknown device ordinal {ordinal}: {count} device(s) visible")]
    UnknownDevice { ordinal: usize, count: usize },

    #[error(
        "method {method} is a registered stub: dispatch is wired but the \
         integration loop is not implemented yet (see src/methods/registry.rs)"
    )]
    NotImplemented { method: &'static str },
}

pub fn launch_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
    devices: &[usize],
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        "dopr54_cpu_port",
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
        devices,
    )
}

pub fn launch_dop853_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
    devices: &[usize],
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        "dop853_cpu_port",
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
        devices,
    )
}

/// One device-resident kernel execution: its stream and the buffers that
/// must outlive the in-flight kernel (dropped only after the gather phase
/// synchronizes every stream).
struct ChunkJob {
    stream: Arc<CudaStream>,
    _dev_state0: DeviceBuffer<f64>,
    _dev_times: DeviceBuffer<f64>,
    /// Full trajectory: `nt * count * DIM` floats, layout (step, particle, dim).
    dev_out: DeviceBuffer<f64>,
    /// Slice of `OutputState::data` that receives this chunk's final states.
    final_range: std::ops::Range<usize>,
    /// Offset/length of the last trajectory step inside `dev_out`.
    final_src: std::ops::Range<usize>,
}

/// Per-device context cache: repeated dispatches skip driver re-init.
/// Dispatch is single-threaded; contexts are reference counted so cached
/// handles stay valid for the lifetime of the process.
fn context_for(ordinal: usize) -> Result<Arc<CudaContext>, GPUDispatchError> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    let cache = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(ctx) = guard.get(&ordinal) {
        return Ok(Arc::clone(ctx));
    }
    let ctx = CudaContext::new(ordinal)?;
    guard.insert(ordinal, Arc::clone(&ctx));
    Ok(ctx)
}

/// Splits `n` particles into at most `k` contiguous, balanced chunks
/// (within one particle). Chunk `i` runs on device `devices[i]`.
fn balanced_chunks(n: usize, k: usize) -> Vec<(usize, usize)> {
    let k = k.clamp(1, n);
    let base = n / k;
    let rem = n % k;
    let mut chunks = Vec::with_capacity(k);
    let mut start = 0;
    for i in 0..k {
        let count = base + usize::from(i < rem);
        chunks.push((start, count));
        start += count;
    }
    chunks
}

/// Loads the embedded kernel image on `device` and enqueues one chunk of
/// `count` particles starting at particle `start`. All failures are
/// returned; nothing panics.
#[allow(clippy::too_many_arguments)]
fn enqueue_chunk(
    kernel_name: &str,
    host_state: &[f64],
    start: usize,
    count: usize,
    times: &[f64],
    rtol: Real,
    atol: Real,
    device: usize,
) -> Result<ChunkJob, GPUDispatchError> {
    let ctx = context_for(device)?;
    let module = ctx.load_module_from_image(KERNEL_IMAGE)?;
    let kernel = module.load_function(kernel_name)?;
    let stream: Arc<CudaStream> = ctx.new_stream()?;

    let in_range = start * INPUT_STATE_DIM..(start + count) * INPUT_STATE_DIM;
    let dev_state0 = DeviceBuffer::<f64>::from_host(&stream, &host_state[in_range.clone()])?;
    let dev_times = DeviceBuffer::<f64>::from_host(&stream, times)?;
    // The kernel writes the full trajectory (step, particle, dim):
    // nt * count * DIM floats (kernels/src/dopr54.rs documents the layout).
    let nt = times.len();
    let dev_out = DeviceBuffer::<f64>::zeroed(&stream, nt * count * INPUT_STATE_DIM)?;

    let (grid, block) = grid_size(count, BLOCK_SIZE);

    // FIXME: hmmmm
    let dt_one_init = -9999.99f64;

    // Kernel signature:
    //   (state0: *const f64, times: *const f64, state_out: *mut f64,
    //    n: usize, nt: usize, rtol: f64, atol: f64, dt_one_init: f64)
    let mut state0_arg = dev_state0.cu_deviceptr() as *const f64;
    let mut times_arg = dev_times.cu_deviceptr() as *const f64;
    let mut state_out_arg = dev_out.cu_deviceptr() as *mut f64;
    let mut n_arg = count;
    let mut nt_arg = times.len();
    let mut rtol_arg = rtol;
    let mut atol_arg = atol;
    let mut dt_one_arg = dt_one_init;

    let mut params: [*mut std::os::raw::c_void; 8] = [
        std::ptr::addr_of_mut!(state0_arg).cast(),
        std::ptr::addr_of_mut!(times_arg).cast(),
        std::ptr::addr_of_mut!(state_out_arg).cast(),
        std::ptr::addr_of_mut!(n_arg).cast(),
        std::ptr::addr_of_mut!(nt_arg).cast(),
        std::ptr::addr_of_mut!(rtol_arg).cast(),
        std::ptr::addr_of_mut!(atol_arg).cast(),
        std::ptr::addr_of_mut!(dt_one_arg).cast(),
    ];

    // SAFETY: `kernel` belongs to `module`, loaded in `ctx`, which owns
    // `stream`; every param points to a value of the size/alignment the
    // kernel signature expects, valid until the launch call returns.
    unsafe {
        launch_kernel_on_stream(&kernel, (grid, 1, 1), (block, 1, 1), 0, &stream, &mut params)?;
    }

    Ok(ChunkJob {
        stream,
        _dev_state0: dev_state0,
        _dev_times: dev_times,
        dev_out,
        final_range: in_range,
        final_src: (nt - 1) * count * INPUT_STATE_DIM..nt * count * INPUT_STATE_DIM,
    })
}

fn launch_kernel_named(
    kernel_name: &str,
    _model_component: &ModelComponent,
    input_state: &InputState,
    _flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
    devices: &[usize],
) -> Result<OutputState, GPUDispatchError> {
    let times: Vec<Real> = times.unwrap_or_else(|| {
        (0..linspace.steps)
            .map(|i| {
                linspace.start
                    + (linspace.end - linspace.start) * (i as Real) / (linspace.steps as Real)
            })
            .collect()
    });

    let mut output_state = OutputState::new_zeroed();
    let n_total = input_state.num_particles;
    if n_total == 0 {
        return Ok(output_state); // nothing to integrate; skip the driver round-trip
    }
    // The state buffers are sized by MAX_PARTICLES; launching more than that
    // would read/write out of bounds on the device.
    if n_total > MAX_PARTICLES {
        return Err(GPUDispatchError::TooManyParticles {
            requested: n_total,
            max: MAX_PARTICLES,
        });
    }
    // The kernel stores every step in a fixed on-stack array of 1024 entries.
    if linspace.steps > MAX_KERNEL_STEPS {
        return Err(GPUDispatchError::TooManySteps {
            requested: linspace.steps,
            max: MAX_KERNEL_STEPS,
        });
    }

    // One chunk per device: contiguous particle ranges, balanced within one
    // particle. Particles are non-interacting, so per-particle results are
    // identical to the single-device run.
    let chunks = balanced_chunks(n_total, devices.len());

    // Enqueue phase: uploads serialize (from_host syncs), but the launches
    // are asynchronous and therefore execute concurrently across devices.
    let mut jobs: Vec<ChunkJob> = Vec::with_capacity(chunks.len());
    let mut failure: Option<GPUDispatchError> = None;
    for (chunk_index, &(start, count)) in chunks.iter().enumerate() {
        let device = devices[chunk_index % devices.len()];
        match enqueue_chunk(
            kernel_name,
            &input_state.data,
            start,
            count,
            &times,
            tolerance.rtol,
            tolerance.atol,
            device,
        ) {
            Ok(job) => jobs.push(job),
            // Remember the failure but do not return yet: dropping the
            // already-enqueued chunks while their kernels are in flight
            // would corrupt the context (sticky "illegal memory access").
            Err(err) => {
                failure = Some(err);
                break;
            }
        }
    }

    // Gather phase: synchronize every successfully-enqueued stream BEFORE
    // propagating any failure (buffer lifetime), then download each chunk
    // into its disjoint slice of the output state.
    for job in &jobs {
        if let Err(err) = job.stream.synchronize() {
            failure.get_or_insert(err.into());
        }
    }
    for job in &jobs {
        match job.dev_out.to_host_vec(&job.stream) {
            Ok(trajectory) => output_state.data[job.final_range.clone()]
                .copy_from_slice(&trajectory[job.final_src.clone()]),
            Err(err) => {
                failure.get_or_insert(err.into());
            }
        }
    }
    match failure {
        Some(err) => Err(err),
        None => Ok(output_state),
    }

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
    config: &Config,
    model: &Model,
    input_frame: &InputFrame,
) -> Result<OutputFrame, GPUDispatchError> {
    for (model_component_opt, input_state_opt) in model.into_iter().zip(input_frame) {
        if let (Some(model_component), Some(input_state)) = (model_component_opt, input_state_opt) {
            match config.method {
                Method::DOPR54 => launch_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                    config.devices_slice(),
                ),
                Method::DOP853 => launch_dop853_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                    config.devices_slice(),
                ),
                // Defensive guard: `run_integration` only routes DOPR54/DOP853
                // here. Every other method is a stub mirror and reports as
                // such instead of launching a kernel.
                other => {
                    return Err(GPUDispatchError::NotImplemented {
                        method: other.canonical_name(),
                    });
                }
            }?;
        }
    }

    // Temp
    Ok(OutputFrame(core::array::from_fn(|_| None)))
}
