use cuda_core::{
    launch_kernel_on_stream, CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream,
    DeviceBuffer, PinnedHostBuffer,
};
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

/// TEMPORARY: cap on the device output buffer per kernel launch. Inputs
/// larger than one chunk are integrated in sequential launches and the
/// outputs concatenated, so N is no longer bounded by VRAM. Replace with a
/// persistent/resizable buffer design later. Override with
/// DRIFT_MAX_LAUNCH_BYTES (bytes of f64 outputs per launch).
const MAX_LAUNCH_OUTPUT_BYTES: usize = 1 << 31;

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

/// Potential specification for recipe-driven launches (previous-method
/// layout). `fparams[0]` = bulge table r_min, `fparams[1]` = dr,
/// `uparams[0]` = supertable element offset, `uparams[1]` = n_ar.
pub struct PotSpec {
    pub fparams: [f64; 6],
    pub uparams: [usize; 6],
    pub supertable: Vec<Real>,
    /// MW2014 + Plummer-stack (pot_type 2): quintic-origin coefficients for
    /// the perturber stack (18 doubles per (particle, division)).
    pub annulus: Option<AnnulusSpec>,
}

/// Annulus perturber-stack specification for the interpolated simulation.
pub struct AnnulusSpec {
    pub coeffs: Vec<Real>,
    pub n_gmc: usize,
    pub division: usize,
    pub final_time: Real,
    pub plummer_amp: Real,
    pub plummer_b: Real,
}

/// Per-slot resources for the double-buffered chunk pipeline.
struct SlotResources {
    dev_in: DeviceBuffer<f64>,
    dev_out: DeviceBuffer<f64>,
    pinned_out: PinnedHostBuffer<f64>,
    ev_kernel: CudaEvent,
    ev_copy: CudaEvent,
}

/// Persistent pipeline resources (streams, slot buffers, events). Pinned
/// allocations are expensive (~1s per GB), so slots are reused across calls
/// and only rebuilt when the required shape (nt, chunk capacity) changes.
struct PipelineCache {
    nt_len: usize,
    chunk_n_max: usize,
    compute: Arc<CudaStream>,
    copy: Arc<CudaStream>,
    slots: Vec<SlotResources>,
}

fn pipeline_slots(
    ctx: &Arc<CudaContext>,
    nt_len: usize,
    chunk_n_max: usize,
) -> std::sync::MutexGuard<'static, PipelineCache> {
    static PIPELINE: OnceLock<std::sync::Mutex<PipelineCache>> = OnceLock::new();
    let mutex = PIPELINE.get_or_init(|| {
        std::sync::Mutex::new(PipelineCache {
            nt_len: 0,
            chunk_n_max: 0,
            compute: ctx.new_stream().expect("compute stream"),
            copy: ctx.new_stream().expect("copy stream"),
            slots: Vec::new(),
        })
    });
    let mut cache = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if cache.nt_len != nt_len || cache.chunk_n_max != chunk_n_max {
        // Free the previous shape's buffers BEFORE allocating new ones: the
        // buffers are sized to the per-launch output budget (up to 2 GiB per
        // slot), so allocating the new shape while still holding the old one
        // transiently needed 2x and OOMed 8 GB cards when a process used two
        // different (nt, n) shapes (e.g. probes then a real integration).
        cache.slots = Vec::new();
        // SAFETY: dev_in is filled by copy_from_host before the kernel reads
        // it, and dev_out is fully overwritten by the kernel (every (step,
        // tid) output slot is written directly to global memory), so neither
        // buffer needs a memset. The pinned host buffers are only read on
        // the host after their copy event has synchronized.
        let slots = (0..2)
            .map(|_| SlotResources {
                dev_in: unsafe {
                    DeviceBuffer::uninitialized_async(&cache.compute, chunk_n_max * NF64)
                        .expect("device input buffer")
                },
                dev_out: unsafe {
                    DeviceBuffer::uninitialized_async(&cache.compute, nt_len * chunk_n_max * NF64)
                        .expect("device output buffer")
                },
                pinned_out: PinnedHostBuffer::zeroed(ctx, nt_len * chunk_n_max * NF64)
                    .expect("pinned output buffer"),
                ev_kernel: ctx.new_event(None).expect("kernel event"),
                ev_copy: ctx.new_event(None).expect("copy event"),
            })
            .collect();
        cache.nt_len = nt_len;
        cache.chunk_n_max = chunk_n_max;
        cache.slots = slots;
    }
    cache
}

/// Interleave one chunk's result planes (time-major over the chunk's own
/// particles, straight out of the pinned D2H buffer) into the global
/// time-major output layout across all n_total particles.
fn assemble_chunk(
    dst: &mut [Real],
    src: &PinnedHostBuffer<Real>,
    nt: usize,
    n_total: usize,
    i0: usize,
    n_chunk: usize,
) {
    let len = nt * n_chunk * NF64;
    let src = &src.as_slice()[..len];
    for t in 0..nt {
        let s = t * n_chunk * NF64;
        let d = ((t * n_total) + i0) * NF64;
        dst[d..d + n_chunk * NF64].copy_from_slice(&src[s..s + n_chunk * NF64]);
    }
}

pub fn launch_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
    pot: Option<&PotSpec>,
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        "dopr54_cpu_port",
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
        pot,
    )
}

pub fn launch_dop853_kernel(
    model_component: &ModelComponent,
    input_state: &InputState,
    flags: ModernFlags,
    tolerance: Tolerance,
    linspace: Linspace,
    times: Option<Vec<Real>>,
    pot: Option<&PotSpec>,
) -> Result<OutputState, GPUDispatchError> {
    launch_kernel_named(
        "dop853_cpu_port",
        model_component,
        input_state,
        flags,
        tolerance,
        linspace,
        times,
        pot,
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
    pot: Option<&PotSpec>,
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
    // TEMPORARY: integrate inputs larger than one launch's output budget in
    // sequential chunked launches, concatenating outputs in particle order
    // (per-particle results are identical to an unchunked launch).
    // Loading the kernel handle is cheap but not free; cache it alongside the
    // module (CudaFunction is Send + Sync). One slot per kernel: dopr54 and
    // dop853 both live in the cubin.
    static CACHED_FUNC: OnceLock<CudaFunction> = OnceLock::new();
    static CACHED_FUNC_853: OnceLock<CudaFunction> = OnceLock::new();
    let kernel = match kernel_name {
        "dop853_cpu_port" => match CACHED_FUNC_853.get() {
            Some(kernel) => kernel,
            None => {
                let kernel = module.load_function(kernel_name)?;
                let _ = CACHED_FUNC_853.set(kernel);
                CACHED_FUNC_853.get().unwrap()
            }
        },
        _ => match CACHED_FUNC.get() {
            Some(kernel) => kernel,
            None => {
                let kernel = module.load_function(kernel_name)?;
                let _ = CACHED_FUNC.set(kernel);
                CACHED_FUNC.get().unwrap()
            }
        },
    };
    let nt_len = times.len();
    let n_total = input_state.num_particles;
    let budget: usize = std::env::var("DRIFT_MAX_LAUNCH_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MAX_LAUNCH_OUTPUT_BYTES);
    let out_bytes_per_particle = nt_len * NF64 * std::mem::size_of::<Real>();
    let mut chunk_n_max = if out_bytes_per_particle > 0 {
        (budget / out_bytes_per_particle).max(1)
    } else {
        1
    };
    // Never size the buffers for more particles than the input actually has:
    // the budget formula alone allocates the full 2 GiB worth of output even
    // for n=1 probe launches (two pipeline slots => ~4.5 GB for a single
    // particle at nt=1001), which OOMs small GPUs and crowds out concurrent
    // processes. Chunking only splits launches; per-particle results are
    // identical, so capping at n_total is behavior-preserving.
    if n_total > 0 {
        chunk_n_max = chunk_n_max.min(n_total);
    }

    let mut output_state = OutputState {
        data: vec![0.0; nt_len * n_total * NF64],
    };
    if nt_len == 0 || n_total == 0 {
        return Ok(output_state);
    }

    // Two-stream double-buffered pipeline: chunk k's kernel runs on the
    // compute stream while chunk k-1's results copy back on the copy stream
    // (D2H into page-locked buffers), overlapping transfer with integration.
    // Streams, slot buffers and events persist across calls (see
    // pipeline_slots) because pinned allocations are expensive.
    let mut pipeline = pipeline_slots(&ctx, nt_len, chunk_n_max);
    let compute = pipeline.compute.clone();
    let copy = pipeline.copy.clone();

    let dev_times = DeviceBuffer::<f64>::from_host(&compute, &times)?;

    // FIXME: hmmmm
    let dt_one_init = -9999.99f64;

    // Potential specification: flattened MW2014 params + bulge force LUT
    // ("supertable", previous-method layout). Uploaded per call; it is tiny
    // (a few thousand doubles at most). None => Kepler fast path.
    // supertable layout: [bulge LUT (n_ar) | quintic origin coefficients]
    let (pot_type, mw_r_min, mw_dr, mw_n_ar, mw_lut_offset, ann_spec, ann_coeff_offset, supertable) =
        match pot {
            Some(spec) => match &spec.annulus {
                None => (
                    1_i32,
                    spec.fparams[0],
                    spec.fparams[1],
                    spec.uparams[1],
                    spec.uparams[0],
                    None,
                    0usize,
                    Some(DeviceBuffer::<f64>::from_host(&compute, &spec.supertable)?),
                ),
                Some(ann) => {
                    // supertable = [bulge LUT | quintic origin coefficients]
                    let mut supertable = spec.supertable.clone();
                    supertable.extend_from_slice(&ann.coeffs);
                    (
                        2_i32,
                        spec.fparams[0],
                        spec.fparams[1],
                        spec.uparams[1],
                        spec.uparams[0],
                        Some(ann),
                        spec.supertable.len(),
                        Some(DeviceBuffer::<f64>::from_host(&compute, &supertable)?),
                    )
                }
            },
            None => (0_i32, 0.0, 0.0, 0usize, 0usize, None, 0usize, None),
        };

    let mut i0: usize = 0;
    let mut chunk: usize = 0;
    // (slot, particle offset, chunk size) for chunks whose D2H may still be
    // in flight.
    let mut inflight: Vec<(usize, usize, usize)> = Vec::new();
    // Persistent device input buffers are sized for the max chunk, but the
    // last chunk can be smaller; copy_from_host requires an exact length, so
    // chunks are staged through this padded host buffer (the kernel only
    // reads tid < n_chunk, so padding is never touched).
    let mut staging_in: Vec<Real> = vec![0.0; chunk_n_max * NF64];

    while i0 < n_total {
        let slot = chunk % 2;
        let n_chunk = (n_total - i0).min(chunk_n_max);
        let slot_res = &mut pipeline.slots[slot];

        // Slot reuse from two iterations back: wait for its D2H, hand the
        // finished data to the output, and order the compute stream behind
        // the copy before refilling the device buffers.
        if chunk >= 2 {
            slot_res.ev_copy.synchronize()?;
            let (slot, prev_i0, prev_n) = inflight[chunk - 2];
            assemble_chunk(&mut output_state.data, &slot_res.pinned_out, nt_len, n_total, prev_i0, prev_n);
            compute.wait(&slot_res.ev_copy)?;
        }

        staging_in[..n_chunk * NF64]
            .copy_from_slice(&input_state.data[i0 * NF64..(i0 + n_chunk) * NF64]);
        staging_in[n_chunk * NF64..].fill(0.0);
        slot_res
            .dev_in
            .copy_from_host(&compute, &staging_in)?;

        let (grid, block) = grid_size(n_chunk, BLOCK_SIZE);

        // Kernel signature:
        //   (state0: *const f64, times: *const f64, state_out: *mut f64,
        //    n: usize, nt: usize, rtol: f64, atol: f64, dt_one_init: f64)
        let mut state0_arg = slot_res.dev_in.cu_deviceptr() as *const f64;
        let mut times_arg = dev_times.cu_deviceptr() as *const f64;
        let mut state_out_arg = slot_res.dev_out.cu_deviceptr() as *mut f64;
        let mut n_arg = n_chunk;
        let mut nt_arg = nt_len;
        let mut rtol_arg = tolerance.rtol;
        let mut atol_arg = tolerance.atol;
        let mut dt_one_arg = dt_one_init;

        let mut supertable_arg = match &supertable {
            Some(buf) => buf.cu_deviceptr() as *const f64,
            None => times_arg,
        };
        let mut pot_type_arg = pot_type;
        let mut mw_r_min_arg = mw_r_min;
        let mut mw_dr_arg = mw_dr;
        let mut mw_n_ar_arg = mw_n_ar;
        let mut mw_lut_offset_arg = mw_lut_offset;
        let (mut ann_n_gmc_arg, mut ann_division_arg, mut ann_final_time_arg, mut ann_plummer_amp_arg, mut ann_plummer_b_arg, mut ann_coeff_offset_arg) = match ann_spec {
            Some(ann) => (ann.n_gmc, ann.division, ann.final_time, ann.plummer_amp, ann.plummer_b, ann_coeff_offset),
            None => (0usize, 0usize, 0.0, 0.0, 0.0, mw_n_ar),
        };

        let mut params: [*mut std::os::raw::c_void; 20] = [
            (&mut supertable_arg as *mut *const f64).cast(),
            (&mut state0_arg as *mut *const f64).cast(),
            (&mut times_arg as *mut *const f64).cast(),
            (&mut state_out_arg as *mut *mut f64).cast(),
            (&mut n_arg as *mut usize).cast(),
            (&mut nt_arg as *mut usize).cast(),
            (&mut rtol_arg as *mut f64).cast(),
            (&mut atol_arg as *mut f64).cast(),
            (&mut dt_one_arg as *mut f64).cast(),
            (&mut pot_type_arg as *mut i32).cast(),
            (&mut mw_r_min_arg as *mut f64).cast(),
            (&mut mw_dr_arg as *mut f64).cast(),
            (&mut mw_n_ar_arg as *mut usize).cast(),
            (&mut mw_lut_offset_arg as *mut usize).cast(),
            (&mut ann_n_gmc_arg as *mut usize).cast(),
            (&mut ann_division_arg as *mut usize).cast(),
            (&mut ann_final_time_arg as *mut f64).cast(),
            (&mut ann_plummer_amp_arg as *mut f64).cast(),
            (&mut ann_plummer_b_arg as *mut f64).cast(),
            (&mut ann_coeff_offset_arg as *mut usize).cast(),
        ];
        let params: &mut [*mut std::os::raw::c_void] = &mut params[..];

        unsafe {
            launch_kernel_on_stream(&kernel, (grid, 1, 1), (block, 1, 1), 0, &compute, params)?;
        }
        slot_res.ev_kernel.record(&compute)?;

        copy.wait(&slot_res.ev_kernel)?;
        unsafe {
            slot_res
                .dev_out
                .copy_to_pinned_host_async(&copy, &mut slot_res.pinned_out)?;
        }
        slot_res.ev_copy.record(&copy)?;

        inflight.push((slot, i0, n_chunk));
        i0 += n_chunk;
        chunk += 1;
    }

    // Drain the at most two chunks still in flight.
    for k in chunk.saturating_sub(2)..chunk {
        let (slot, i0, n_chunk) = inflight[k];
        let slot_res = &mut pipeline.slots[slot];
        slot_res.ev_copy.synchronize()?;
        assemble_chunk(&mut output_state.data, &slot_res.pinned_out, nt_len, n_total, i0, n_chunk);
    }

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
    pot: Option<&PotSpec>,
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
                    pot,
                ),
                Method::DOP853 => launch_dop853_kernel(
                    model_component,
                    input_state,
                    config.flags,
                    config.settings.tolerance,
                    config.settings.ts,
                    None,
                    pot,
                ),
            }?;
            output_frame.0[stage] = Some(out_state);
        }
    }

    Ok(output_frame)
}
