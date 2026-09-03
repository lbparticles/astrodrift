//! GPU introspection and analytic throughput estimation, exposed to Python.
//!
//! All driver calls are non-panicking: failures are returned as
//! `GPUDispatchError` and converted to Python exceptions exactly once, at
//! the `PyO3` boundary (same policy as `src/dispatch/gpu.rs`).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::{Container, PyConfig};
use crate::dispatch::gpu::GPUDispatchError;
use cuda_core::{sys, IntoResult};
use pyo3::types::PyTuple;

use shared::MAX_PARTICLES;

/// Hard on-stack trajectory limit inside the device kernels (must match
/// `MAX_KERNEL_STEPS` in src/dispatch/gpu.rs).
const MAX_KERNEL_STEPS: usize = 1024;

/// STUB placeholder: 1e9 particles/s per device (calibration pending via
/// tsuchiya-benchmark).
const STUB_PARTICLES_PER_SECOND_PER_DEVICE: f64 = 1_000_000_000.0;

/// Stages per integration step (Dormand-Prince pair sizes).
fn stages_per_step(method: &str) -> Option<usize> {
    match method {
        "DOPR54" => Some(6),
        "DOP853" => Some(12),
        _ => None,
    }
}

/// CUDA cores per streaming multiprocessor, by compute capability, from
/// NVIDIA's occupancy tables. Unknown capabilities fall back to 64 (the
/// conservative pre-Ampere value).
fn cores_per_sm(major: i32, minor: i32) -> u32 {
    // 8.0 (A100) has 64 cores/SM; everything newer we target has 128.
    if (major, minor) == (8, 0) { 64 } else { 128 }
}

// Ownership is irrelevant: the error is only formatted into the exception.
#[allow(clippy::needless_pass_by_value)]
fn to_py(err: GPUDispatchError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn ensure_init() -> Result<(), GPUDispatchError> {
    // SAFETY: cuInit with zero flags; reference counted and idempotent.
    unsafe { cuda_core::init(0) }.map_err(GPUDispatchError::from)
}

fn device_handle(ordinal: usize) -> Result<sys::CUdevice, GPUDispatchError> {
    ensure_init()?;
    let count = device_count_impl()?;
    if ordinal >= count {
        return Err(GPUDispatchError::UnknownDevice { ordinal, count });
    }
    let mut dev: sys::CUdevice = 0;
    // SAFETY: valid out pointer; cuDeviceGet validates the ordinal itself
    // (an out-of-range ordinal returns CUDA_ERROR_INVALID_VALUE, which we
    // propagate instead of panicking).
    unsafe { sys::cuDeviceGet(&mut dev, ordinal as std::os::raw::c_int) }
        .result()
        .map_err(GPUDispatchError::from)?;
    Ok(dev)
}

fn device_attr(
    dev: sys::CUdevice,
    attr: sys::CUdevice_attribute_enum,
) -> Result<i32, GPUDispatchError> {
    let mut value: i32 = 0;
    // SAFETY: valid out pointer and a well-formed attribute selector.
    unsafe { sys::cuDeviceGetAttribute(&mut value, attr, dev) }
        .result()
        .map_err(GPUDispatchError::from)?;
    Ok(value)
}

fn device_count_impl() -> Result<usize, GPUDispatchError> {
    ensure_init()?;
    let mut count: std::os::raw::c_int = 0;
    // SAFETY: valid out pointer.
    unsafe { sys::cuDeviceGetCount(&mut count) }
        .result()
        .map_err(GPUDispatchError::from)?;
    usize::try_from(count).map_err(|_| {
        GPUDispatchError::IO(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "negative device count from the driver",
        ))
    })
}

/// Static device properties, gathered without retaining a context.
struct DeviceSummary {
    ordinal: usize,
    name: String,
    compute_capability: String,
    multiprocessor_count: i32,
    cuda_cores: u32,
    clock_khz: i32,
    memory_clock_khz: i32,
    memory_bus_width_bits: i32,
    total_memory_bytes: usize,
    max_threads_per_block: i32,
    max_threads_per_multiprocessor: i32,
    estimated_peak_gflops: f64,
}

fn device_summary(ordinal: usize) -> Result<DeviceSummary, GPUDispatchError> {
    let dev = device_handle(ordinal)?;

    let mut name_buf = [0 as std::os::raw::c_char; 256];
    // SAFETY: buffer is large enough for CUDA's device-name limit.
    unsafe { sys::cuDeviceGetName(name_buf.as_mut_ptr(), 256, dev) }
        .result()
        .map_err(GPUDispatchError::from)?;
    let name: String = name_buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8 as char)
        .collect();

    let mut total_memory_bytes: usize = 0;
    // SAFETY: valid out pointer and device handle.
    unsafe { sys::cuDeviceTotalMem_v2(&mut total_memory_bytes, dev) }
        .result()
        .map_err(GPUDispatchError::from)?;

    let cc_major = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let cc_minor = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;
    let multiprocessor_count = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )?;
    let clock_khz =
        device_attr(dev, sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?;
    let memory_clock_khz = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    )?;
    let memory_bus_width_bits = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    )?;
    let max_threads_per_block = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    )?;
    let max_threads_per_multiprocessor = device_attr(
        dev,
        sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    )?;

    let cuda_cores = multiprocessor_count as u32 * cores_per_sm(cc_major, cc_minor);
    let estimated_peak_gflops =
        f64::from(cuda_cores) * f64::from(clock_khz) * 1_000.0 * 2.0 / 1_000_000_000.0;

    Ok(DeviceSummary {
        ordinal,
        name,
        compute_capability: format!("{cc_major}.{cc_minor}"),
        multiprocessor_count,
        cuda_cores,
        clock_khz,
        memory_clock_khz,
        memory_bus_width_bits,
        total_memory_bytes,
        max_threads_per_block,
        max_threads_per_multiprocessor,
        estimated_peak_gflops,
    })
}

fn summary_to_py<'a>(py: Python<'a>, s: &DeviceSummary) -> Bound<'a, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("ordinal", s.ordinal).ok();
    dict.set_item("name", &s.name).ok();
    dict.set_item("compute_capability", &s.compute_capability).ok();
    dict.set_item("multiprocessor_count", s.multiprocessor_count).ok();
    dict.set_item("cuda_cores", s.cuda_cores).ok();
    dict.set_item("clock_mhz", s.clock_khz / 1_000).ok();
    dict.set_item("memory_clock_mhz", s.memory_clock_khz / 1_000).ok();
    dict.set_item("memory_bus_width_bits", s.memory_bus_width_bits).ok();
    dict.set_item("total_memory_bytes", s.total_memory_bytes).ok();
    dict.set_item("max_threads_per_block", s.max_threads_per_block).ok();
    dict.set_item("max_threads_per_multiprocessor", s.max_threads_per_multiprocessor).ok();
    dict.set_item("estimated_peak_gflops", s.estimated_peak_gflops).ok();
    dict
}

/// Number of CUDA devices visible to the driver. Zero (CPU-only machine)
/// is a valid answer, not an error.
#[pyfunction]
pub fn device_count(py: Python<'_>) -> PyResult<usize> {
    let _ = py;
    device_count_impl().map_err(to_py)
}

/// Properties of one device (driver-ordinal addressed).
#[pyfunction]
pub fn device_info(py: Python<'_>, ordinal: usize) -> PyResult<Bound<'_, PyDict>> {
    let summary = device_summary(ordinal).map_err(to_py)?;
    Ok(summary_to_py(py, &summary))
}

/// Properties of every visible device.
#[pyfunction]
pub fn list_devices(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let count = device_count_impl().map_err(to_py)?;
    let list = PyList::empty(py);
    for ordinal in 0..count {
        let summary = device_summary(ordinal).map_err(to_py)?;
        list.append(summary_to_py(py, &summary))?;
    }
    Ok(list)
}

/// STUB throughput estimate for a full simulation setup.
///
/// Mirrors `Config.run`: pass the `Config` plus the same containers in the
/// same order. Every piece of context the setup carries (method, variant,
/// steps, tolerance, devices, per-component particle counts) is available
/// here, and the returned dict reports it alongside the estimate.
///
/// The real numbers will come from the `tsuchiya-benchmark` suite
/// (`tsuchiya_benchmark.bench.throughput` writes per-`(backend, N)`
/// throughput tables; see the sibling checkout). Until that calibration
/// is wired in, this returns a fixed placeholder of 1e9 particles/s per
/// device. The signature and return keys are stable; only the numbers
/// (and the "source" tag) will change.
#[pyfunction(signature = (sim, *args))]
#[allow(clippy::needless_pass_by_value)] // pyo3 borrows the pyclass; the tuple is the variadic
pub fn estimate_throughput<'py>(
    py: Python<'py>,
    sim: &PyConfig,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyDict>> {
    let method = format!("{:?}", sim.inner.method);
    let steps = sim.inner.settings.ts.steps;
    let Some(stages) = stages_per_step(&method) else {
        return Err(PyValueError::new_err(format!(
            "method {method:?} has no throughput model: use \"DOPR54\" or \"DOP853\""
        )));
    };

    // Particle workload: only containers that carry an integration state
    // are dispatched on the GPU (background potentials are not).
    let mut components_integrated = 0usize;
    let mut particles_total = 0usize;
    let mut particles_max_component = 0usize;
    for i in 0..args.len() {
        let obj = args.get_item(i)?;
        let container = obj.extract::<pyo3::PyRef<Container>>()?;
        // The launch path integrates `InputState::num_particles` particles
        // (not Container::num_particles, which stores the raw element count).
        if let Some(state) = container.state.as_ref() {
            let n = state.num_particles;
            components_integrated += 1;
            particles_total += n;
            particles_max_component = particles_max_component.max(n);
        }
    }
    if particles_total == 0 {
        return Err(PyValueError::new_err(
            "no containers with particle states: nothing to estimate",
        ));
    }

    let devices = sim.inner.devices_slice();

    let estimated_particles_per_second =
        STUB_PARTICLES_PER_SECOND_PER_DEVICE * devices.len() as f64;
    let estimated_wall_time_seconds = particles_total as f64 / estimated_particles_per_second;

    let dict = PyDict::new(py);
    dict.set_item("method", &method)?;
    dict.set_item("variant", format!("{:?}", sim.inner.variant))?;
    dict.set_item("steps", steps)?;
    dict.set_item("rtol", sim.inner.settings.tolerance.rtol)?;
    dict.set_item("atol", sim.inner.settings.tolerance.atol)?;
    dict.set_item("devices", devices)?;
    dict.set_item("components_integrated", components_integrated)?;
    dict.set_item("particles_total", particles_total)?;
    dict.set_item("particles_max_component", particles_max_component)?;
    dict.set_item("stages_per_step", stages)?;
    dict.set_item("estimated_particles_per_second", estimated_particles_per_second)?;
    dict.set_item("estimated_wall_time_seconds", estimated_wall_time_seconds)?;
    dict.set_item("source", "stub")?;
    dict.set_item("max_particles", MAX_PARTICLES)?;
    dict.set_item("max_steps", MAX_KERNEL_STEPS)?;
    Ok(dict)
}
