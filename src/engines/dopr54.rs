use pyo3::prelude::*;
use cust::prelude::*;
use shared::Potential;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const NF64: usize = 7;
// FIXME: double check in MPA impl
const FAC_MIN: f64 = 0.33;
const FAC_MAX: f64 = 6.0;
const SAFETY: f64 = 0.9;
const DT_MIN: f64 = 1.0e-12;
const DT_MAX: f64 = 0.25;

const BLOCK_SIZE: u32 = 128;

fn grid_size(n: usize, block: u32) -> (u32, u32) {
    let blocks = (n as u32).div_ceil(block);
    (blocks, block)
}

/// An extension trait may be cleaner (e.g. let _ctx = cust::quick_init().into_py()?;)
fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}


#[pyclass]
#[derive(Clone)]
pub struct Bounds {
    steps_cap: Option<usize>,
    atol: Option<f64>,
    rtol: Option<f64>,
}

const MAX_PARTICLE = 10_000;

pub fn dopr54_adaptive<T>(
    potential: Potential<T>,
    state: [[f64;6],MAX_PARTICLES],
    t_end: f64,
    dt0: f64,
    bounds: Option<Bounds>,
    reverse: Option<bool>,
) ->  {
    let n: usize = ic.shape()[0];
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("N must be > 0"));
    }

    let atol = bounds.atol.unwrap(); // no-no. default?
    let rtol = bounds.rtol.unwrap();
    let steps_cap = bounds.steps_cap;
    let reverse = reverse.unwrap_or(false);
    let time_direction: f64 = if reverse { -1.0 } else { 1.0 };

    // galpy-style: reverse integrate to -t_end from t=0
    let target_t_end = if reverse { -t_end } else { t_end };

    // CUDA setup
    let _ctx = py_runtime_err(cust::quick_init())?;
    let module = py_runtime_err(Module::from_ptx(PTX, &[]))?;
    let stream = py_runtime_err(Stream::new(StreamFlags::DEFAULT, None))?;

    let kernel1 = py_runtime_err(module.get_function("dopr54_adaptive"))?;
    let mut state_out = vec![0.0f64; steps_cap * n * NF64];

    // copy in initial conditions
    //FIXME: copy in 6state into 1-6 leaving 0 for time
    if let Some(slice) = ic.as_slice() {
        // contiguous fast path
        for i in 0..n {
            let src = &slice[i * NF64..i * NF64 + NF64];
            let off0 = i * NF64;
            // let off0 = (0 * n + i) * NF64;
            state_out[off0..off0 + NF64].copy_from_slice(src);
        }
    } else {
        // slow generic copy
        for i in 0..n {
            let off0 = i * NF64;
            // let off0 = (0 * n + i) * NF64;
            let row = ic.row(i);
            state_out[off0..off0 + NF64].copy_from_slice(row.as_slice().unwrap());
        }
    }

    // per-particle scalars
    let mut dt_host = vec![time_direction * dt0; n];
    let mut w_host = vec![0u32; n];
    let mut done_host = vec![0u8; n];
    let mut err_host = vec![0.0f64; n];

    // device buffers
    let dev_state_out = py_runtime_err(DeviceBuffer::<f64>::from_slice(&state_out))?;
    let dev_t = py_runtime_err(DeviceBuffer::<f64>::from_slice(&t_host))?;
    let dev_dt = py_runtime_err(DeviceBuffer::<f64>::from_slice(&dt_host))?;
    let dev_w = py_runtime_err(DeviceBuffer::<u32>::from_slice(&w_host))?;
    let dev_done = py_runtime_err(DeviceBuffer::<u8>::from_slice(&done_host))?;
    let dev_err = py_runtime_err(DeviceBuffer::<f64>::from_slice(&err_host))?;
    let dev_time_out = py_runtime_err(DeviceBuffer::<f64>::zeroed(steps_cap * n))?;

    // we launch until all threads are done (or we hit capacity)
    let (grid, block) = grid_size(n, BLOCK_SIZE);
    let mut iter = 0usize;
    let max_outer_iters = 200_000usize; // guard against infinite loops; can raise

    let dev_ar_table = py_runtime_err(DeviceBuffer::from_slice(&ar_table_host))?;

    loop {
        unsafe {
            py_runtime_err(launch!(
                kernel1<<<grid, block, 0, stream>>>(
                    dev_state_out.as_device_ptr(),
                    n,
                    steps_cap,
                    dev_t.as_device_ptr(),
                    dev_dt.as_device_ptr(),
                    dev_w.as_device_ptr(),
                    dev_done.as_device_ptr(),
                    target_t_end,
                    atol,
                    rtol,
                    FAC_MIN,
                    FAC_MAX,
                    SAFETY,
                    DT_MIN,
                    DT_MAX,
                    dev_err.as_device_ptr(),
                    dev_ar_table.as_device_ptr(),
                    r_min,
                    dr,
                    crate::tables::sphwcutoff::N_AR as u32,
                    time_direction
                )
            ))?;
        }

        py_runtime_err(stream.synchronize())?;
        iter += 1;

        // copy back "done" each iteration. Maybe we collapse this on device or do it less frequently?
        py_runtime_err(dev_done.copy_to(&mut done_host))?;

        // stop if all done
        let any_active = done_host.contains(&0);
        if !any_active {
            break;
        }

        // stop if cap reached
        if iter >= max_outer_iters {
            // eprintln!(
            //     "Reached max iterations ({}) before all particles finished; stopping.",
            //     max_outer_iters
            // );
            break;
        }
    }

    py_runtime_err(dev_state_out.copy_to(&mut state_out))?;
    py_runtime_err(dev_t.copy_to(&mut t_host))?;
    py_runtime_err(dev_dt.copy_to(&mut dt_host))?;
    py_runtime_err(dev_w.copy_to(&mut w_host))?;
    py_runtime_err(dev_err.copy_to(&mut err_host))?;

    Ok(state_out)
}

