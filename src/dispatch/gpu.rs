use crate::PyInterface;
use crate::index_helpers::find_last_times_and_indices;
// use crate::tables::build_sphericalcutoff_force_table;
use cust::prelude::*;
use pyo3::prelude::*;
use shared::{PotentialRecipe, StaticInterface};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

const BLOCK_SIZE: u32 = 128;
const NF64: usize = 6;

fn grid_size(n: usize, block: u32) -> (u32, u32) {
    let blocks = ((n as u32) + block - 1) / block;
    (blocks, block)
}

pub fn gpu_dispatch(
    states: Vec<Vec<Vec<f64>>>,
    stages: Vec<Vec<PotentialRecipe>>,
    statics: StaticInterface,
    config: PyInterface,
) -> PyResult<(f64, f64)> {

    let ts: Vec<f64> = (0..config.poll_number)
        .map(|i| config.t_end * (i as f64) / (config.poll_number as f64 - 1.))
        .collect();
    let module = py_runtime_err(Module::from_ptx(PTX, &[]))?;
    let stream = py_runtime_err(Stream::new(StreamFlags::DEFAULT, None))?;
    let kernel = py_runtime_err(module.get_function("dopr54_adaptive"))?;

    let mut iter = 0usize;
    let max_outer_iters = 200_000usize; // guard against infinite loops; can raise

    for (stage, initial_condition) in stages.iter().zip(states.iter()) {
        let n: usize = initial_condition.len();
        let mut state_out = vec![0.0f64; config.steps_cap * n * NF64];
        for (i, row) in initial_condition.iter().enumerate() {
            let off0 = (0 * n + i) * NF64;
            state_out[off0..off0 + NF64].copy_from_slice(row);
        }

        let (grid, block) = grid_size(n, BLOCK_SIZE);
        let mut time_out = vec![0.0f64; config.steps_cap * n];
        let mut t_host = vec![0.0f64; n];
        let mut dt_host = vec![statics.time_direction * config.dt0; n];
        let mut w_host = vec![0u32; n];
        let done_host = vec![0u8; n];
        let mut err_host = vec![0.0f64; n];
        let gate_index = vec![0_usize; n];
        // device buffers
        let dev_gate = py_runtime_err(DeviceBuffer::<usize>::from_slice(&gate_index))?;
        let dev_state_out = py_runtime_err(DeviceBuffer::<f64>::from_slice(&state_out))?;
        let dev_t = py_runtime_err(DeviceBuffer::<f64>::from_slice(&t_host))?;
        let dev_dt = py_runtime_err(DeviceBuffer::<f64>::from_slice(&dt_host))?;
        let dev_w = py_runtime_err(DeviceBuffer::<u32>::from_slice(&w_host))?;
        let dev_done = py_runtime_err(DeviceBuffer::<u8>::from_slice(&done_host))?;
        let dev_err = py_runtime_err(DeviceBuffer::<f64>::from_slice(&err_host))?;
        let dev_time_out = py_runtime_err(DeviceBuffer::<f64>::zeroed(config.steps_cap * n))?;
    
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("N must be > 0"));
        }
        let mut clamp_recipes: [PotentialRecipe; 10] = [PotentialRecipe::default();10];
        let supertable: Vec<f64> = Vec::new();
        let count = stage.len().min(10);
        let dev_supertable = py_runtime_err(DeviceBuffer::from_slice(&supertable))?;
        clamp_recipes[..count].copy_from_slice(&stage[..count]);
        loop {
            unsafe {
                py_runtime_err(launch!(
                    kernel<<<grid, block, 0, stream>>>(
                        dev_state_out.as_device_ptr(),
                        dev_time_out.as_device_ptr(),
                        dev_t.as_device_ptr(),
                        dev_err.as_device_ptr(),
                        dev_dt.as_device_ptr(),
                        dev_w.as_device_ptr(),
                        dev_done.as_device_ptr(),
                        dev_gate.as_device_ptr(),
                        StaticInterface{n,..statics},
                        clamp_recipes,
                        dev_supertable.as_device_ptr(),
                    )
                ))?;
            }

            py_runtime_err(stream.synchronize())?;
            iter += 1;

            let any_active = done_host.iter().any(|&d| d == 0);
            if !any_active {
                break;
            }
            if iter >= max_outer_iters {
                // eprintln!(
                //     "Reached max iterations ({}) before all particles finished; stopping.",
                //     max_outer_iters
                // );
                break;
            }
        }
        let _gate_out = vec![0_usize; n];
        let _dev_dt_out = vec![0_f64; n];
        let filled_lens: Vec<usize> = w_host
            .iter()
            .map(|&w| (w as usize + 1).min(config.steps_cap)) // accepted steps + initial state
            .collect();
        let (_app_ts0, indices) =
            find_last_times_and_indices(&time_out, &ts, n, config.steps_cap, &filled_lens);
        // construct_coeff_table(indices,state_out);
        py_runtime_err(dev_state_out.copy_to(&mut state_out))?;
        py_runtime_err(dev_time_out.copy_to(&mut time_out))?;
        py_runtime_err(dev_t.copy_to(&mut t_host))?;
        py_runtime_err(dev_dt.copy_to(&mut dt_host))?;
        py_runtime_err(dev_w.copy_to(&mut w_host))?;
        py_runtime_err(dev_err.copy_to(&mut err_host))?;
        let w0 = w_host[0] as usize;
        if w0 >= config.steps_cap - 1 {
            eprintln!(
                "WARNING: particle 0 hit steps_cap-1; last step may have been overwritten multiple times."
            );
        }
    }


    Ok((0.0, 0.0))
}
