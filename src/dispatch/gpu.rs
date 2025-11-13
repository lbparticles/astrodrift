use cust::prelude::*;
use pyo3::prelude::*;
use shared::{PotentialRecipe, StaticInterface};
use crate::tables::build_sphericalcutoff_force_table;
use crate::PyInterface;
use crate::index_helpers::find_last_times_and_indices;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

const BLOCK_SIZE: u32 = 128;
const N_AR: usize = 100000;
const R_MIN: f64 = 1e-4;
const R_MAX: f64 = 100.0;

fn grid_size(n: usize, block: u32) -> (u32, u32) {
    let blocks = ((n as u32) + block - 1) / block;
    (blocks, block)
}

pub fn gpu_dispatch(state_out:&mut Vec<f64>,recipes:Vec<PotentialRecipe>,statics:StaticInterface,_lut:Vec<f64>,config:PyInterface )->PyResult<(f64,f64)>{
    let n = statics.n;
    let ts: Vec<f64> = (0..config.poll_number)
        .map(|i| config.t_end * (i as f64) / (config.poll_number as f64 - 1.))
        .collect();
    let module = py_runtime_err(Module::from_ptx(PTX, &[]))?;
    let stream = py_runtime_err(Stream::new(StreamFlags::DEFAULT, None))?;
    let kernel = py_runtime_err(module.get_function("dopr54_adaptive"))?;
    let mut time_out = vec![0.0f64; config.steps_cap * n];

    // per-particle scalars
    let mut t_host = vec![0.0f64; statics.n];
    let mut dt_host = vec![statics.time_direction * config.dt0; statics.n];
    let mut w_host = vec![0u32; statics.n];
    let done_host = vec![0u8; statics.n];
    let mut err_host = vec![0.0f64; statics.n];
    let gate_index = vec![0_usize;statics.n];
    // device buffers
    let dev_gate = py_runtime_err(DeviceBuffer::<usize>::from_slice(&gate_index))?;
    let dev_state_out = py_runtime_err(DeviceBuffer::<f64>::from_slice(&state_out))?;
    let dev_t = py_runtime_err(DeviceBuffer::<f64>::from_slice(&t_host))?;
    let dev_dt = py_runtime_err(DeviceBuffer::<f64>::from_slice(&dt_host))?;
    let dev_w = py_runtime_err(DeviceBuffer::<u32>::from_slice(&w_host))?;
    let dev_done = py_runtime_err(DeviceBuffer::<u8>::from_slice(&done_host))?;
    let dev_err = py_runtime_err(DeviceBuffer::<f64>::from_slice(&err_host))?;
    let dev_time_out = py_runtime_err(DeviceBuffer::<f64>::zeroed(config.steps_cap * statics.n))?;
    let dev_recipes = py_runtime_err(DeviceBuffer::<PotentialRecipe>::from_slice(&recipes))?;

    // we launch until all threads are done (or we hit capacity)
    let (grid, block) = grid_size(n, BLOCK_SIZE);
    let mut iter = 0usize;
    let max_outer_iters = 200_000usize; // guard against infinite loops; can raise

    let bulge_amp = 0.029994597188218296;
    let bulge_alpha = 1.8;
    let bulge_r1 = 1.0;
    let bulge_rc = 1.9 / 8.0;

    let (ar_table_host, _r_min, _dr) =
        build_sphericalcutoff_force_table(bulge_amp, bulge_alpha, bulge_r1, bulge_rc,N_AR,R_MIN,R_MAX);
    let dev_ar_table = py_runtime_err(DeviceBuffer::from_slice(&ar_table_host))?;
    let _gate_out = vec![0_usize;n];
    let _dev_dt_out = vec![0_f64;n];

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
                    statics,
                    dev_recipes.as_device_ptr(),
                    dev_ar_table.as_device_ptr(),
                )
            ))?;
        }

        py_runtime_err(stream.synchronize())?;
        iter += 1;

        // copy back "done" each iteration. Maybe we collapse this on device or do it less frequently?
        // py_runtime_err(dev_done.copy_to(&mut done_host))?;
        // py_runtime_err(dev_gate.copy_to(&mut gate_out))?;
        // py_runtime_err(dev_dt.copy_to(&mut dev_dt_out))?;
        // py_runtime_err(dev_err.copy_to(&mut err_host))?;
        // eprintln!("{:?}", gate_out);
        // eprintln!("{:?}", dev_dt_out);
        // eprintln!("{:?}", err_host);
        // stop if all done
        let any_active = done_host.iter().any(|&d| d == 0);
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
    py_runtime_err(dev_state_out.copy_to(state_out))?;
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

    let filled_lens: Vec<usize> = w_host
        .iter()
        .map(|&w| (w as usize + 1).min(config.steps_cap)) // accepted steps + initial state
        .collect();
    let (_app_ts0,_indices )= find_last_times_and_indices(
        &time_out,
        &ts,
        statics.n,
        config.steps_cap,
        &filled_lens
    );
    Ok((0.0,0.0))
}

