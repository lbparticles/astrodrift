use cust::prelude::*;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use crate::tables::{build_sphericalcutoff_force_table};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
const NF64: usize = 6;
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

type EngineOutput<'a> = PyResult<(Bound<'a, PyArray3<f64>>, Bound<'a, PyArray2<f64>>)>;

#[pyclass]
#[derive(Clone)]
pub struct Bounds {
    steps_cap: usize,
    atol: Option<f64>,
    rtol: Option<f64>,
}

#[pyfunction]
pub fn integrate_gpu<'py>(
    py: Python<'py>,
    state0: PyReadonlyArray2<'py, f64>,
    t_end: f64,
    dt0: f64,
    bounds: Bounds,
    reverse: Option<bool>,
) -> EngineOutput<'py> {
    // GIL held for entire function

    let ic = state0.as_array();
    if ic.ndim() != 2 || ic.shape()[1] != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "state0 must be float64 with shape (N, 6)",
        ));
    }
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

    let kernel = py_runtime_err(module.get_function("dopr54_adaptive"))?;
    // keeping these separate for now but vec7 would be ideal
    let mut state_out = vec![0.0f64; steps_cap * n * NF64];
    let mut time_out = vec![0.0f64; steps_cap * n];

    // copy in initial conditions
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
    let mut t_host = vec![0.0f64; n];
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

    // create table
    let bulge_amp = 0.029994597188218296;
    let bulge_alpha = 1.8;
    let bulge_r1 = 1.0;
    let bulge_rc = 1.9 / 8.0;

    let (ar_table_host, r_min, dr) =
        build_sphericalcutoff_force_table(bulge_amp, bulge_alpha, bulge_r1, bulge_rc);
    // let (ar_table_eval_host, r_min, dr) =
    //     build_sphericalcutoff_eval_table(bulge_amp, bulge_alpha, bulge_rc);
    let dev_ar_table = py_runtime_err(DeviceBuffer::from_slice(&ar_table_host))?;

    loop {
        unsafe {
            py_runtime_err(launch!(
                kernel<<<grid, block, 0, stream>>>(
                    dev_state_out.as_device_ptr(),
                    dev_time_out.as_device_ptr(),
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
    py_runtime_err(dev_time_out.copy_to(&mut time_out))?;
    py_runtime_err(dev_t.copy_to(&mut t_host))?;
    py_runtime_err(dev_dt.copy_to(&mut dt_host))?;
    py_runtime_err(dev_w.copy_to(&mut w_host))?;
    py_runtime_err(dev_err.copy_to(&mut err_host))?;

    let w0 = w_host[0] as usize;
    if w0 >= steps_cap - 1 {
        // eprintln!(
        //     "WARNING: particle 0 hit steps_cap-1; last step may have been overwritten multiple times."
        // );
    }

    // println!("Integration finished after {} kernel launches.", iter);

    // a few diagnostics
    let final_timestep = w_host[0] as usize;
    let _final_off = (final_timestep * n) * 6;
    // let _final_off = (final_timestep * n + 0) * 6;
    // println!(
    //     "Particle 0 finished at t = {:.12}, timestep = {}",
    //     t_host[0], final_timestep
    // );
    // println!(
    //     "Final state (x,y,z) = ({:.12}, {:.12}, {:.12})",
    //     state_out[final_off + 0],
    //     state_out[final_off + 1],
    //     state_out[final_off + 2],
    // );
    // println!(
    //     "Last normalized RK error (particle 0) = {:.3e}",
    //     err_host[0]
    // );

    let w0 = w_host[0] as usize;
    let _traj_len = (w0 + 1).min(steps_cap);
    // let mut file = File::create("particle0_steps_with_time.csv")?;
    // writeln!(file, "timestep,time,x,y,z,vx,vy,vz")?;

    // for s in 0..traj_len {
    //     let off = (s * n + 0) * 6;
    //     let time = time_out[s * n + 0];
    //     let x = state_out[off + 0];
    //     let y = state_out[off + 1];
    //     let z = state_out[off + 2];
    //     let vx = state_out[off + 3];
    //     let vy = state_out[off + 4];
    //     let vz = state_out[off + 5];
    //     // writeln!(
    //     //     file,
    //     //     "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
    //     //     s, time, x, y, z, vx, vy, vz
    //     // )?;
    // }
    // println!("Wrote {} rows to particle0_steps_with_time.csv", traj_len);

    let accepts0 = w_host[0] as usize; // steps advanced for particle 0
    let _accept_rate = if iter > 0 {
        100.0 * (accepts0 as f64) / (iter as f64)
    } else {
        0.0
    };
    // println!("Kernel launches (outer iterations): {}", iter);
    // println!("Particle 0 accepted steps: {}", accepts0);
    // println!("Approx overall accept rate (p0): {:.2}%", accept_rate);
    // println!(
    //     "Final t[0] = {:.12} (target T_END = {:.12})",
    //     t_host[0], target_t_end
    // );

    // TODO: How to compute total energy when we have diverging times per particle?

    // zero-copy wrap from vecs
    let flat_state = PyArray1::from_vec(py, state_out);
    let arr_state = flat_state.reshape([steps_cap, n, 6])?;
    let flat_time = PyArray1::from_vec(py, time_out);
    let arr_time = flat_time.reshape([steps_cap, n])?;

    Ok((arr_state, arr_time))
}

#[pymodule]
fn drift_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(integrate_gpu, m)?)?;
    Ok(())
}
