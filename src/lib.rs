use cust::prelude::*;
use libm::pow;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use shared::{LookUpTable, PotentialNames, PotentialRecipe, StaticInterface};
use statrs::function::gamma::{gamma, gamma_lr};
use std::f64::consts::PI;
// use std::fs::File;
// use std::io::Write;


fn find_last_times_for_particles(
    time_out: &[f64],
    ts: &[f64],
    n_particles: usize,
    steps_cap: usize,
) -> Vec<Vec<Option<f64>>> {
    assert_eq!(time_out.len(), n_particles * steps_cap);

    // Results per particle
    let mut results = Vec::with_capacity(n_particles);

    for p in 0..n_particles {
        // Extract this particle’s time series: p(t0), p(t1), …
        let mut particle_times = Vec::with_capacity(steps_cap);
        for t_step in 0..steps_cap {
            particle_times.push(time_out[t_step * n_particles + p]);
        }

        // Now apply the "last value in each ts interval" logic
        let mut particle_result = Vec::with_capacity(ts.len());
        for window in ts.windows(2) {
            let (t_start, t_end) = (window[0], window[1]);
            let found = particle_times
                .iter()
                .copied()
                .filter(|&t| t >= t_start && t < t_end)
                .last();
            particle_result.push(found);
        }

        // For the last ts value (no next boundary)
        particle_result.push(None);
        results.push(particle_result);
    }

    results
}

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
    let blocks = ((n as u32) + block - 1) / block;
    (blocks, block)
}

const N_AR: usize = 10000;
const R_MIN: f64 = 1e-4;
const R_MAX: f64 = 100.0;

pub fn mass(r2: f64, alpha: f64, rc: f64) -> f64 {
    2.0 * PI
        * pow(rc, 3.0 - alpha)
        * gamma(1.5 - 0.5 * alpha)
        * gamma_lr(1.5 - 0.5 * alpha, r2 / (rc * rc))
}

fn build_sphericalcutoff_force_table(
    amp: f64,
    alpha: f64,
    r1: f64,
    rc: f64,
) -> (Vec<f64>, f64, f64) {
    let mut table = Vec::with_capacity(N_AR);
    let dr = (R_MAX - R_MIN) / (N_AR as f64 - 1.0);
    for i in 0..N_AR {
        let r = R_MIN + i as f64 * dr;
        let r2 = r * r;
        let m = amp * pow(r1, alpha) * mass(r2, alpha, rc);
        let ar = -m / r2;
        table.push(ar);
    }
    (table, R_MIN, dr)
}
// fn build_sphericalcutoff_eval_table(amp: f64, alpha: f64, rc: f64) -> (Vec<f64>, f64, f64) {
//     let mut table = Vec::with_capacity(N_AR);
//     let dr = (R_MAX - R_MIN) / (N_AR as f64 - 1.0);
//     for i in 0..N_AR {
//         let r = R_MIN + i as f64 * dr;
//         let ratio = pow(r / rc, 2.);
//         let out = 2.
//             * PI
//             * amp
//             * pow(rc, 3. - alpha)
//             * (1. / rc)
//             * gamma(1. - alpha / 2.)
//             * gamma_lr(1. - alpha / 2., ratio)
//             - gamma(1.5 - alpha / 2.) * gamma_lr(1.5 - alpha / 2., ratio / r);
//         table.push(out);
//     }
//     (table, R_MIN, dr)
// }

/// An extension trait may be cleaner (e.g. let _ctx = cust::quick_init().into_py()?;)
fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn integrate_gpu<'py>(
    py: Python<'py>,
    state0: PyReadonlyArray2<'py, f64>,
    steps_cap: usize,
    t_end: f64,
    dt0: f64,
    atol: Option<f64>,
    rtol: Option<f64>,
    reverse: Option<bool>,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<f64>>)> {
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

    let atol = atol.unwrap(); // no-no. default?
    let rtol = rtol.unwrap();
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
            let off0 = (0 * n + i) * NF64;
            state_out[off0..off0 + NF64].copy_from_slice(src);
        }
    } else {
        // slow generic copy
        for i in 0..n {
            let off0 = (0 * n + i) * NF64;
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

    let statics = StaticInterface {
        t_end: target_t_end,
        n,
        steps_cap,
        atol,
        rtol,
        fac_min: FAC_MIN,
        fac_max: FAC_MAX,
        safety: SAFETY,
        dt_min: DT_MIN,
        dt_max: DT_MAX,
        time_direction,
    };
    // let book = Bookkeeping{
    //     error_out:dev_err.as_device_ptr().as_raw() as *mut f64,
    //     dt:dev_dt.as_device_ptr().as_raw() as *mut f64,
    //     w:dev_w.as_device_ptr().as_raw() as *mut u32,
    //     done:dev_done.as_device_ptr().as_raw() as *mut u8,
    // };
    let recipe = PotentialRecipe {
        potential_id: PotentialNames::Bovy14,
        fparams: [r_min, dr, 0., 0., 0., 0.],
        uparams: [0, 0, 0, 0, 0, 0],
        lut_info: Some(LookUpTable {
            offset: 0.,
            length: N_AR,
        }),
    };
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
                    statics,
                    recipe,
                    dev_ar_table.as_device_ptr(),
                )
            ))?;
        }

        py_runtime_err(stream.synchronize())?;
        iter += 1;

        // copy back "done" each iteration. Maybe we collapse this on device or do it less frequently?
        py_runtime_err(dev_done.copy_to(&mut done_host))?;

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

    py_runtime_err(dev_state_out.copy_to(&mut state_out))?;
    py_runtime_err(dev_time_out.copy_to(&mut time_out))?;
    py_runtime_err(dev_t.copy_to(&mut t_host))?;
    py_runtime_err(dev_dt.copy_to(&mut dt_host))?;
    py_runtime_err(dev_w.copy_to(&mut w_host))?;
    py_runtime_err(dev_err.copy_to(&mut err_host))?;

    let w0 = w_host[0] as usize;
    if w0 >= steps_cap - 1 {
        eprintln!(
            "WARNING: particle 0 hit steps_cap-1; last step may have been overwritten multiple times."
        );
    }
    eprintln!(
        "WARNING: particle 0 hit steps_cap-1; last step may have been overwritten multiple times."
    );

    // println!("Integration finished after {} kernel launches.", iter);

    // a few diagnostics
    let final_timestep = w_host[0] as usize;
    let _final_off = (final_timestep * n + 0) * 6;
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
#[pyfunction]
fn integrate_gpu2<'py>(
    py: Python<'py>,
    state0: PyReadonlyArray2<'py, f64>,
    state1: PyReadonlyArray2<'py, f64>,
    ts0: PyReadonlyArray1<'py, f64>,
    steps_cap: usize,
    t_end: f64,
    dt0: f64,
    atol: Option<f64>,
    rtol: Option<f64>,
    reverse: Option<bool>,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray2<f64>>,Bound<'py, PyArray2<f64>>)> {
    // GIL held for entire function

    let ic = state0.as_array();
    if ic.ndim() != 2 || ic.shape()[1] != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "state0 must be float64 with shape (N, 6)",
        ));
    }
    let ts = ts0.as_array();
    let n: usize = ic.shape()[0];
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("N must be > 0"));
    }

    let atol = atol.unwrap(); // no-no. default?
    let rtol = rtol.unwrap();
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
            let off0 = (0 * n + i) * NF64;
            state_out[off0..off0 + NF64].copy_from_slice(src);
        }
    } else {
        // slow generic copy
        for i in 0..n {
            let off0 = (0 * n + i) * NF64;
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
    let dev_ar_table = py_runtime_err(DeviceBuffer::from_slice(&ar_table_host))?;

    let statics = StaticInterface {
        t_end: target_t_end,
        n,
        steps_cap,
        atol,
        rtol,
        fac_min: FAC_MIN,
        fac_max: FAC_MAX,
        safety: SAFETY,
        dt_min: DT_MIN,
        dt_max: DT_MAX,
        time_direction,
    };
    let recipe = PotentialRecipe {
        potential_id: PotentialNames::Bovy14,
        fparams: [r_min, dr, 0., 0., 0., 0.],
        uparams: [0, 0, 0, 0, 0, 0],
        lut_info: Some(LookUpTable {
            offset: 0.,
            length: N_AR,
        }),
    };
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
                    statics,
                    recipe,
                    dev_ar_table.as_device_ptr(),
                )
            ))?;
        }

        py_runtime_err(stream.synchronize())?;
        iter += 1;

        // copy back "done" each iteration. Maybe we collapse this on device or do it less frequently?
        py_runtime_err(dev_done.copy_to(&mut done_host))?;

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

    py_runtime_err(dev_state_out.copy_to(&mut state_out))?;
    py_runtime_err(dev_time_out.copy_to(&mut time_out))?;
    py_runtime_err(dev_t.copy_to(&mut t_host))?;
    py_runtime_err(dev_dt.copy_to(&mut dt_host))?;
    py_runtime_err(dev_w.copy_to(&mut w_host))?;
    py_runtime_err(dev_err.copy_to(&mut err_host))?;
    
    let w0 = w_host[0] as usize;
    if w0 >= steps_cap - 1 {
        eprintln!(
            "WARNING: particle 0 hit steps_cap-1; last step may have been overwritten multiple times."
        );
    }

    let app_ts0 = find_last_times_for_particles(
        &time_out,
        ts.as_slice().expect("ts must be contiguous"),
        n,
        steps_cap,
    );
    // println!("Integration finished after {} kernel launches.", iter);

    // a few diagnostics
    let final_timestep = w_host[0] as usize;
    let _final_off = (final_timestep * n + 0) * 6;
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
    let input = app_ts0.into_iter()
    .map(|v| {
        v.into_iter()
            .map(|opt| opt.unwrap_or(f64::NAN))
            .collect::<Vec<f64>>()
    })
    .collect::<Vec<Vec<f64>>>();
    // zero-copy wrap from vecs
    let app_ts = PyArray2::from_vec2(py,&input)?;
    let flat_state = PyArray1::from_vec(py, state_out);
    let arr_state = flat_state.reshape([steps_cap, n, 6])?;
    let flat_time = PyArray1::from_vec(py, time_out);
    let arr_time = flat_time.reshape([steps_cap, n])?;

    Ok((arr_state, arr_time,app_ts))
}

#[pymodule]
fn drift_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(integrate_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_gpu2, m)?)?;
    Ok(())
}
