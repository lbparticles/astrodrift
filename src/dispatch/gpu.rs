use crate::index_helpers::find_last_times_and_indices;
use crate::python::PyConfig;
// use crate::tables::build_sphericalcutoff_force_table;
use cust::prelude::*;
use pyo3::prelude::*;
use shared::{Config, PotentialRecipe};

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

pub fn gpu_dispatch(
    states: Vec<Vec<Vec<f64>>>,
    stages: Vec<Vec<PotentialRecipe>>,
    config: Config,
    py_config: PyConfig,
) -> PyResult<(f64, f64)> {
    let ts: Vec<f64> = (0..config.poll_number)
        .map(|i| config.t_end * (i as f64) / (config.poll_number as f64 - 1.))
        .collect();
    let module = py_runtime_err(Module::from_ptx(PTX, &[]))?;
    let stream = py_runtime_err(Stream::new(StreamFlags::DEFAULT, None))?;
    let kernel = py_runtime_err(module.get_function("dopr54_adaptive"))?;
    let post_kernel = py_runtime_err(module.get_function("post_kernel"))?;
    let coeff_kernel = py_runtime_err(module.get_function("coeff_kernel"))?;

    for (stage, initial_condition) in stages.iter().zip(states.iter()) {
        let n: usize = initial_condition.len();
        let mut state_out = vec![0.0f64; py_config.steps_cap * n * NF64];
        for (i, row) in initial_condition.iter().enumerate() {
            let off0 = (0 * n + i) * NF64;
            state_out[off0..off0 + NF64].copy_from_slice(row);
        }

        let (grid, block) = grid_size(n, BLOCK_SIZE);
        let mut time_out = vec![0.0f64; py_config.steps_cap * n];
        let mut t_host = vec![0.0f64; n];
        let mut dt_host = vec![config.time_direction * py_config.dt0; n];
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
        let dev_time_out = py_runtime_err(DeviceBuffer::<f64>::zeroed(py_config.steps_cap * n))?;

        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("N must be > 0"));
        }
        let mut clamp_recipes: [PotentialRecipe; 10] = [PotentialRecipe::default(); 10];
        let supertable: Vec<f64> = Vec::new();
        let count = stage.len().min(10);
        let dev_supertable = py_runtime_err(DeviceBuffer::from_slice(&supertable))?;
        clamp_recipes[..count].copy_from_slice(&stage[..count]);
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
                    Config{n,..config},
                    clamp_recipes,
                    dev_supertable.as_device_ptr(),
                )
            ))?;
        }

        py_runtime_err(stream.synchronize())?;

        let _gate_out = vec![0_usize; n];
        let _dev_dt_out = vec![0_f64; n];
        // construct_coeff_table(indices,state_out);
        py_runtime_err(dev_state_out.copy_to(&mut state_out))?;
        py_runtime_err(dev_time_out.copy_to(&mut time_out))?;
        py_runtime_err(dev_t.copy_to(&mut t_host))?;
        py_runtime_err(dev_dt.copy_to(&mut dt_host))?;
        py_runtime_err(dev_w.copy_to(&mut w_host))?;
        py_runtime_err(dev_err.copy_to(&mut err_host))?;
        // println!("{:?}",time_out);
        // println!("{:?}",state_out);
        let filled_lens: Vec<usize> = w_host
            .iter()
            .map(|&w| (w as usize + 1).min(py_config.steps_cap)) // accepted steps + initial state
            .collect();
        let (ts0, step, indices) =
            find_last_times_and_indices(&time_out, &ts, n, py_config.steps_cap, &filled_lens);
        // eprintln!("{:?}",time_out);
        eprintln!("{:?}",ts);
        // eprintln!("{:?}",ts0);
        // eprintln!("{:?}",step);
        let flat_indices: Vec<usize> = indices
            .iter()
            .flat_map(|x| x.iter().map(|&x| x as usize))
            .collect();
        // let eq_state = gather_states(&state_out,&flat_indices,n,config.poll_number);
        let post_state: Vec<f64> =
            gather_states_nested_extended(&state_out, &indices, n, config.poll_number)
                .iter()
                .flat_map(|x| x.iter().map(|&x| x as f64))
                .collect();
        let dev_post_state = py_runtime_err(DeviceBuffer::<f64>::from_slice(&post_state))?;
        unsafe {
            py_runtime_err(launch!(
                post_kernel<<<grid, block, 0, stream>>>(
                    dev_post_state.as_device_ptr(),
                    Config{n,..config},
                    clamp_recipes,
                    dev_supertable.as_device_ptr(),
                )
            ))?;
        }
        py_runtime_err(stream.synchronize())?;
        let mut post_state_out = vec![0.0f64; config.poll_number * n * 9];
        py_runtime_err(dev_post_state.copy_to(&mut post_state_out))?;

        let mut coeff_out = vec![0.0f64; (config.poll_number - 1) * n * 18];
        let dev_coeff = py_runtime_err(DeviceBuffer::<f64>::from_slice(&coeff_out))?;

        unsafe {
            py_runtime_err(launch!(
                coeff_kernel<<<grid, block, 0, stream>>>(
                    dev_post_state.as_device_ptr(),
                    dev_coeff.as_device_ptr(),
                    Config{n,..config},
                    clamp_recipes,
                    dev_supertable.as_device_ptr(),
                )
            ))?;
        }
        py_runtime_err(dev_coeff.copy_to(&mut coeff_out))?;
        let w0 = w_host[0] as usize;
        if w0 >= py_config.steps_cap - 1 {
            eprintln!(
                "WARNING: particle 0 hit steps_cap-1; last step may have been overwritten multiple times."
            );
        }
    }

    Ok((0.0, 0.0))
}
