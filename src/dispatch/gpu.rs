use cust::error::CudaError;
// use crate::tables::build_sphericalcutoff_force_table;
use cust::prelude::*;
use pyo3::prelude::*;
use shared::{Config, Meal, MAX_STATES};
use std::array;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::os::raw::{c_double, c_int};
use std::path::Path;
use thiserror::Error;

use crate::state::{InputFrame, OutputFrame, OutputState};

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

const DIM: usize = 6;

#[derive(Debug, Error)]
pub enum GPUDispatchError {
    #[error("CUDA error: {0:?}")]
    Cuda(#[from] CudaError),

    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
}

pub fn gpu_dispatch2(
    // states: Vec<Vec<Vec<f64>>>,
    // stages: Vec<Vec<PotentialRecipe>>,
    // config: Config
// ) -> PyResult<(f64, f64)> {
    config: Config, 
    recipes: Meal, 
    arrays: InputFrame
) -> Result<OutputFrame, GPUDispatchError> {

    let _ctx = cust::quick_init()?;

    let dump = parse_dopr54_dump("dopr54_init_dump.txt").expect("msg");
    let dim = dump.dim as usize;
    let nt  = dump.nt as usize;

    assert_eq!(dim, DIM, "dump dim ({}) != DIM ({})", dim, DIM);
    assert_eq!(dump.t.len(), nt, "t length != nt");
    assert_eq!(dump.yo.len(), dim, "yo length != dim");

    let n: usize = 1;
    let mut state0 = vec![0.0_f64; n * DIM];
    state0[..DIM].copy_from_slice(&dump.yo[..DIM]);

    let times = dump.t.clone();

    let mut state_out = vec![0.0_f64; nt * n * DIM];

    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    let kernel = module.get_function("dopr54_cpu_port")?;
    let dev_state0 = DeviceBuffer::<f64>::from_slice(&state0)?;
    let dev_times = DeviceBuffer::<f64>::from_slice(&times)?;
    let dev_state_out = DeviceBuffer::<f64>::from_slice(&state_out)?;

    let (grid, block) = grid_size(n, BLOCK_SIZE);

    let rtol = dump.rtol as f64;
    let atol = dump.atol as f64;
    let dt_one_init = dump.dt_one as f64;

    unsafe {
        launch!(
            kernel<<<grid, block, 0, stream>>>(
                dev_state0.as_device_ptr(),
                dev_times.as_device_ptr(),
                dev_state_out.as_device_ptr(),
                n,
                nt,
                rtol,
                atol,
                dt_one_init
            )
        )?;
    }
    stream.synchronize()?;

    dev_state_out.copy_to(&mut state_out)?;


    let mut f = File::create("dopr54_rust_out_gpu_yay.txt")?;
    let err: c_int = 0;
    writeln!(f, "err {}", err)?;
    writeln!(f, "dim {}", dump.dim)?;
    writeln!(f, "nt {}", dump.nt)?;
    writeln!(f, "t")?;
    for ti in &dump.t {
        writeln!(f, "{:.16e}", ti)?;
    }
    writeln!(f, "states")?;

    // (step, particle, dim) = ((s * n) + tid) * DIM
    // here n=1, tid=0, so index = s*DIM + j
    for step in 0..nt {
        write!(f, "step {}", step)?;
        let base = step * n * DIM;
        for j in 0..dim {
            let v = state_out[base + j];
            write!(f, " {:.16e}", v)?;
        }
        writeln!(f)?;
    }

    // Temp
    Ok(OutputFrame(core::array::from_fn(|_| None)))
}