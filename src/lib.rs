// use libm::pow;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use shared::StaticInterface;
// use std::f64::consts::PI;

mod dispatch;
mod index_helpers;
mod tables;
mod translation;
use crate::dispatch::gpu_dispatch;

use translation::{
    PyDebug, PyEngine, PyIntMethod, PyInterpolation, PyOptimisation, PyPotentialNames,
    PyPotentialRecipe, translate_recipe,
};

const NF64: usize = 6;
// FIXME: double check in MPA impl
const FAC_MIN: f64 = 0.33;
const FAC_MAX: f64 = 6.0;
const SAFETY: f64 = 0.9;
const DT_MIN: f64 = 1.0e-12;
const DT_MAX: f64 = 0.25;

/// An extension trait may be cleaner (e.g. let _ctx = cust::quick_init().into_py()?;)
fn py_runtime_err<T, E: std::fmt::Display>(res: Result<T, E>) -> PyResult<T> {
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[allow(dead_code)]
#[pyclass(name = "Interface")]
#[derive(Clone)]
pub struct PyInterface {
    poll_number: usize,
    steps_cap: usize,
    t_end: f64,
    dt0: f64,
    atol: f64,
    rtol: f64,
    reverse: bool,
    engine: PyEngine,
    method: PyIntMethod,
    optimisation: PyOptimisation,
    interpolation: PyInterpolation,
    debug: PyDebug,
}

#[pymethods]
impl PyInterface {
    #[new]
    pub fn new(
        poll_number: usize,
        steps_cap: usize,
        t_end: f64,
        dt0: f64,
        atol: f64,
        rtol: f64,
        reverse: bool,
        engine: PyEngine,
        method: PyIntMethod,
        optimisation: PyOptimisation,
        interpolation: PyInterpolation,
        debug: PyDebug,
    ) -> Self {
        PyInterface {
            poll_number,
            steps_cap,
            t_end,
            dt0,
            atol,
            rtol,
            reverse,
            engine,
            method,
            optimisation,
            interpolation,
            debug,
        }
    }
}

#[pyfunction]
fn simulation_ctx<'py>(
    _py: Python<'py>,
    py_recipes: Vec<Vec<PyPotentialRecipe>>,
    states: Vec<PyReadonlyArray2<'py, f64>>,
    config: PyInterface,
) -> PyResult<()>
// PyResult<(
//     Bound<'py, PyArray3<f64>>,
//     Bound<'py, PyArray2<f64>>,
//     Bound<'py, PyArray2<f64>>,
//     Bound<'py, PyArray2<isize>>,
// )>
{
    let _ctx = py_runtime_err(cust::quick_init())?;
    let ic = states[0].as_array();

    let n: usize = ic.shape()[0];

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("N must be > 0"));
    }

    let time_direction: f64 = if config.reverse { -1.0 } else { 1.0 };
    let target_t_end = if config.reverse {
        -config.t_end
    } else {
        config.t_end
    };

    let mut state_out = vec![0.0f64; config.steps_cap * n * NF64];

    if let Some(slice) = ic.as_slice() {
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
    let mut goffset: usize = 0;
    let lookuptable: Vec<f64> = Vec::new();
    let recipes = vec![translate_recipe(py_recipes[0][0].clone(), &mut goffset); 1];
    let statics = StaticInterface {
        t_end: target_t_end,
        n,
        steps_cap: config.steps_cap,
        atol: config.atol,
        rtol: config.rtol,
        fac_min: FAC_MIN,
        fac_max: FAC_MAX,
        safety: SAFETY,
        dt_min: DT_MIN,
        dt_max: DT_MAX,
        poll_number: config.poll_number,
        time_direction,
    };

    let (_results, _debug) = match config.engine {
        PyEngine::CPU => (0.0f64, 0.0f64),
        PyEngine::GPU => py_runtime_err(gpu_dispatch(
            &mut state_out,
            recipes,
            statics,
            lookuptable,
            config,
        ))?,
    };

    // let app_ts = PyArray2::from_vec2(py, &debug.app_ts0)?;
    // let app_indices = PyArray2::from_vec2(py, &debug.indices)?;
    // let flat_state = PyArray1::from_vec(py, results);
    // let arr_state = flat_state.reshape([config.steps_cap, n, 6])?;
    // let flat_time = PyArray1::from_vec(py, results.time_out);
    // let arr_time = flat_time.reshape([config.steps_cap, n])?;

    Ok(())
    // Ok((arr_state, arr_time, app_ts, app_indices))
}

#[pymodule]
fn drift_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulation_ctx, m)?)?;
    m.add_class::<PyPotentialNames>()?;
    m.add_class::<PyPotentialRecipe>()?;
    m.add_class::<PyIntMethod>()?;
    m.add_class::<PyOptimisation>()?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyDebug>()?;
    m.add_class::<PyInterpolation>()?;
    m.add_class::<PyInterface>()?;
    Ok(())
}
