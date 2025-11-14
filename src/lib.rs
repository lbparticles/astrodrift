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

fn states_to_nested_vec(states: Vec<PyReadonlyArray2<'_, f64>>) -> Vec<Vec<Vec<f64>>> {
    states
        .into_iter()
        .map(|arr| {
            arr.as_array()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f64>>>()
        })
        .collect()
}
#[pyfunction]
fn simulation_ctx<'py>(
    _py: Python<'py>,
    py_recipes: Vec<Vec<PyPotentialRecipe>>,
    initial_conditions: Vec<PyReadonlyArray2<'py, f64>>,
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
    let states = states_to_nested_vec(initial_conditions);

    let time_direction: f64 = if config.reverse { -1.0 } else { 1.0 };
    let target_t_end = if config.reverse {
        -config.t_end
    } else {
        config.t_end
    };

    let mut goffset: usize = 0;

    let stages = py_recipes
        .iter()
        .map(|stage| stage
            .iter()
            .map(|recipe| translate_recipe(recipe, &mut goffset))
            .collect())
        .collect();

    let statics = StaticInterface {
        n: 0,
        t_end: target_t_end,
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
            states,
            stages,
            statics,
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
