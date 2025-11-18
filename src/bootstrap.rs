use crate::dispatch::gpu_dispatch;
use crate::python::{PyConfig, PyEngine, PyPotentialRecipe, py_runtime_err, translate_recipe};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use shared::Config;

#[pyfunction]
pub fn simulation_ctx<'py>(
    _py: Python<'py>,
    py_recipes: Vec<Vec<PyPotentialRecipe>>,
    initial_conditions: Vec<PyReadonlyArray2<'py, f64>>,
    py_config: PyConfig,
) -> PyResult<()>
// PyResult<(
//     Bound<'py, PyArray3<f64>>,
//     Bound<'py, PyArray2<f64>>,
//     Bound<'py, PyArray2<f64>>,
//     Bound<'py, PyArray2<isize>>,
// )>
{
    let _ctx = py_runtime_err(cust::quick_init())?;

    let states = initial_conditions
        .into_iter()
        .map(|arr| {
            arr.as_array()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f64>>>()
        })
        .collect();

    let stages = py_recipes
        .iter()
        .map(|stage| {
            let mut goffset: usize = 0;
            stage
                .iter()
                .map(|recipe| translate_recipe(recipe, &mut goffset))
                .collect()
        })
        .collect();

    let config: Config = py_config.clone().into();

    let (_results, _debug) = match py_config.engine {
        PyEngine::CPU => (0.0f64, 0.0f64),
        PyEngine::GPU => py_runtime_err(gpu_dispatch(states, stages, config, py_config))?,
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
