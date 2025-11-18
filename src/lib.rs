use pyo3::prelude::*;

mod dispatch;
mod index_helpers;
mod tables;
mod python;
mod bootstrap;
mod dopr54_cpu;


use crate::bootstrap::simulation_ctx;
use crate::python::{
    PyPotentialNames,
    PyPotentialRecipe,
    PyIntMethod,
    PyOptimisation,
    PyEngine,
    PyDebug,
    PyInterpolation,
    PyConfig,
};


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
    m.add_class::<PyConfig>()?;
    Ok(())
}
