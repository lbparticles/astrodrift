use pyo3::prelude::*;

mod bootstrap;
mod dispatch;
mod dopr54_cpu;
mod dopr54_cpu_no_libc;
pub mod index_helpers;
mod python;
mod tables;

use crate::bootstrap::simulation_ctx;
use crate::python::{
    PyConfig, PyDebug, PyEngine, PyIntMethod, PyInterpolation, PyOptimisation, PyPotentialNames,
    PyPotentialRecipe,
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
