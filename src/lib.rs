pub mod engines;
pub mod tables;
use pyo3::prelude::*;

pub use crate::engines::{integrate_gpu};

#[pymodule]
fn drift_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ingest, m)?)?;
    Ok(())
}
