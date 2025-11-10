pub mod engines;
pub mod tables;

pub use crate::engines::integrate_gpu;

use pyo3::prelude::*;
#[pymodule]
fn drift_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ingest, m)?)?;
    Ok(())
}
