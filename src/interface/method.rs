use pyo3::prelude::*;
#[pyclass(name = "Method")]
#[derive(Default, Clone)]
pub struct PyMethod {
    pub inner: shared::Method,
}
#[pymethods]
impl PyMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "DOP853" => shared::Method::DOP853,
            "DOPR54" => shared::Method::DOPR54,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Method")),
        };
        Ok(Self { inner })
    }
}
