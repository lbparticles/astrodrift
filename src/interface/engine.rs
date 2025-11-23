use pyo3::prelude::*;
#[pyclass(name = "Engine")]
#[derive(Clone)]
pub struct PyEngine {
    pub inner: shared::Engine,
}
#[pymethods]
impl PyEngine {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "GPU" => shared::Engine::GPU,
            "CPU" => shared::Engine::CPU,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Engine")),
        };
        Ok(Self { inner })
    }
}
