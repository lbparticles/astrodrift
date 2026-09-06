use pyo3::prelude::*;

use super::selector::extract_value;

#[derive(Default)]
pub struct PyEngine(shared::Engine);

impl<'a, 'py> FromPyObject<'a, 'py> for PyEngine {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match extract_value(object, "Engine")?.as_str() {
            "CPU" => Ok(Self(shared::Engine::CPU)),
            "GPU" => Ok(Self(shared::Engine::GPU)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "invalid Engine value",
            )),
        }
    }
}

impl From<PyEngine> for shared::Engine {
    fn from(value: PyEngine) -> Self {
        value.0
    }
}
