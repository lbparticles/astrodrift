use pyo3::prelude::*;

use super::selector::extract_value;

#[derive(Default)]
pub struct PyImplementation(shared::Implementation);

impl<'a, 'py> FromPyObject<'a, 'py> for PyImplementation {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match extract_value(object, "Implementation")?.as_str() {
            "GALPY" => Ok(Self(shared::Implementation::GALPY)),
            "SCIPY" => Ok(Self(shared::Implementation::SCIPY)),
            "DRIFT" => Ok(Self(shared::Implementation::DRIFT)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "invalid Implementation value",
            )),
        }
    }
}

impl From<PyImplementation> for shared::Implementation {
    fn from(value: PyImplementation) -> Self {
        value.0
    }
}
