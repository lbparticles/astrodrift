use pyo3::prelude::*;

use super::selector::extract_value;

#[derive(Default)]
pub struct PyMethod(shared::Method);

impl<'a, 'py> FromPyObject<'a, 'py> for PyMethod {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match extract_value(object, "Method")?.as_str() {
            "DOPR54" => Ok(Self(shared::Method::DOPR54)),
            "DOP853" => Ok(Self(shared::Method::DOP853)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "invalid Method value",
            )),
        }
    }
}

impl From<PyMethod> for shared::Method {
    fn from(value: PyMethod) -> Self {
        value.0
    }
}
