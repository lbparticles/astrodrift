use pyo3::prelude::*;

use super::selector::extract_value;

#[derive(Default)]
pub struct PyVariant(shared::Variant);

impl<'a, 'py> FromPyObject<'a, 'py> for PyVariant {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match extract_value(object, "Variant")?.as_str() {
            "Compatible" => Ok(Self(shared::Variant::Compatible)),
            "Modern" => Ok(Self(shared::Variant::Modern)),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "invalid Variant value",
            )),
        }
    }
}

impl From<PyVariant> for shared::Variant {
    fn from(value: PyVariant) -> Self {
        value.0
    }
}
