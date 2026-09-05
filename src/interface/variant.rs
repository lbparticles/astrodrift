use pyo3::prelude::*;
#[pyclass(name = "Variant", from_py_object)]
#[derive(Default, Clone)]
pub struct PyVariant {
    pub inner: shared::Variant,
}
#[pymethods]
impl PyVariant {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "Modern" => shared::Variant::Modern,
            "Compatible" => shared::Variant::Compatible,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Variant")),
        };
        Ok(Self { inner })
    }
}
