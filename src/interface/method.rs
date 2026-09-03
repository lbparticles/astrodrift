use pyo3::prelude::*;

use crate::methods::registry;

/// Python-facing wrapper around `shared::Method`. Accepts every upstream
/// spelling (galpy `dopr54_c`, scipy `RK45`, REBOUND `IAS15`, gala
/// `Ruth4Integrator`, ...) via the registry parser; see
/// [`registry::parse_name`].
#[pyclass(name = "Method", from_py_object)]
#[derive(Default, Clone)]
pub struct PyMethod {
    pub inner: shared::Method,
}
#[pymethods]
impl PyMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match registry::parse_name(name) {
            Some(inner) => Ok(Self { inner }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid Method {name:?}; accepted spellings include: {} \
                 (galpy, scipy, REBOUND and gala names are also accepted)",
                registry::accepted_names()
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Method({})",
            registry::spec(self.inner).method.canonical_name()
        )
    }
}
