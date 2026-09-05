use pyo3::prelude::*;

/// Integration scheme.
///
/// Access members as attributes, e.g. ``Method.DOPR54``.
#[pyclass(eq, eq_int, name = "Method")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PyMethod {
    /// Dormand-Prince 5(4): the CUDA-accelerated fixed-step scheme.
    #[default]
    DOPR54,
    /// Dormand-Prince 8(5) 3: high-order CPU scheme (dispatch pending).
    DOP853,
}

impl From<PyMethod> for shared::Method {
    fn from(value: PyMethod) -> Self {
        match value {
            PyMethod::DOPR54 => shared::Method::DOPR54,
            PyMethod::DOP853 => shared::Method::DOP853,
        }
    }
}
