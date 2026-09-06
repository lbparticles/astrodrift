//! Translate backend failures into Python exceptions.
//!
//! Input validation and unsupported configurations keep their conventional
//! built-in exceptions. Failures returned by a supported backend use
//! [`DriftError`] and its [`IntegrationError`] subclass.

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::{Bound, create_exception};

use crate::integrators::IntegrationError as RustIntegrationError;

create_exception!(
    drift,
    DriftError,
    PyRuntimeError,
    "Base class for operational errors reported by drift."
);

create_exception!(
    drift,
    IntegrationError,
    DriftError,
    "An integration could not be completed."
);

/// Register the backend exception types on the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("DriftError", module.py().get_type::<DriftError>())?;
    module.add(
        "IntegrationError",
        module.py().get_type::<IntegrationError>(),
    )?;
    Ok(())
}

impl From<RustIntegrationError> for PyErr {
    fn from(error: RustIntegrationError) -> Self {
        match error {
            error @ RustIntegrationError::UnsupportedConfiguration { .. } => {
                PyNotImplementedError::new_err(error.to_string())
            }
            RustIntegrationError::Dispatch(error) => IntegrationError::new_err(error.to_string()),
        }
    }
}
