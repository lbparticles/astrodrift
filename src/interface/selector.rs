use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pub(super) fn extract_value<'a, 'py>(
    object: Borrowed<'a, 'py, PyAny>,
    selector_name: &str,
) -> PyResult<String> {
    let selector = object
        .py()
        .import("drift.selectors")?
        .getattr(selector_name)?;
    if !object.is_exact_instance(&selector) {
        return Err(PyTypeError::new_err(format!(
            "expected a drift.{selector_name} member"
        )));
    }

    object
        .getattr("value")?
        .extract()
        .map_err(|_| PyValueError::new_err(format!("invalid {selector_name} value")))
}
