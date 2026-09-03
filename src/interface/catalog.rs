//! Python-facing mirror of [`crate::methods::registry::CATALOG`]: one dict
//! per integration method with its family, order, throughput-model stages,
//! status and upstream mirrors. Static data only — no driver calls, no
//! panics.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::methods::registry::{self, Family, Origin, Status};

/// Folder-style family label used in the Python-facing catalog.
fn family_name(family: Family) -> &'static str {
    family.folder()
}

/// Lowercase library label used in the Python-facing catalog.
fn origin_name(origin: Origin) -> &'static str {
    origin.name()
}

/// Status label used in the Python-facing catalog.
fn status_name(status: Status) -> &'static str {
    status.name()
}

/// Catalog row for one method as a Python dict (shared by both accessors).
fn spec_to_py(py: Python<'_>, method: shared::Method) -> PyResult<Bound<'_, PyDict>> {
    let row = registry::spec(method);
    let dict = PyDict::new(py);
    dict.set_item("name", row.method.canonical_name())?;
    dict.set_item("family", family_name(row.family))?;
    dict.set_item("order", row.order)?;
    dict.set_item("stages", row.stages)?;
    dict.set_item("status", status_name(row.status))?;
    let mirrors = PyList::empty(py);
    for &(origin, identifier) in row.mirrors {
        let mirror = PyDict::new(py);
        mirror.set_item("origin", origin_name(*origin))?;
        mirror.set_item("identifier", identifier)?;
        mirrors.append(mirror)?;
    }
    dict.set_item("mirrors", mirrors)?;
    Ok(dict)
}

/// Catalog of every integration method drift knows about.
///
/// Returns a list of dicts with keys:
/// - `name`: canonical drift name (`Method("DOP853")`, ...)
/// - `family`: mirror folder under `src/methods/`
///   (`"rk"`, `"symplectic"`, `"implicit"`, `"hybrid"`)
/// - `order`: integration order / scheme notes
/// - `stages`: force evaluations per step feeding the throughput model,
///   or `None` while uncalibrated
/// - `status`: `"implemented"` or `"stub"`
/// - `mirrors`: list of `{origin, identifier}` upstream spellings
#[pyfunction]
pub fn method_catalog(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let list = PyList::empty(py);
    for row in registry::CATALOG {
        list.append(spec_to_py(py, row.method)?)?;
    }
    Ok(list)
}

/// Convenience accessor: catalog row for one method name, in any accepted
/// upstream spelling. Raises `ValueError` on unknown names.
#[pyfunction]
pub fn method_info<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let Some(method) = registry::parse_name(name) else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid Method {name:?}; accepted spellings include: {}",
            registry::accepted_names()
        )));
    };
    spec_to_py(py, method)
}
