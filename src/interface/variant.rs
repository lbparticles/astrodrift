use pyo3::prelude::*;

/// Kernel variant selection.
///
/// Access members as attributes, e.g. ``Variant.Compatible``.
#[pyclass(eq, eq_int, name = "Variant")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PyVariant {
    /// Conservative code path shared by the CPU and GPU backends.
    #[default]
    Compatible,
    /// Experimental fast path (dispatch pending).
    Modern,
}

impl From<PyVariant> for shared::Variant {
    fn from(value: PyVariant) -> Self {
        match value {
            PyVariant::Compatible => shared::Variant::Compatible,
            PyVariant::Modern => shared::Variant::Modern,
        }
    }
}
