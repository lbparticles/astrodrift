use pyo3::prelude::*;

/// Execution backend for an integration.
///
/// Access members as attributes, e.g. ``Engine.CPU``. The default is
/// :attr:`CPU`; GPU requires a CUDA device and must be requested explicitly.
#[pyclass(eq, eq_int, name = "Engine")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PyEngine {
    /// Integrate on the CPU.
    #[default]
    CPU,
    /// Integrate on the GPU (requires a CUDA device).
    GPU,
}

impl From<PyEngine> for shared::Engine {
    fn from(value: PyEngine) -> Self {
        match value {
            PyEngine::CPU => shared::Engine::CPU,
            PyEngine::GPU => shared::Engine::GPU,
        }
    }
}
