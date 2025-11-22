use pyo3::prelude::*;
// use numpy::PyReadonlyArray1;

use shared::{
    Config,
    Engine,
    Index,
    Linspace,
    Method,
    Tolerance,
    Variant,
    // Real,
};

//
// Wrap the core enums
//
#[pyclass(name = "Engine")]
#[derive(Clone)]
pub struct PyEngine {
    inner: Engine,
}
#[pymethods]
impl PyEngine {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "GPU" => Engine::GPU,
            "CPU" => Engine::CPU,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Engine")),
        };
        Ok(Self { inner })
    }
}

#[pyclass(name = "Method")]
#[derive(Clone)]
pub struct PyMethod {
    inner: Method,
}
#[pymethods]
impl PyMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "DOP853" => Method::DOP853,
            "DOPR54" => Method::DOPR54,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Method")),
        };
        Ok(Self { inner })
    }
}

#[pyclass(name = "Variant")]
#[derive(Clone)]
pub struct PyVariant {
    inner: Variant,
}
#[pymethods]
impl PyVariant {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "Modern" => Variant::Modern,
            "Compatible" => Variant::Compatible,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Variant")),
        };
        Ok(Self { inner })
    }
}

//
// Wrap Config
//
#[pyclass(name = "Config")]
pub struct PyConfig {
    inner: Config,
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new(
        engine: &PyEngine,
        method: &PyMethod,
        variant: &PyVariant,
        ts: Linspace,
        tolerance: Tolerance,
        part_num: Index,
    ) -> Self {
        Self {
            inner: Config::new(
                engine.inner.clone(),
                method.inner.clone(),
                variant.inner.clone(),
                ts,
                tolerance,
                part_num,
            ),
        }
    }

    fn run(&self) {
        self.inner.run();
    }
}

//
// Python Module Declaration
//
#[pymodule]
fn python_bindings(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyMethod>()?;
    m.add_class::<PyVariant>()?;
    m.add_class::<PyConfig>()?;
    Ok(())
}
