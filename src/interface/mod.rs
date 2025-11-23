use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyArray1;
use pyo3::types::{PyAny,PyList,PyDict, PyModule,PyTuple};
use pyo3::ffi::c_str;
use numpy::PyArrayMethods;

mod flag;
mod recipe;
mod potential;
mod container;
mod engine;
mod variant;
mod method;

use container::Container;
use engine::PyEngine;
use variant::PyVariant;
use method::PyMethod;
use flag::Modern;
use potential::PyPotential;
use recipe::PyRecipe;


#[derive(Default,Clone, Debug)]
pub struct BoundLinspace(pub shared::Linspace);
impl<'a,'py> FromPyObject<'a,'py> for BoundLinspace {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // --- Case 1: (start, end, num) tuple ---
        if let Ok(tup) = obj.cast::<PyTuple>() {
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "Linspace tuple must have 3 elements: (start, end, num)"
                ));
            }
            let start: f64 = tup.get_item(0)?.extract()?;
            let end: f64 = tup.get_item(1)?.extract()?;
            let num: usize = tup.get_item(2)?.extract()?;
            return Ok(BoundLinspace(shared::Linspace(start, end, num)));
        }

        // --- Case 2: NumPy array ---
        if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
            let slice = unsafe { arr.as_slice_mut()? };
            let n = slice.len();

            if n < 2 {
                return Err(PyValueError::new_err(
                    "NumPy linspace must contain at least two points"
                ));
            }

            let start = slice.first().copied().unwrap_or(0.0);
            let end = slice.last().copied().unwrap_or(start);
            let num = n;
            return Ok(BoundLinspace(shared::Linspace(start, end, num)));
        }
        if let Ok(seq) = obj.extract::<Vec<f64>>() {
            if seq.len() < 2 {
                return Err(PyValueError::new_err(
                    "List must have at least two elements to form Linspace"
                ));
            }
            let start = seq[0];
            let end = *seq.last().unwrap();
            let num = seq.len();
            return Ok(BoundLinspace(shared::Linspace(start, end, num)));
        }

        Err(PyValueError::new_err(
            "Expected (start, end, num) tuple or 1D numpy.linspace array",
        ))
    }
}

#[derive(Default,Clone, Debug)]
pub struct BoundTolerance(pub shared::Tolerance);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundTolerance{
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tup) = obj.cast::<PyTuple>() && tup.len() == 2 {
            let rtol: f64 = tup.get_item(0)?.extract()?;
            let atol: f64 = tup.get_item(1)?.extract()?;
            return Ok(BoundTolerance(shared::Tolerance(rtol, atol)));
        }

        // Accept single float for convenience
        if let Ok(val) = obj.extract::<f64>() {
            return Ok(BoundTolerance(shared::Tolerance(val, val)));
        }

        Err(PyValueError::new_err(
            "Expected (rtol, atol) tuple or single float tolerance value",
        ))
    }
}


#[pyclass(name = "Config")]
pub struct PyConfig {
    inner: shared::Config,
}

impl PyConfig {
    fn build_tree(_containers:Vec<Container>)->(shared::Meal,shared::IStates){
       ([[shared::Recipe::default();shared::MAX_RECIPES];shared::MAX_COURSES],[[0.0; shared::ILENGTH]; shared::MAX_STATES]) 
    }
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (engine=None,method=None,variant=None,flags=None,ts=None,tolerance=None))]
    fn new(
        engine: Option<PyEngine>,
        method: Option<PyMethod>,
        variant: Option<PyVariant>,
        flags: Option<Modern>,
        ts: Option<BoundLinspace>,
        tolerance: Option<BoundTolerance>,
    ) -> Self {
        Self {
            inner: shared::Config::new(
                engine.unwrap_or_default().inner,
                method.unwrap_or_default().inner,
                variant.unwrap_or_default().inner,
                flags.unwrap_or_default().inner,
                ts.unwrap_or_default().0,
                tolerance.unwrap_or_default().0,
            ),
        }
    }


    // #[pyo3(signature = (*args))]
    fn run<'py>(&self,py:Python<'py>,args: &Bound<'py,PyTuple>)->PyResult<Bound<'py, PyList>>{
        let mut containers: Vec<Container> = Vec::new();

        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            containers.push(container.clone());
        }
        let (meal,istates) = Self::build_tree(containers);

        let results = self.inner.run(meal, istates);
        
    	PyList::new(py, results.iter().map(|a| a.to_vec()))
    }
    
}


//
// Python Module Declaration
//
#[pymodule]
fn drift_rs(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyMethod>()?;
    m.add_class::<PyVariant>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyPotential>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<Modern>()?;
    m.add_class::<Container>()?;
    m.add_function(wrap_pyfunction!(container::test_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::part_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::bg_feature, m)?)?;

    // Define enum.Flag in Python
    let locals = PyDict::new(py);
    py.run(c_str!(
        r#"
import enum

class ModernFlag(enum.Flag):
    NONE        = 0
    READ        = 1 << 0
    WRITE       = 1 << 1
    EXECUTE     = 1 << 2
    DELETE      = 1 << 3
    READ_WRITE  = READ | WRITE
    FULL_ACCESS = READ | WRITE | EXECUTE | DELETE
"#),
        None,
        Some(&locals),
    )?;

    let py_enum = locals.get_item("ModernFlag").unwrap();
    m.add("ModernFlag", py_enum)?;

    Ok(())
}
