use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyTuple};

mod container;
mod engine;
mod flag;
mod method;
mod recipe;
mod variant;

pub use container::Container;
pub use engine::PyEngine;
pub use flag::Modern;
pub use method::PyMethod;
pub use recipe::PyRecipe;
pub use variant::PyVariant;
use crate::integrators::run_integration;
use crate::tree::AdjacencyMatrix;

#[derive(Default, Clone, Debug)]
pub struct BoundLinspace(pub shared::Linspace);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundLinspace {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // --- Case 1: (start, end, num) tuple ---
        if let Ok(tup) = obj.cast::<PyTuple>() {
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "Linspace tuple must have 3 elements: (start, end, num)",
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
                    "NumPy linspace must contain at least two points",
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
                    "List must have at least two elements to form Linspace",
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

#[derive(Default, Clone, Debug)]
pub struct BoundTolerance(pub shared::Tolerance);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundTolerance {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tup) = obj.cast::<PyTuple>()
            && tup.len() == 2
        {
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
#[derive(Debug)]
pub struct PyConfig {
    inner: shared::Config,
    adjacency_matrix: AdjacencyMatrix,
}

impl PyConfig {
    fn build_tree(&self,
         containers: Vec<Container>) -> (shared::Meal, shared::InputFrame) {
        let input = vec_to_option_array_11(containers);
        let (x,y) = self.adjacency_matrix.build(input);
        // println!("{:?}",x);
        // println!("{:?}",y);
        (x,y)
    }
}


fn vec_to_option_array_11(mut v: Vec<Container>) -> Box<[Option<Container>; 11]> {
    if v.len() > 11 {
        v.truncate(11);
    }
    let mut out: Box<[Option<Container>; 11]> = Box::new([None,None,None,None,None,None,None,None,None,None,None]);

    // Copy by cloning into out[i]
    for (i, item) in v.iter().enumerate() {
        // i is guaranteed < 11 due to truncate above
        out[i] = Some(item.clone());
    }
    out 

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
            adjacency_matrix: AdjacencyMatrix(0),
        }
    }

    #[pyo3(signature = (*args))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
    ) 
    -> PyResult<Bound<'py, PyList>> 
    {
        let mut containers: Vec<Container> = Vec::new();

        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            containers.push(container.clone());
        }
        let (meal, istates) = self.build_tree(containers);

        let results = run_integration(self.inner, meal, istates).unwrap();
        let items: Vec<Py<PyAny>> = results.0
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|arr| {
                // Build a Python list for each Some(...)
                // PyList::new -> PyResult<Bound<PyList>>
                // .into_any() -> Bound<PyAny>
                // .unbind() -> Py<PyAny>
                PyList::new(py, arr.0.as_slice()).map(|lst| lst.into_any().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;

        PyList::new(py, items)
    }

    #[pyo3(signature = (node,*args))]
    fn dependency<'py>(
        &mut self,
        _py: Python<'py>,
        node: Container,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        let mut dep: Vec<shared::Index> = Vec::new();

        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            dep.push(container.dependency_label);
        }
        for x in dep.iter() {
            self.adjacency_matrix
                .set(x.clone(), node.dependency_label, true);
        }
        Ok(())
    }
    #[pyo3(signature = ())]
    fn info(&self) -> () {
        println!("{:?}", self);
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
    m.add_class::<PyRecipe>()?;
    m.add_class::<Modern>()?;
    m.add_class::<Container>()?;
    m.add_function(wrap_pyfunction!(container::test_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::part_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::bg_feature, m)?)?;

    // Define enum.Flag in Python
    let locals = PyDict::new(py);
    py.run(
        c_str!(
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
"#
        ),
        None,
        Some(&locals),
    )?;

    let py_enum = locals.get_item("ModernFlag").unwrap();
    m.add("ModernFlag", py_enum)?;

    Ok(())
}
