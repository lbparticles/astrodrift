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
use shared::{Linspace,Tolerance,Model,Config,Index};
pub use variant::PyVariant;
use crate::integrators::run_integration;
use crate::state::InputFrame;
use crate::tree::AdjacencyMatrix;

#[derive(Default, Clone, Debug)]
pub struct BoundLinspace(pub Linspace);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundLinspace {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // --- Case 1: (start, end, num) tuple ---
        if let Ok(tup) = obj.cast::<PyTuple>() {
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "Linspace tuple must have 3 elements: (start, end, steps)",
                ));
            }
            let start: f64 = tup.get_item(0)?.extract()?;
            let end: f64 = tup.get_item(1)?.extract()?;
            let steps: usize = tup.get_item(2)?.extract()?;
            return Ok(BoundLinspace(Linspace{start, end, steps}));
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
            let steps = n;
            return Ok(BoundLinspace(Linspace{start, end, steps}));
        }
        if let Ok(seq) = obj.extract::<Vec<f64>>() {
            if seq.len() < 2 {
                return Err(PyValueError::new_err(
                    "List must have at least two elements to form Linspace",
                ));
            }
            let start = seq[0];
            let end = *seq.last().unwrap();
            let steps = seq.len();
            return Ok(BoundLinspace(Linspace{start, end, steps}));
        }

        Err(PyValueError::new_err(
            "Expected (start, end, steps) tuple or 1D numpy.linspace array",
        ))
    }
}

#[derive(Default, Clone, Debug)]
pub struct BoundTolerance(pub Tolerance);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundTolerance {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tup) = obj.cast::<PyTuple>()
            && tup.len() == 2
        {
            let rtol: f64 = tup.get_item(0)?.extract()?;
            let atol: f64 = tup.get_item(1)?.extract()?;
            return Ok(BoundTolerance(Tolerance{rtol, atol}));
        }

        // Accept single float for convenience
        if let Ok(val) = obj.extract::<f64>() {
            return Ok(BoundTolerance(Tolerance{rtol: val, atol: val}));
        }

        Err(PyValueError::new_err(
            "Expected (rtol, atol) tuple or single float tolerance value",
        ))
    }
}

#[pyclass(name = "Config")]
#[derive(Debug)]
pub struct PyConfig {
    inner: Config,
    adjacency_matrix: AdjacencyMatrix,
}

impl PyConfig {
    fn build_tree(&self, containers: Vec<Container>) -> (Model, InputFrame) {
        let input = containers_to_array(containers);
        let (x,y) = self.adjacency_matrix.build(&input);
        // println!("{:?}",x);
        // println!("{:?}",y);
        (x,y)
    }
}


fn containers_to_array(mut v: Vec<Container>) -> [Option<Container>; 11] {
    if v.len() > 11 {
        v.truncate(11);
    }
    let mut out: [Option<Container>; 11] = std::array::from_fn(|_| None);

    for (i, item) in v.into_iter().enumerate() {
        // i is guaranteed < 11 due to truncate above
        out[i] = Some(item);
    }
    out
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (engine=None,method=None,variant=None,flags=None,ts=None,tolerance=None,devices=None))]
    // `devices` carries an allow: the owned Vec is how pyo3 extracts the
    // Python sequence; it is only read afterwards.
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        engine: Option<PyEngine>,
        method: Option<PyMethod>,
        variant: Option<PyVariant>,
        flags: Option<Modern>,
        ts: Option<BoundLinspace>,
        tolerance: Option<BoundTolerance>,
        devices: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let mut inner = Config::new(
            engine.unwrap_or_default().inner,
            method.unwrap_or_default().inner,
            variant.unwrap_or_default().inner,
            flags.unwrap_or_default().inner,
            ts.unwrap_or_default().0,
            tolerance.unwrap_or_default().0,
        );
        if let Some(devs) = &devices {
            inner.set_devices(devs)
                .map_err(PyValueError::new_err)?;
        }
        let thing = Self {
            inner,
            adjacency_matrix: AdjacencyMatrix(0),
        };
        println!("newpyconfig");
        Ok(thing)
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
        // Failure modes from the dispatch layer surface in Python as
        // RuntimeError (pyo3 converts the Err into a raised exception).
        let results = run_integration(&self.inner, &meal, &istates)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let _items: Vec<Py<PyAny>> = results.0
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|arr| {
                // Build a Python list for each Some(...)
                // PyList::new -> PyResult<Bound<PyList>>
                // .into_any() -> Bound<PyAny>
                // .unbind() -> Py<PyAny>
                PyList::new(py, arr.data.as_slice()).map(|lst| lst.into_any().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let items: Vec<Py<PyAny>> = Vec::new();
        PyList::new(py, items)
    }

    #[pyo3(signature = (node,*args))]
    #[allow(clippy::needless_pass_by_value)] // pyo3 extracts pyclass args by value (clone-based)
    fn dependency<'py>(
        &mut self,
        _py: Python<'py>,
        node: Container,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        let mut dep: Vec<Index> = Vec::new();

        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            dep.push(container.dependency_label);
        }
        for &x in &dep {
            self.adjacency_matrix
                .set(x, node.dependency_label, true);
        }
        Ok(())
    }
    #[pyo3(signature = ())]
    fn info(&self) {
        println!("{self:?}");
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
