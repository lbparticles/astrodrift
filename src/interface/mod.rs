use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyTuple};

mod container;
mod engine;
mod flag;
mod method;
mod recipe;
mod variant;

use crate::integrators::run_integration;
use crate::tree::{AdjacencyMatrix, IntegrationPlan};
pub use container::Container;
pub use engine::PyEngine;
pub use flag::Modern;
pub use method::PyMethod;
pub use recipe::PyRecipe;
use shared::{Config, INPUT_STATE_DIM, Linspace, MAX_STATES, Tolerance};
pub use variant::PyVariant;

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
            return Ok(BoundLinspace(Linspace { start, end, steps }));
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
            return Ok(BoundLinspace(Linspace { start, end, steps }));
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
            return Ok(BoundLinspace(Linspace { start, end, steps }));
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
            return Ok(BoundTolerance(Tolerance { rtol, atol }));
        }

        // Accept single float for convenience
        if let Ok(val) = obj.extract::<f64>() {
            return Ok(BoundTolerance(Tolerance {
                rtol: val,
                atol: val,
            }));
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

struct PyIntegrationPlan {
    inner: IntegrationPlan,
    argument_position_by_label: [Option<usize>; MAX_STATES],
}

impl PyConfig {
    fn build_tree(&self, containers: Vec<Container>) -> PyResult<PyIntegrationPlan> {
        let mut containers_by_label = core::array::from_fn(|_| None);
        let mut argument_position_by_label = [None; MAX_STATES];

        for (position, container) in containers.into_iter().enumerate() {
            let label = container.dependency_label;
            if label >= MAX_STATES {
                return Err(PyValueError::new_err(format!(
                    "container label {label} exceeds the current model capacity of {MAX_STATES}"
                )));
            }
            if containers_by_label[label].is_some() {
                return Err(PyValueError::new_err(
                    "run() received the same container more than once",
                ));
            }
            argument_position_by_label[label] = Some(position);
            containers_by_label[label] = Some(container);
        }

        Ok(PyIntegrationPlan {
            inner: self.adjacency_matrix.build(containers_by_label),
            argument_position_by_label,
        })
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
        let thing = Self {
            inner: Config::new(
                engine.unwrap_or_default().inner,
                method.unwrap_or_default().inner,
                variant.unwrap_or_default().inner,
                flags.unwrap_or_default().inner,
                ts.unwrap_or_default().0,
                tolerance.unwrap_or_default().0,
            ),
            adjacency_matrix: AdjacencyMatrix(0),
        };
        println!("newpyconfig");
        thing
    }

    #[pyo3(signature = (*args))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut containers = Vec::with_capacity(args.len().min(MAX_STATES));
        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            containers.push(container.clone());
        }
        let has_state: Vec<bool> = containers.iter().map(|c| c.state.is_some()).collect();
        let plan = self.build_tree(containers)?;
        let PyIntegrationPlan {
            inner: plan,
            argument_position_by_label,
        } = plan;
        let IntegrationPlan {
            model,
            input_frame,
            container_label_by_stage,
        } = plan;
        let results = run_integration(self.inner, model, input_frame)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

        let mut items: Vec<Option<Py<PyAny>>> = core::iter::repeat_with(|| None)
            .take(has_state.len())
            .collect();
        for (result, container_label) in results.0.iter().zip(container_label_by_stage) {
            let Some(state) = result else { continue };
            let Some(container_label) = container_label else {
                return Err(PyRuntimeError::new_err(
                    "integration returned an output without a source container",
                ));
            };
            let Some(position) = argument_position_by_label[container_label] else {
                return Err(PyRuntimeError::new_err(
                    "integration returned an output for an unknown container",
                ));
            };
            let array = PyArray1::from_slice(py, &state.data)
                .reshape([state.num_times, state.num_particles, INPUT_STATE_DIM])?
                .into_any()
                .unbind();
            items[position] = Some(array);
        }

        let items = items
            .into_iter()
            .zip(has_state)
            .map(|(item, has_state)| match (item, has_state) {
                (Some(item), _) => Ok(item),
                (None, false) => Ok(py.None()),
                (None, true) => Err(PyNotImplementedError::new_err(
                    "run() produced no output for a particle group: the supported combinations \
                     are Engine.GPU with Method.DOPR54 or Method.DOP853 and Variant.Compatible",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    #[pyo3(signature = (node,*args))]
    fn dependency<'py>(&mut self, node: Container, args: &Bound<'py, PyTuple>) -> PyResult<()> {
        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            self.adjacency_matrix
                .set(container.dependency_label, node.dependency_label, true);
        }
        Ok(())
    }
    #[pyo3(signature = ())]
    fn info(&self) {
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
