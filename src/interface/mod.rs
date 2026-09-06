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
use shared::{
    Config, INPUT_STATE_DIM, Linspace, MAX_MODEL_COMPONENTS, MAX_OUTPUT_TIMES, MAX_STATES, Real,
    Tolerance,
};
pub use variant::PyVariant;

// Python extraction and validation live here so shared::Linspace remains usable
// by GPU code without PyO3 or NumPy dependencies.
#[derive(Default, Clone, Debug)]
pub struct PyLinspace(pub Linspace);
impl<'a, 'py> FromPyObject<'a, 'py> for PyLinspace {
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
            return Self::new(start, end, steps);
        }

        // --- Case 2: NumPy array ---
        if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
            let readonly = arr.try_readonly().map_err(|_| {
                PyValueError::new_err("NumPy time array is already mutably borrowed")
            })?;
            let slice = readonly
                .as_slice()
                .map_err(|_| PyValueError::new_err("NumPy time arrays must be contiguous"))?;
            return Self::from_times(slice);
        }
        if let Ok(seq) = obj.extract::<Vec<f64>>() {
            return Self::from_times(&seq);
        }

        Err(PyValueError::new_err(
            "Expected (start, end, steps) tuple or 1D numpy.linspace array",
        ))
    }
}

impl PyLinspace {
    const GRID_ULP_TOLERANCE: Real = 8.0;

    fn new(start: Real, end: Real, steps: usize) -> PyResult<Self> {
        if !(2..=MAX_OUTPUT_TIMES).contains(&steps) {
            return Err(PyValueError::new_err(format!(
                "ts must contain between 2 and {MAX_OUTPUT_TIMES} output times"
            )));
        }
        if !start.is_finite() || !end.is_finite() {
            return Err(PyValueError::new_err("ts endpoints must be finite"));
        }
        let span = end - start;
        if span == 0.0 || !span.is_finite() {
            return Err(PyValueError::new_err(
                "ts must span a non-zero finite interval",
            ));
        }

        Ok(Self(Linspace { start, end, steps }))
    }

    // FIXME: Store and pass arbitrary requested times directly, then remove
    // this temporary affine-grid restriction and reconstruction check.
    fn from_times(times: &[Real]) -> PyResult<Self> {
        if !(2..=MAX_OUTPUT_TIMES).contains(&times.len()) {
            return Err(PyValueError::new_err(format!(
                "ts must contain between 2 and {MAX_OUTPUT_TIMES} output times"
            )));
        }
        if times.iter().any(|time| !time.is_finite()) {
            return Err(PyValueError::new_err("all ts values must be finite"));
        }

        let grid = Self::new(times[0], times[times.len() - 1], times.len())?;
        let increasing = grid.0.end > grid.0.start;
        if times.windows(2).any(|pair| {
            if increasing {
                pair[1] <= pair[0]
            } else {
                pair[1] >= pair[0]
            }
        }) {
            return Err(PyValueError::new_err(
                "ts values must be strictly increasing or strictly decreasing",
            ));
        }

        let step = (grid.0.end - grid.0.start) / ((grid.0.steps - 1) as Real);
        for (index, &actual) in times.iter().enumerate() {
            let expected = grid.0.sample(index);
            let scale = actual
                .abs()
                .max(expected.abs())
                .max(step.abs())
                .max(Real::MIN_POSITIVE);
            let tolerance = Self::GRID_ULP_TOLERANCE * Real::EPSILON * scale;
            if (actual - expected).abs() > tolerance {
                return Err(PyValueError::new_err(
                    "ts values must be uniformly spaced; arbitrary output times are not yet supported",
                ));
            }
        }

        Ok(grid)
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
    // Dependencies use stable identities until build_tree assigns dense labels.
    dependencies: Vec<(u64, u64)>,
}

impl PyConfig {
    fn build_tree(&self, containers: Vec<Container>) -> PyResult<IntegrationPlan> {
        let mut containers_by_label = core::array::from_fn(|_| None);
        let mut identity_by_label = [None; MAX_STATES];

        for (label, container) in containers.into_iter().enumerate() {
            if identity_by_label.contains(&Some(container.identity)) {
                return Err(PyValueError::new_err(
                    "run() received the same container more than once",
                ));
            }
            identity_by_label[label] = Some(container.identity);
            containers_by_label[label] = Some(container);
        }

        let mut adjacency_matrix = AdjacencyMatrix(0);
        for &(dependency, node) in &self.dependencies {
            let dependency_label = identity_by_label
                .iter()
                .position(|&identity| identity == Some(dependency));
            let node_label = identity_by_label
                .iter()
                .position(|&identity| identity == Some(node));
            let (Some(dependency_label), Some(node_label)) = (dependency_label, node_label) else {
                return Err(PyValueError::new_err(
                    "every container used in dependency() must be passed to run()",
                ));
            };
            adjacency_matrix.set(dependency_label, node_label, true);
        }

        Ok(adjacency_matrix.build(containers_by_label))
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
        ts: Option<PyLinspace>,
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
            dependencies: Vec::new(),
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
        if args.len() > MAX_MODEL_COMPONENTS {
            return Err(PyValueError::new_err(format!(
                "run() received {} containers but a model supports at most \
                 {MAX_MODEL_COMPONENTS}",
                args.len()
            )));
        }
        let mut containers = Vec::with_capacity(args.len().min(MAX_STATES));
        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            containers.push(container.clone());
        }
        let has_state: Vec<bool> = containers.iter().map(|c| c.state.is_some()).collect();
        let plan = self.build_tree(containers)?;
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
            let Some(item) = items.get_mut(container_label) else {
                return Err(PyRuntimeError::new_err(
                    "integration returned an output for an unknown container",
                ));
            };
            let array = PyArray1::from_slice(py, &state.data)
                .reshape([state.num_times, state.num_particles, INPUT_STATE_DIM])?
                .into_any()
                .unbind();
            *item = Some(array);
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
            self.dependencies.push((container.identity, node.identity));
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
