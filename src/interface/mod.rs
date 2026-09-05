use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Mutex;
use pyo3::types::{PyAny, PyList, PyModule, PyTuple};

mod container;
mod engine;
mod method;
mod recipe;
mod variant;

use crate::integrators::run_integration;
use crate::state::InputFrame;
use crate::tree::AdjacencyMatrix;
pub use container::Container;
pub use engine::PyEngine;
pub use method::PyMethod;
pub use recipe::PyRecipe;
use shared::{
    Config, Index, Linspace, MAX_CONTAINERS, Model, OUTPUT_STATE_DIM, Real, Tolerance,
};
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
            Self::validate_uniform(slice)?;
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
            Self::validate_uniform(&seq)?;
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

impl BoundLinspace {
    /// The engine stores output times as (start, end, steps), so an array
    /// argument is rebuilt as a uniform grid between its endpoints. A
    /// non-uniform grid would silently diverge from the caller's array, so
    /// it is rejected instead.
    fn validate_uniform(times: &[Real]) -> PyResult<()> {
        let step = times[1] - times[0];
        let matches = |a: Real, b: Real| {
            (a - b).abs() <= 1e-9 * (a.abs() + b.abs() + 1.0)
        };
        for pair in times.windows(2) {
            if !matches(pair[1] - pair[0], step) {
                return Err(PyValueError::new_err(
                    "ts arrays must be uniformly spaced: output times are stored \
                     as (start, stop, num), so a non-uniform grid such as \
                     np.logspace would be silently replaced by \
                     np.linspace(ts[0], ts[-1], len(ts)). Pass (start, stop, num) \
                     for uniform output times.",
                ));
            }
        }
        Ok(())
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
    /// Every container the config has seen (added directly, listed as a
    /// dependency, or passed to run()), used by run() with no arguments.
    containers: Mutex<Vec<Container>>,
}

impl PyConfig {
    fn register(&self, container: &Container) {
        let mut containers = self.containers.lock().unwrap();
        if !containers
            .iter()
            .any(|c| c.dependency_label == container.dependency_label)
        {
            containers.push(container.clone());
        }
    }

    fn build_tree(&self, containers: Vec<Box<Container>>) -> (Model, InputFrame) {
        let input = vec_to_option_array_11(containers);
        let (x, y) = self.adjacency_matrix.build(input);
        (x, y)
    }
}

fn vec_to_option_array_11(mut v: Vec<Box<Container>>) -> Box<[Option<Box<Container>>; 11]> {
    if v.len() > 11 {
        v.truncate(11);
    }
    let mut out: Box<[Option<Box<Container>>; 11]> = Box::new([
        None, None, None, None, None, None, None, None, None, None, None,
    ]);

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
    #[pyo3(signature = (engine=None,method=None,variant=None,ts=None,tolerance=None))]
    fn new(
        engine: Option<PyEngine>,
        method: Option<PyMethod>,
        variant: Option<PyVariant>,
        ts: Option<BoundLinspace>,
        tolerance: Option<BoundTolerance>,
    ) -> Self {
        let thing = Self {
            inner: Config::new(
                engine.unwrap_or_default().into(),
                method.unwrap_or_default().into(),
                variant.unwrap_or_default().into(),
                Default::default(),
                ts.unwrap_or_default().0,
                tolerance.unwrap_or_default().0,
            ),
            adjacency_matrix: AdjacencyMatrix(0),
            containers: Mutex::new(Vec::new()),
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
        let mut containers: Vec<Box<Container>> = Vec::new();
        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            self.register(&container);
            containers.push(Box::new(container.clone()));
        }
        // run() with no arguments integrates every registered container.
        let mut selected = if args.is_empty() {
            let registered = self.containers.lock().unwrap();
            registered
                .iter()
                .map(|c| Box::new(c.clone()))
                .collect()
        } else {
            containers
        };
        // Dependency edges are keyed by creation order, so integrate in that
        // order regardless of how the caller lists the containers.
        selected.sort_unstable_by_key(|c| c.dependency_label);
        let group_sizes: Vec<Option<usize>> = selected
            .iter()
            .map(|c| c.state.as_ref().map(|s| s.num_particles as usize))
            .collect();
        if selected.len() > MAX_CONTAINERS {
            return Err(PyValueError::new_err(format!(
                "run() resolved {} containers but a model supports at most \
                 {MAX_CONTAINERS}",
                selected.len()
            )));
        }
        let (meal, istates) = self.build_tree(selected);
        let results = run_integration(self.inner, meal, istates)
            .map_err(|()| PyRuntimeError::new_err("integration failed"))?;

        // Align outputs with the containers passed to run(). Background
        // containers carry no particle state and contribute `None`; a group
        // container that produced no output means the engine/method/variant
        // combination is not implemented.
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(group_sizes.len());
        for (result, group_size) in results.0.iter().zip(group_sizes.iter()) {
            let num_particles = match group_size {
                Some(n) => *n,
                None => {
                    items.push(py.None());
                    continue;
                }
            };
            let state = match result {
                Some(state) => state,
                None => {
                    return Err(PyNotImplementedError::new_err(
                        "run() produced no output for a particle group: this \
                         engine/method/variant combination is not implemented \
                         (implemented: Engine.CPU or Engine.GPU with \
                         Method.DOPR54 and Variant.Compatible)",
                    ));
                }
            };
            let values = &state.data[..num_particles * OUTPUT_STATE_DIM];
            let array = PyArray1::from_slice(py, values)
                .reshape([num_particles, OUTPUT_STATE_DIM])?
                .into_any()
                .unbind();
            items.push(array);
        }
        PyList::new(py, items)
    }

    /// Register that ``node`` is integrated with ``requires`` as inputs.
    #[pyo3(signature = (node, *requires))]
    fn add<'py>(
        &mut self,
        _py: Python<'py>,
        node: Container,
        requires: &Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        let mut requires_labels: Vec<Index> = Vec::new();

        for i in 0..requires.len() {
            let obj = requires.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            self.register(&container);
            requires_labels.push(container.dependency_label);
        }
        self.register(&node);
        for x in requires_labels.iter() {
            self.adjacency_matrix
                .set(x.clone(), node.dependency_label, true);
        }
        Ok(())
    }

    #[pyo3(signature = (node, *args))]
    fn dependency<'py>(
        &mut self,
        py: Python<'py>,
        node: Container,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        py.import("warnings")?.call_method1(
            "warn",
            (
                "Config.dependency() is deprecated, use Config.add(node, *requires)",
                py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            ),
        )?;
        self.add(py, node, args)
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
fn drift_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyMethod>()?;
    m.add_class::<PyVariant>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<Container>()?;
    m.add_function(wrap_pyfunction!(container::test_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::part_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::bg_feature, m)?)?;

    Ok(())
}
