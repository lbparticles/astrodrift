use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyDeprecationWarning, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PyTuple};

mod container;
mod engine;
mod method;
mod recipe;
mod selector;
mod variant;

use crate::integrators::run_integration;
use crate::tree::{AdjacencyMatrix, IntegrationPlan};
pub use container::Container;
use engine::PyEngine;
use method::PyMethod;
pub use recipe::PyRecipe;
use shared::{
    Config, INPUT_STATE_DIM, Linspace, MAX_MODEL_COMPONENTS, MAX_OUTPUT_TIMES, MAX_STATES, Real,
    Tolerance,
};
use variant::PyVariant;

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

// Python accepts conventional positive tolerances; reference kernels consume
// their natural logarithms.
#[derive(Default, Clone, Debug)]
pub struct PyTolerance(pub Tolerance);
impl<'a, 'py> FromPyObject<'a, 'py> for PyTolerance {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tup) = obj.cast::<PyTuple>()
            && tup.len() == 2
        {
            let rtol: f64 = tup.get_item(0)?.extract()?;
            let atol: f64 = tup.get_item(1)?.extract()?;
            return Self::new(rtol, atol);
        }

        // Accept single float for convenience
        if let Ok(val) = obj.extract::<f64>() {
            return Self::new(val, val);
        }

        Err(PyValueError::new_err(
            "Expected (rtol, atol) tuple or single float tolerance value",
        ))
    }
}

impl PyTolerance {
    fn new(rtol: Real, atol: Real) -> PyResult<Self> {
        if !rtol.is_finite() || !atol.is_finite() || rtol <= 0.0 || atol <= 0.0 {
            return Err(PyValueError::new_err(
                "rtol and atol must be finite and greater than zero",
            ));
        }

        Ok(Self(Tolerance::from_linear(rtol, atol)))
    }
}

/// Configuration and registered container graph for an integration.
///
/// The default selectors are ``Engine.CPU``, ``Method.DOPR54``, and
/// ``Variant.Compatible``. Register force relationships with :meth:`add`,
/// then execute the complete model with :meth:`run`.
#[pyclass(name = "Config")]
#[derive(Debug)]
pub struct PyConfig {
    inner: Config,
    // Edges are (potential source, integrated dependent) stable identities.
    // build_tree assigns dense labels only for a particular run.
    dependencies: Vec<(u64, u64)>,
    // add() deliberately retains model components for zero-argument run().
    containers: Vec<Container>,
}

impl PyConfig {
    fn register(containers: &mut Vec<Container>, container: &Container) {
        if !containers
            .iter()
            .any(|candidate| candidate.identity == container.identity)
        {
            containers.push(container.clone());
        }
    }

    /// Depth-first reachability over the edge list.
    ///
    /// This is O(VE + V^2) time and O(V) space because visited identities use
    /// a linear set. Both are deliberately simple with V bounded at 11.
    fn dependency_path_exists(&self, start: u64, target: u64) -> bool {
        let mut pending = vec![start];
        let mut visited = Vec::new();

        while let Some(current) = pending.pop() {
            if current == target {
                return true;
            }
            if visited.contains(&current) {
                continue;
            }
            visited.push(current);
            pending.extend(
                self.dependencies
                    .iter()
                    .filter_map(|&(source, dependent)| (source == current).then_some(dependent)),
            );
        }

        false
    }

    /// Validate all incoming edges before mutating the graph.
    ///
    /// Every proposed edge ends at `node`, so it creates a cycle exactly when
    /// `node` already reaches that requirement. For R requirements this is
    /// O(R(VE + V^2)); R and V are both at most 11.
    fn add_dependencies(&mut self, node: u64, requires: Vec<u64>) -> PyResult<()> {
        if requires.is_empty() {
            return Err(PyValueError::new_err(
                "add() requires at least one dependency",
            ));
        }
        if requires
            .iter()
            .any(|&dependency| self.dependency_path_exists(node, dependency))
        {
            return Err(PyValueError::new_err(
                "adding these dependencies would create a cycle",
            ));
        }

        for dependency in requires {
            let edge = (dependency, node);
            if !self.dependencies.contains(&edge) {
                self.dependencies.push(edge);
            }
        }
        Ok(())
    }

    fn validate_registered_model(&self) -> PyResult<()> {
        if self.containers.iter().any(|container| {
            container.state.is_some()
                && !self
                    .dependencies
                    .iter()
                    .any(|&(_, dependent)| dependent == container.identity)
        }) {
            return Err(PyValueError::new_err(
                "each particle container must be registered with add(node, *requires) before integration",
            ));
        }
        Ok(())
    }

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
            let node_label = identity_by_label
                .iter()
                .position(|&identity| identity == Some(node))
                .ok_or_else(|| PyRuntimeError::new_err("registered node is missing"))?;
            let dependency_label = identity_by_label
                .iter()
                .position(|&identity| identity == Some(dependency))
                .ok_or_else(|| PyRuntimeError::new_err("registered dependency is missing"))?;
            adjacency_matrix.set(dependency_label, node_label, true);
        }

        adjacency_matrix.build(containers_by_label).ok_or_else(|| {
            PyValueError::new_err("container dependencies must form an acyclic graph")
        })
    }
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (engine=None,method=None,variant=None,ts=None,tolerance=None))]
    fn new(
        engine: Option<PyEngine>,
        method: Option<PyMethod>,
        variant: Option<PyVariant>,
        ts: Option<PyLinspace>,
        tolerance: Option<PyTolerance>,
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
            dependencies: Vec::new(),
            containers: Vec::new(),
        };
        println!("newpyconfig");
        thing
    }

    /// Integrate the registered model.
    ///
    /// Results follow first-registration order. Each state-bearing container
    /// produces a float64 array shaped ``(time, particle, 6)``; stationary
    /// background containers produce ``None``.
    #[pyo3(signature = ())]
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.validate_registered_model()?;
        let output_identities: Vec<u64> = self
            .containers
            .iter()
            .map(|container| container.identity)
            .collect();
        let has_state: Vec<bool> = self
            .containers
            .iter()
            .map(|container| container.state.is_some())
            .collect();
        let plan = self.build_tree(self.containers.clone())?;
        let IntegrationPlan {
            model,
            input_frame,
            container_identity_by_stage,
        } = plan;
        let results = run_integration(self.inner, model, input_frame)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;

        let mut items: Vec<Option<Py<PyAny>>> = core::iter::repeat_with(|| None)
            .take(has_state.len())
            .collect();
        for (result, container_identity) in results.0.iter().zip(container_identity_by_stage) {
            let Some(state) = result else { continue };
            let Some(container_identity) = container_identity else {
                return Err(PyRuntimeError::new_err(
                    "integration returned an output without a source container",
                ));
            };
            let Some(output_index) = output_identities
                .iter()
                .position(|&identity| identity == container_identity)
            else {
                return Err(PyRuntimeError::new_err(
                    "integration returned an output for an unknown container",
                ));
            };
            let item = &mut items[output_index];
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
                     are Engine.CPU or Engine.GPU with Method.DOPR54 or Method.DOP853 and \
                     Variant.Compatible",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    /// Register that ``node`` is integrated with ``requires`` as inputs.
    ///
    /// This adds directed force-source edges from every item in ``requires``
    /// to ``node`` and registers all supplied containers with this config.
    #[pyo3(signature = (node, *requires))]
    fn add<'py>(&mut self, node: Container, requires: &Bound<'py, PyTuple>) -> PyResult<()> {
        let mut dependency_ids = Vec::with_capacity(requires.len());
        let mut dependency_containers = Vec::with_capacity(requires.len());
        for i in 0..requires.len() {
            let obj = requires.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            if !dependency_ids.contains(&container.identity) {
                dependency_ids.push(container.identity);
                dependency_containers.push(container.clone());
            }
        }
        let mut containers = self.containers.clone();
        Self::register(&mut containers, &node);
        for dependency in &dependency_containers {
            Self::register(&mut containers, dependency);
        }
        if containers.len() > MAX_MODEL_COMPONENTS {
            return Err(PyValueError::new_err(format!(
                "add() would register {} containers but a model supports at most {MAX_MODEL_COMPONENTS}",
                containers.len()
            )));
        }

        self.add_dependencies(node.identity, dependency_ids)?;
        self.containers = containers;
        Ok(())
    }

    /// Deprecated alias for :meth:`add`.
    #[pyo3(signature = (node, *requires))]
    fn dependency<'py>(
        &mut self,
        py: Python<'py>,
        node: Container,
        requires: &Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        PyErr::warn(
            py,
            &py.get_type::<PyDeprecationWarning>(),
            c"Config.dependency() is deprecated; use Config.add(node, *requires)",
            1,
        )?;
        self.add(node, requires)
    }

    /// Return a human-readable summary of this configuration.
    #[pyo3(signature = ())]
    fn info(&self) -> String {
        let settings = self.inner.settings;
        format!(
            "Config(engine={:?}, method={:?}, variant={:?}, times=({}, {}, {}), \
             tolerance=(rtol={:.6e}, atol={:.6e}), containers={}, dependencies={})",
            self.inner.engine,
            self.inner.method,
            self.inner.variant,
            settings.ts.start,
            settings.ts.end,
            settings.ts.steps,
            settings.tolerance.rtol.exp(),
            settings.tolerance.atol.exp(),
            self.containers.len(),
            self.dependencies.len(),
        )
    }
}

/// Low-level bindings for configuration, potentials, containers, and their
/// constructors. The public selector enums are provided by the ``drift``
/// Python package.
#[pymodule]
fn drift_rs(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<Container>()?;
    m.add_function(wrap_pyfunction!(container::test_particles, m)?)?;
    m.add_function(wrap_pyfunction!(container::particles, m)?)?;
    m.add_function(wrap_pyfunction!(container::background, m)?)?;

    Ok(())
}
