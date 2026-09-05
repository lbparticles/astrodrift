use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use shared::{
    CustomKeplerRecipe, CustomPlummerRecipe, Index, INPUT_STATE_DIM, MAX_CONTAINERS,
    MAX_PARTICLES, PotentialName, Real, Recipe,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DEP_LABEL: AtomicU64 = AtomicU64::new(0);

fn next_dep_label() -> PyResult<Index> {
    let i: Index = NEXT_DEP_LABEL.fetch_add(1, Ordering::Relaxed) as Index;
    if i >= MAX_CONTAINERS {
        return Err(PyValueError::new_err(format!(
            "too many containers: a model supports at most {MAX_CONTAINERS} \
             containers per process"
        )));
    }
    Ok(i)
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Container {
    /// Number of particles in this group (0 for background containers).
    #[pyo3(get)]
    pub num_particles: Option<Index>,
    pub recipe: Option<PyRecipe>,
    pub state: Option<InputState>,
    /// Creation-order id used to wire dependencies between containers.
    #[pyo3(get)]
    pub dependency_label: Index,
}

#[pymethods]
impl Container {
    fn __repr__(&self) -> String {
        let kind = match &self.recipe {
            Some(recipe) => format!("potential={:?}", recipe.inner),
            None => "test particles".to_string(),
        };
        format!(
            "Container(label={}, particles={:?}, {})",
            self.dependency_label, self.num_particles, kind
        )
    }
}

/// Validate an initial-state array and return the particle count.
///
/// Accepts an (N, 6) array or a flat length-6N array. Columns are the
/// phase-space coordinates [x, y, z, vx, vy, vz].
fn validate_istate(istate: &PyReadonlyArrayDyn<Real>) -> PyResult<Index> {
    let view = istate.as_array();
    let shape = view.shape();
    let (particles, well_formed) = match shape.len() {
        1 => (shape[0] / INPUT_STATE_DIM, shape[0] % INPUT_STATE_DIM == 0),
        2 => (shape[0], shape[1] == INPUT_STATE_DIM),
        _ => (0, false),
    };
    if !well_formed {
        return Err(PyValueError::new_err(format!(
            "istate must have shape (N, 6) with columns [x, y, z, vx, vy, vz] \
             (or a flat array of length 6N); got shape {shape:?}"
        )));
    }
    if particles == 0 {
        return Err(PyValueError::new_err(
            "istate must contain at least one particle",
        ));
    }
    if particles > MAX_PARTICLES {
        return Err(PyValueError::new_err(format!(
            "istate has {particles} particles but the current engine supports \
             at most {MAX_PARTICLES}; split the group into multiple containers"
        )));
    }
    Ok(particles)
}

fn initialize_container<'py>(
    _py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
    recipe: Option<PyRecipe>,
) -> PyResult<Py<Container>> {
    let n = validate_istate(&istate)?;
    let state = InputState::from_py_array(&istate);

    let container = Container {
        num_particles: Some(n),
        recipe: recipe,
        state: Some(state),
        dependency_label: next_dep_label()?,
    };
    Ok(Py::new(_py, container)?)
}

/// A group of test particles integrated in the given potential.
#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_particles<'py>(
    _py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    initialize_container(_py, istate, None)
}

#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group<'py>(
    py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    deprecate(py, "test_group", "test_particles")?;
    test_particles(py, istate)
}

/// A group of particles moving in the given potential.
#[pyfunction]
#[pyo3(signature = (potential, istate))]
pub fn particles<'py>(
    _py: Python<'py>,
    potential: PyRecipe,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    let recipe: Option<PyRecipe> = match potential.inner {
        Recipe::Kepler(p) => Some(PyRecipe {
            inner: Recipe::CustomKepler(CustomKeplerRecipe {
                length: 0,
                offset: 0,
                division: 0,
                final_time: 0.,
                amp: p.amp,
                name: PotentialName::CustomKepler,
            }),
        }),
        Recipe::Plummer(p) => Some(PyRecipe {
            inner: Recipe::CustomPlummer(CustomPlummerRecipe {
                length: 0,
                offset: 0,
                division: 0,
                final_time: 0.,
                amp: p.amp,
                radius: p.radius,
                name: PotentialName::CustomPlummer,
            }),
        }),
        _ => {
            return Err(PyNotImplementedError::new_err(
                "this potential cannot be attached to particles yet: Bovy is \
                 not implemented. Use Potential.kepler() or Potential.plummer().",
            ));
        }
    };
    initialize_container(_py, istate, recipe)
}

#[pyfunction]
#[pyo3(signature = (potential, istate))]
pub fn part_group<'py>(
    py: Python<'py>,
    potential: PyRecipe,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    deprecate(py, "part_group", "particles")?;
    particles(py, potential, istate)
}

/// A background potential feature shared by every particle group.
#[pyfunction]
#[pyo3(signature = (potential))]
pub fn background<'py>(_py: Python<'py>, potential: PyRecipe) -> PyResult<Py<Container>> {
    let container = Container {
        num_particles: None,
        recipe: Some(potential),
        state: None,
        dependency_label: next_dep_label()?,
    };
    Py::new(_py, container)
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature<'py>(py: Python<'py>, potential: PyRecipe) -> PyResult<Py<Container>> {
    deprecate(py, "bg_feature", "background")?;
    background(py, potential)
}

fn deprecate(py: Python<'_>, old: &str, new: &str) -> PyResult<()> {
    py.import("warnings")?.call_method1(
        "warn",
        (
            format!("{old}() is deprecated, use {new}()"),
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
        ),
    )?;
    Ok(())
}
