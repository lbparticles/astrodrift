use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use shared::{
    CustomKeplerRecipe, CustomPlummerRecipe, Index, INPUT_STATE_DIM, MAX_CONTAINERS,
    PotentialName, Real, Recipe,
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
#[derive(Clone)]
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

fn initialize_container<'py>(
    _py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
    recipe: Option<PyRecipe>,
) -> PyResult<Py<Container>> {
    // num_particles is a particle count, not an element count: istate holds
    // INPUT_STATE_DIM (6) numbers per particle.
    let n = istate.as_array().len() / INPUT_STATE_DIM;
    let state = InputState::from_py_array(&istate);

    let container = Container {
        num_particles: Some(n),
        recipe: recipe,
        state: Some(state),
        dependency_label: next_dep_label()?,
    };
    Ok(Py::new(_py, container)?)
}

#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group<'py>(
    _py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    initialize_container(_py, istate, None)
}

#[pyfunction]
#[pyo3(signature = (potential,istate))]
pub fn part_group<'py>(
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
#[pyo3(signature = (potential))]
pub fn bg_feature<'py>(_py: Python<'py>, potential: PyRecipe) -> PyResult<Py<Container>> {
    let container = Container {
        num_particles: None,
        recipe: Some(potential),
        state: None,
        dependency_label: next_dep_label()?,
    };
    Py::new(_py, container)
}
