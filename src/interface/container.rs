use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use shared::{
    CustomKeplerRecipe, CustomPlummerRecipe, Index, MAX_CONTAINERS, PotentialName, Real, Recipe,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DEP_LABEL: AtomicU64 = AtomicU64::new(0);

fn next_dep_label() -> Index {
    // Labels grow by one per container; the counter can never realistically
    // approach `usize::MAX`, so fall back to it instead of truncating.
    let i: Index =
        Index::try_from(NEXT_DEP_LABEL.fetch_add(1, Ordering::Relaxed)).unwrap_or(Index::MAX);
    if i >= MAX_CONTAINERS {
        println!("Error!!!! To many containers");
    }
    i
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Container {
    pub num_particles: Option<Index>,
    pub recipe: Option<PyRecipe>,
    pub state: Option<InputState>,
    pub dependency_label: Index,
}

fn initialize_container(
    py: Python<'_>,
    istate: &PyReadonlyArrayDyn<Real>,
    recipe: Option<PyRecipe>,
) -> PyResult<Py<Container>> {
    let state = InputState::from_py_array(istate);

    let container = Container {
        num_particles: Some(istate.as_array().len()),
        recipe,
        state: Some(state),
        dependency_label: next_dep_label(),
    };
    Py::new(py, container)
}

// `#[pyfunction]` arguments are extracted from Python objects by value by
// design; pyo3 does not support `&PyReadonlyArrayDyn` extraction.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group(py: Python<'_>, istate: PyReadonlyArrayDyn<Real>) -> PyResult<Py<Container>> {
    initialize_container(py, &istate, None)
}

// `#[pyfunction]` arguments are extracted from Python objects by value by
// design; pyo3 does not support `&PyReadonlyArrayDyn` extraction.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (potential,istate))]
pub fn part_group(
    py: Python<'_>,
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
            eprintln!("Bovy isn't implemented, or how have you passed in a custom Potential???");
            None
        }
    };
    initialize_container(py, &istate, recipe)
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature(_py: Python<'_>, potential: PyRecipe) -> Container {
    Container {
        num_particles: None,
        recipe: Some(potential),
        state: None,
        dependency_label: next_dep_label(),
    }
}
