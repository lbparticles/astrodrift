use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use shared::{CustomKeplerRecipe, CustomPlummerRecipe, Index, PotentialName, Real, Recipe};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_IDENTITY: AtomicU64 = AtomicU64::new(0);

fn next_identity() -> u64 {
    NEXT_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Container {
    pub num_particles: Option<Index>,
    pub recipe: Option<PyRecipe>,
    pub state: Option<InputState>,
    // Object identity is stable; bounded graph labels are assigned per run.
    pub(super) identity: u64,
}

fn initialize_container<'py>(
    py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
    recipe: Option<PyRecipe>,
) -> PyResult<Py<Container>> {
    let n = istate.as_array().len();
    let state = InputState::from_py_array(&istate);

    let container = Container {
        num_particles: Some(n),
        recipe,
        state: Some(state),
        identity: next_identity(),
    };
    Py::new(py, container)
}

#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group<'py>(
    py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    initialize_container(py, istate, None)
}

#[pyfunction]
#[pyo3(signature = (potential,istate))]
pub fn part_group<'py>(
    py: Python<'py>,
    potential: PyRecipe,
    istate: PyReadonlyArrayDyn<Real>,
) -> PyResult<Py<Container>> {
    let recipe = match potential.inner {
        Recipe::Kepler(p) => PyRecipe {
            inner: Recipe::CustomKepler(CustomKeplerRecipe {
                length: 0,
                offset: 0,
                division: 0,
                final_time: 0.,
                amp: p.amp,
                name: PotentialName::CustomKepler,
            }),
        },
        Recipe::Plummer(p) => PyRecipe {
            inner: Recipe::CustomPlummer(CustomPlummerRecipe {
                length: 0,
                offset: 0,
                division: 0,
                final_time: 0.,
                amp: p.amp,
                radius: p.radius,
                name: PotentialName::CustomPlummer,
            }),
        },
        _ => {
            return Err(PyNotImplementedError::new_err(
                "only Kepler and Plummer potentials can currently be attached to particles",
            ));
        }
    };
    initialize_container(py, istate, Some(recipe))
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature(potential: PyRecipe) -> Container {
    Container {
        num_particles: None,
        recipe: Some(potential),
        state: None,
        identity: next_identity(),
    }
}
