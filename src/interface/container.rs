use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use shared::{
    CustomKeplerRecipe, CustomPlummerRecipe, INPUT_STATE_DIM, Index, MAX_PARTICLES, PotentialName,
    Real, Recipe,
};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

static NEXT_IDENTITY: AtomicU64 = AtomicU64::new(0);

fn next_identity() -> u64 {
    NEXT_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct Container {
    /// Number of particles in this group (`None` for background containers).
    #[pyo3(get)]
    pub num_particles: Option<Index>,
    pub recipe: Option<PyRecipe>,
    // Container registration and integration plans share this immutable buffer.
    pub state: Option<Arc<InputState>>,
    // Object identity is stable; bounded graph labels are assigned per run.
    pub(crate) identity: u64,
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
    py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
    recipe: Option<PyRecipe>,
) -> PyResult<Py<Container>> {
    let num_particles = validate_istate(&istate)?;
    let state = InputState::from_py_array(&istate);
    debug_assert_eq!(state.num_particles, num_particles);

    let container = Container {
        num_particles: Some(num_particles),
        recipe,
        state: Some(Arc::new(state)),
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
