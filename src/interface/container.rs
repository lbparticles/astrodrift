use crate::interface::potential::PyPotential;
use crate::interface::recipe::PyRecipe;
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArrayDyn};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DEP_LABEL: AtomicU64 = AtomicU64::new(0);

fn next_dep_label() -> shared::Index {
    let i: shared::Index = NEXT_DEP_LABEL.fetch_add(1, Ordering::Relaxed) as shared::Index;
    if i >= shared::MAX_CONTAINERS {
        println!("Error!!!! To many containers")
    }
    i
}

#[pyclass]
#[derive(Clone)]
pub struct Container {
    pub recipe: Option<PyRecipe>,
    pub state: Option<shared::InputState>,
    pub dependency_label: shared::Index,
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Container")
            .field("recipe", &self.recipe.as_ref().map(|_| "<PyRecipe>"))
            .field("state_len", &self.state.as_ref().map(|s| s.len()))
            .field("dependency_label", &self.dependency_label)
            .finish()
    }
}


#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group<'py>(_py: Python<'py>, istate:PyReadonlyArrayDyn<shared::Real>) -> Container {
    let mut boxed: shared::InputState = Box::new([0.0; shared::INPUT_LENGTH]);

    let mut i = 0usize;
    for v in istate.as_array().iter().copied() {
        if i >= shared::INPUT_LENGTH {
            break; // truncate
        }
        boxed[i] = v;
        i += 1;
    }
    Container {
        recipe: None,
        state: Some(boxed),
        dependency_label: next_dep_label(),
    }
}

#[pyfunction]
#[pyo3(signature = (potential,istate))]
pub fn part_group<'py>(_py: Python<'py>, potential: PyPotential,istate:PyReadonlyArrayDyn<shared::Real>) -> Container {
    let mut boxed: shared::InputState = Box::new([0.0; shared::INPUT_LENGTH]);

    let mut i = 0usize;
    for v in istate.as_array().iter().copied() {
        if i >= shared::INPUT_LENGTH {
            break; // truncate
        }
        boxed[i] = v;
        i += 1;
    }
    Container {
        recipe: Some(potential.into()),
        state: Some(boxed),
        dependency_label: next_dep_label(),
    }
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature<'py>(_py: Python<'py>, potential: PyPotential) -> Container {
    Container {
        recipe: Some(potential.into()),
        state: None,
        dependency_label: next_dep_label(),
    }
}
