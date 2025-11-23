use crate::interface::potential::PyPotential;
use crate::interface::recipe::PyRecipe;
use std::sync::atomic::{AtomicU64, Ordering};
use pyo3::prelude::*;

static NEXT_DEP_LABEL: AtomicU64 = AtomicU64::new(0);

fn next_dep_label() -> shared::Index {
    let i: shared::Index = NEXT_DEP_LABEL.fetch_add(1, Ordering::Relaxed) as shared::Index;
    if i >= shared::MAX_CONTAINERS{
        println!("Error!!!! To many containers")
    }
    i
}

#[pyclass]
#[derive(Debug,Clone)]
pub struct Container {
    pub recipe: Option<PyRecipe>,
    pub state: Option<shared::IState>,
    pub dependency_label: shared::Index,
}


#[pyfunction]
#[pyo3(signature = ())]
pub fn test_group<'py>(_py: Python<'py>) -> Container {
    Container {
        recipe: None,
        state: None,
        dependency_label:next_dep_label(),
    }
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn part_group<'py>(_py: Python<'py>, potential: PyPotential) -> Container {
    Container {
        recipe: Some(potential.into()),
        state: None,
        dependency_label:next_dep_label(),
    }
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature<'py>(_py: Python<'py>, potential: PyPotential) -> Container {
    Container {
        recipe: Some(potential.into()),
        state: None,
        dependency_label:next_dep_label(),
    }
}
