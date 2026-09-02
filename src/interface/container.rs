use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn};
use std::sync::{atomic::{AtomicU64, Ordering}, Mutex};
use shared::{CustomKeplerRecipe, CustomPlummerRecipe, Recipe, PotentialName, Index, MAX_CONTAINERS, Real};

static NEXT_DEP_LABEL: AtomicU64 = AtomicU64::new(0);

/// Labels of containers that have been deallocated, available for reuse.
/// MAX_CONTAINERS bounds the number of *live* containers; without recycling,
/// a process could only ever create 11 containers in total, which made
/// running several simulations in one session (e.g. benchmarks) impossible.
static FREE_LABELS: Mutex<Vec<Index>> = Mutex::new(Vec::new());

fn next_dep_label() -> Index {
    if let Ok(mut free) = FREE_LABELS.lock() {
        if let Some(label) = free.pop() {
            return label;
        }
    }
    let i: Index = NEXT_DEP_LABEL.fetch_add(1, Ordering::Relaxed) as Index;
    if i >= MAX_CONTAINERS {
        println!("Error!!!! To many containers")
    }
    i
}

fn release_dep_label(label: Index) {
    if let Ok(mut free) = FREE_LABELS.lock() {
        free.push(label);
    }
}

#[pyclass]
#[derive(Debug)]
pub struct Container {
    pub num_particles: Option<Index>,
    pub recipe: Option<PyRecipe>,
    pub state: Option<InputState>,
    pub dependency_label: Index,
    /// True only for the Python-owned original; clones made internally by
    /// `Config::run` share the label but must not release it on drop.
    pub owns_label: bool,
}

impl Clone for Container {
    fn clone(&self) -> Self {
        Container {
            num_particles: self.num_particles,
            recipe: self.recipe.clone(),
            state: self.state.clone(),
            dependency_label: self.dependency_label,
            owns_label: false,
        }
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        if self.owns_label {
            release_dep_label(self.dependency_label);
        }
    }
}

fn initialize_container<'py>(_py: Python<'py>, istate: PyReadonlyArrayDyn<Real>, recipe: Option<PyRecipe>) -> PyResult<Py<Container>> {
    let n = &istate.as_array().len();
    let state = InputState::from_py_array(&istate);

    let container = Container {
        num_particles: Some(*n),
        recipe: recipe,
        state: Some(state),
        dependency_label: next_dep_label(),
        owns_label: true,
    };
    Ok(Py::new(_py, container)?)
}


#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group<'py>(_py: Python<'py>, istate: PyReadonlyArrayDyn<Real>) -> PyResult<Py<Container>> {
    initialize_container(_py, istate, None)
}

#[pyfunction]
#[pyo3(signature = (potential,istate))]
pub fn part_group<'py>(_py: Python<'py>, potential: PyRecipe,istate:PyReadonlyArrayDyn<Real>) -> PyResult<Py<Container>> {
    let recipe:Option<PyRecipe> = match potential.inner {
        Recipe::Kepler(p) => Some(PyRecipe{inner:Recipe::CustomKepler(CustomKeplerRecipe{length:0,offset:0,division:0,final_time:0.,amp:p.amp,name:PotentialName::CustomKepler})}),
        Recipe::Plummer(p) => Some(PyRecipe{inner:Recipe::CustomPlummer(CustomPlummerRecipe{length:0,offset:0,division:0,final_time:0.,amp:p.amp,radius:p.radius,name:PotentialName::CustomPlummer})}),
        _ => {eprintln!("Bovy isn't implemented, or how have you passed in a custom Potential???"); None},
    };    
    initialize_container(_py, istate, recipe)
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature<'py>(_py: Python<'py>, potential: PyRecipe) -> Container {
    Container {
        num_particles: None,
        recipe: Some(potential),
        state: None,
        dependency_label: next_dep_label(),
        owns_label: true,
    }
}
