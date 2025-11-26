use crate::interface::recipe::PyRecipe;
use crate::state::InputState;
use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn};
use std::sync::atomic::{AtomicU64, Ordering};
use shared::{CustomKeplerRecipe,CustomPlummerRecipe};
use shared::{Recipe,PotentialName};

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
    pub state: Option<InputState>,
    pub dependency_label: shared::Index,
}

fn initialize_container<'py>(_py: Python<'py>, istate: PyReadonlyArrayDyn<shared::Real>, recipe: Option<PyRecipe>) -> PyResult<Py<Container>> {
    let state = InputState::from_py_array(&istate);

    let container = Container {
        recipe: recipe,
        state: Some(state),
        dependency_label: next_dep_label(),
    };
    Ok(Py::new(_py, container)?)
}


#[pyfunction]
#[pyo3(signature = (istate))]
pub fn test_group<'py>(_py: Python<'py>, istate: PyReadonlyArrayDyn<shared::Real>) -> PyResult<Py<Container>> {
    initialize_container(_py, istate, None)
}

#[pyfunction]
#[pyo3(signature = (potential,istate))]
pub fn part_group<'py>(_py: Python<'py>, potential: PyRecipe,istate:PyReadonlyArrayDyn<shared::Real>) -> PyResult<Py<Container>> {
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
        recipe: Some(potential),
        state: None,
        dependency_label: next_dep_label(),
    }
}
