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

/// Bulge force LUT for the MW2014 composite ("previous method" layout).
#[derive(Debug)]
pub struct TableData {
    pub values: Vec<Real>,
    pub r_min: Real,
    pub dr: Real,
}

/// Quintic-origin coefficient table for the annulus perturber stack
/// (18 doubles per (particle, division); load_data-branch layout).
#[derive(Debug)]
pub struct AnnulusData {
    pub coeffs: Vec<Real>,
    pub n_gmc: usize,
    pub division: usize,
    pub final_time: Real,
    pub plummer_amp: Real,
    pub plummer_b: Real,
}

#[pyclass]
#[derive(Debug)]
pub struct Container {
    pub num_particles: Option<Index>,
    pub recipe: Option<PyRecipe>,
    pub state: Option<InputState>,
    /// Background-potential force LUT (Bovy/MW2014 recipes only).
    pub table: Option<std::sync::Arc<TableData>>,
    /// Annulus perturber-stack coefficients (on the test-particle container).
    pub annulus: Option<std::sync::Arc<AnnulusData>>,
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
            table: self.table.clone(),
            annulus: self.annulus.clone(),
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
    initialize_container_with_annulus(_py, istate, recipe, None)
}

fn initialize_container_with_annulus<'py>(
    _py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
    recipe: Option<PyRecipe>,
    annulus: Option<std::sync::Arc<AnnulusData>>,
) -> PyResult<Py<Container>> {
    let n = &istate.as_array().len();
    let state = InputState::from_py_array(&istate);

    let container = Container {
        num_particles: Some(*n),
        recipe: recipe,
        state: Some(state),
        table: None,
        annulus,
        dependency_label: next_dep_label(),
        owns_label: true,
    };
    Ok(Py::new(_py, container)?)
}


#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (istate, annulus_coeffs=None, n_gmc=None, division=None,
                    final_time=None, plummer_amp=None, plummer_b=None))]
pub fn test_group<'py>(
    _py: Python<'py>,
    istate: PyReadonlyArrayDyn<Real>,
    annulus_coeffs: Option<Vec<Real>>,
    n_gmc: Option<usize>,
    division: Option<usize>,
    final_time: Option<Real>,
    plummer_amp: Option<Real>,
    plummer_b: Option<Real>,
) -> PyResult<Py<Container>> {
    let annulus = match (annulus_coeffs, n_gmc, division, final_time, plummer_amp, plummer_b) {
        (Some(coeffs), Some(n_gmc), Some(division), Some(final_time),
         Some(plummer_amp), Some(plummer_b)) => {
            let expected = 18 * n_gmc * division;
            if coeffs.len() != expected {
                panic!(
                    "annulus_coeffs length {} != 18 * n_gmc * division = {expected}",
                    coeffs.len()
                );
            }
            Some(std::sync::Arc::new(AnnulusData {
                coeffs,
                n_gmc,
                division,
                final_time,
                plummer_amp,
                plummer_b,
            }))
        }
        (None, None, None, None, None, None) => None,
        _ => panic!(
            "annulus requires all of: annulus_coeffs, n_gmc, division, \
             final_time, plummer_amp, plummer_b"
        ),
    };
    initialize_container_with_annulus(_py, istate, None, annulus)
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

/// Create a background-potential container.
///
/// For the MW2014 composite (`Potential.bovy()`), the bulge is integrated
/// through a radial force LUT ("previous method"): pass the sampled table,
/// its start radius and its uniform spacing:
///   bg_feature(Potential.bovy(), ar_table=table, r_min=1e-3, dr=...)
#[pyfunction]
#[pyo3(signature = (potential, ar_table=None, r_min=None, dr=None))]
pub fn bg_feature<'py>(
    _py: Python<'py>,
    potential: PyRecipe,
    ar_table: Option<Vec<Real>>,
    r_min: Option<Real>,
    dr: Option<Real>,
) -> Container {
    let is_bovy = matches!(potential.inner, shared::Recipe::Bovy(_));
    let table = match (is_bovy, ar_table, r_min, dr) {
        (true, Some(values), Some(r_min), Some(dr)) => {
            if values.len() < 2 {
                panic!("bovy ar_table must contain at least two entries");
            }
            Some(std::sync::Arc::new(TableData { values, r_min, dr }))
        }
        (true, None, None, None) => panic!(
            "bovy background requires the bulge force LUT: \
             bg_feature(Potential.bovy(), ar_table=..., r_min=..., dr=...)"
        ),
        _ => None,
    };
    Container {
        num_particles: None,
        recipe: Some(potential),
        state: None,
        table,
        annulus: None,
        dependency_label: next_dep_label(),
        owns_label: true,
    }
}
