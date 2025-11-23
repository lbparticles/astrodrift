// use numpy::PyArrayMethods;
// use numpy::{
//     // PyReadonlyArray2
//     // PyArray2,
// };
// use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList,PyDict, PyModule,PyTuple};
use pyo3::ffi::c_str;

mod flag;
mod recipe;
mod potential;
mod container;
mod engine;
mod variant;
mod method;

use container::IntegratorContainer;
use engine::PyEngine;
use variant::PyVariant;
use method::PyMethod;
use flag::Modern;
use potential::PyPotential;
use recipe::PyRecipe;


#[pyclass(name = "Config")]
pub struct PyConfig {
    inner: shared::Config,
}

// impl PyConfig {
//     fn parse_py_meal<'py>(py_meal:&Bound<'py, PyList>,meal:&mut shared::Meal)-> PyResult<()>{
//         let course_count = py_meal.len().min(shared::MAX_COURSES);
//         for (i, course_obj) in py_meal.iter().take(course_count).enumerate() {
//             let py_course: &Bound<PyList> = course_obj.cast::<PyList>().map_err(|_| {
//                 PyTypeError::new_err("Each course must be a list of Recipe objects")
//             })?;

//             let recipe_count = py_course.len().min(shared::MAX_RECIPES);
//             if py_course.len() > shared::MAX_RECIPES {
//                 eprintln!(
//                     "Warning: Course {} has more than MAX_RECIPES ({}). Extra items ignored.",
//                     i, shared::MAX_RECIPES
//                 );
//             }

//             for (j, recipe_obj) in py_course.iter().take(recipe_count).enumerate() {
//                 let r: Py<PyRecipe> = recipe_obj
//                     .extract()
//                     .map_err(|_| PyTypeError::new_err("Expected a Recipe in course list"))?;

//                 Python::attach(|py| -> PyResult<()> {
//                     let py_recipe_borrow = r.borrow(py);
//                     meal[i][j] = py_recipe_borrow.inner;
//                     Ok(())
//                 })?;
//             }
//         }
//         if py_meal.len() > shared::MAX_COURSES {
//             eprintln!(
//                 "Warning: Received {} courses, only first {} used.",
//                 py_meal.len(),
//                 shared::MAX_COURSES
//             );
//         }
//         Ok(())
//     }
//     fn parse_py_istates<'py>(py_istates:&Bound<'py, PyList>,istates:&mut shared::IStates)-> PyResult<()>{
//         let state_count = py_istates.len().min(shared::MAX_STATES);
//         if py_istates.len() > shared::MAX_STATES {
//             eprintln!(
//                 "Warning: Received {} initial states, only first {} used.",
//                 py_istates.len(),
//                 shared::MAX_STATES
//             );
//         }

//         for (k, arr_obj) in py_istates.iter().take(state_count).enumerate() {
//             let arr_bound = arr_obj.cast::<PyArray2<f64>>()?;
//             let readonly = arr_bound.readonly();
//             let view = readonly.as_array();

//             let flat: Vec<f64> = view.iter().copied().collect();
//             let copied_len = flat.len().min(shared::ILENGTH);

//             if flat.len() > shared::ILENGTH {
//                 eprintln!(
//                     "Warning: Initial condition {} exceeds ILENGTH ({}), truncating.",
//                     k, shared::ILENGTH
//                 );
//             }

//             istates[k][..copied_len].copy_from_slice(&flat[..copied_len]);
//         }
//         Ok(())
    
//     }
// }

impl PyConfig {
    fn build_tree(_containers:Vec<IntegratorContainer>)->(shared::Meal,shared::IStates){
       ([[shared::Recipe::default();shared::MAX_RECIPES];shared::MAX_COURSES],[[0.0; shared::ILENGTH]; shared::MAX_STATES]) 
    }
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new(
        engine: &PyEngine,
        method: &PyMethod,
        variant: &PyVariant,
        ts: shared::Linspace,
        tolerance: shared::Tolerance,
        part_num: shared::Index,
    ) -> Self {
        Self {
            inner: shared::Config::new(
                engine.inner.clone(),
                method.inner.clone(),
                variant.inner.clone(),
                ts,
                tolerance,
                part_num,
            ),
        }
    }


    fn run<'py>(&self,py:Python<'py>,args: &Bound<'py,PyTuple>)->PyResult<Bound<'py, PyList>>{
        let mut containers: Vec<IntegratorContainer> = Vec::new();

        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<IntegratorContainer> = obj.extract()?;
            containers.push(container.clone());
        }
        let (meal,istates) = Self::build_tree(containers);

        let results = self.inner.run(meal, istates);
        
    	PyList::new(py, results.iter().map(|a| a.to_vec()))
    }
    
    // fn _run<'py>(
    //     &self,
    //     py: Python<'py>,
    //     py_meal: &Bound<'py, PyList>,
    //     py_istates: &Bound<'py, PyList>,
    // ) -> Bound<'py, PyList> {
    //     // ----- Parse recipes: List[List[PyRecipe]] -----
    //     let mut meal: shared::Meal = [[shared::Recipe::default();shared::MAX_RECIPES];shared::MAX_COURSES]; 
    //     let _ = Self::parse_py_meal(py_meal,&mut meal);

    //     // ----- Parse arrays: List[np.ndarray] -----


    //     let mut istates: shared::IStates = [[0.0; shared::ILENGTH]; shared::MAX_STATES];
    //     let _ = Self::parse_py_istates(py_istates,&mut istates);

    //     // ----- Run your Rust core logic -----
    //     let results = self.inner.run(meal, istates);
        
    // 	let py_results = PyList::new(py, results.iter().map(|a| a.to_vec()));

    //     // Return as owned Py<PyList>
    //     py_results.expect("REASON")
        
    // }
}


//
// Python Module Declaration
//
#[pymodule]
fn drift_rs(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyMethod>()?;
    m.add_class::<PyVariant>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyPotential>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<Modern>()?;

    // Define enum.Flag in Python
    let locals = PyDict::new(py);
    py.run(c_str!(
        r#"
import enum

class ModernFlag(enum.Flag):
    NONE        = 0
    READ        = 1 << 0
    WRITE       = 1 << 1
    EXECUTE     = 1 << 2
    DELETE      = 1 << 3
    READ_WRITE  = READ | WRITE
    FULL_ACCESS = READ | WRITE | EXECUTE | DELETE
"#),
        None,
        Some(&locals),
    )?;

    let py_enum = locals.get_item("ModernFlag").unwrap();
    m.add("ModernFlag", py_enum)?;

    Ok(())
}
