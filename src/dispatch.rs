use numpy::PyArrayMethods;
use numpy::{
    // PyReadonlyArray2
    PyArray2,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList,PyDict, PyModule};
use pyo3::ffi::c_str;


use shared::{
    Config,
    Engine,
    ModernFlags,
    Index,
    IndexParams,
    Linspace,
    Method,
    Potential,
    RealParams,
    Recipe,
    Tolerance,
    Variant,
    // Real,
};

//
// Wrap the core enums
//
#[pyclass(name = "Engine")]
#[derive(Clone)]
pub struct PyEngine {
    inner: Engine,
}
#[pymethods]
impl PyEngine {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "GPU" => Engine::GPU,
            "CPU" => Engine::CPU,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Engine")),
        };
        Ok(Self { inner })
    }
}

#[pyclass(name = "Method")]
#[derive(Clone)]
pub struct PyMethod {
    inner: Method,
}
#[pymethods]
impl PyMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "DOP853" => Method::DOP853,
            "DOPR54" => Method::DOPR54,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Method")),
        };
        Ok(Self { inner })
    }
}

#[pyclass(name = "Variant")]
#[derive(Clone)]
pub struct PyVariant {
    inner: Variant,
}
#[pymethods]
impl PyVariant {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "Modern" => Variant::Modern,
            "Compatible" => Variant::Compatible,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid Variant")),
        };
        Ok(Self { inner })
    }
}

//
// Wrap Config
//
#[pyclass(name = "Recipe")]
pub struct PyRecipe {
    inner: Recipe,
}

#[pymethods]
impl PyRecipe {
    #[new]
    fn new(
        real_params: RealParams,
        index_params: IndexParams,
        potential: Potential,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Recipe {
                real_params,
                index_params,
                potential,
            },
        })
    }
}

#[pyclass(name = "Config")]
pub struct PyConfig {
    inner: Config,
}

impl PyConfig {
    fn parse_py_meal<'py>(py_meal:&Bound<'py, PyList>,meal:&mut shared::Meal)-> PyResult<()>{
        let course_count = py_meal.len().min(shared::MAX_COURSES);
        for (i, course_obj) in py_meal.iter().take(course_count).enumerate() {
            let py_course: &Bound<PyList> = course_obj.cast::<PyList>().map_err(|_| {
                PyTypeError::new_err("Each course must be a list of Recipe objects")
            })?;

            let recipe_count = py_course.len().min(shared::MAX_RECIPES);
            if py_course.len() > shared::MAX_RECIPES {
                eprintln!(
                    "Warning: Course {} has more than MAX_RECIPES ({}). Extra items ignored.",
                    i, shared::MAX_RECIPES
                );
            }

            for (j, recipe_obj) in py_course.iter().take(recipe_count).enumerate() {
                let r: Py<PyRecipe> = recipe_obj
                    .extract()
                    .map_err(|_| PyTypeError::new_err("Expected a Recipe in course list"))?;

                Python::attach(|py| -> PyResult<()> {
                    let py_recipe_borrow = r.borrow(py);
                    meal[i][j] = py_recipe_borrow.inner;
                    Ok(())
                })?;
            }
        }
        if py_meal.len() > shared::MAX_COURSES {
            eprintln!(
                "Warning: Received {} courses, only first {} used.",
                py_meal.len(),
                shared::MAX_COURSES
            );
        }
        Ok(())
    }
    fn parse_py_istates<'py>(py_istates:&Bound<'py, PyList>,istates:&mut shared::IStates)-> PyResult<()>{
        let state_count = py_istates.len().min(shared::MAX_STATES);
        if py_istates.len() > shared::MAX_STATES {
            eprintln!(
                "Warning: Received {} initial states, only first {} used.",
                py_istates.len(),
                shared::MAX_STATES
            );
        }

        for (k, arr_obj) in py_istates.iter().take(state_count).enumerate() {
            let arr_bound = arr_obj.cast::<PyArray2<f64>>()?;
            let readonly = arr_bound.readonly();
            let view = readonly.as_array();

            let flat: Vec<f64> = view.iter().copied().collect();
            let copied_len = flat.len().min(shared::ILENGTH);

            if flat.len() > shared::ILENGTH {
                eprintln!(
                    "Warning: Initial condition {} exceeds ILENGTH ({}), truncating.",
                    k, shared::ILENGTH
                );
            }

            istates[k][..copied_len].copy_from_slice(&flat[..copied_len]);
        }
        Ok(())
    
    }
}

#[pymethods]
impl PyConfig {
    #[new]
    fn new(
        engine: &PyEngine,
        method: &PyMethod,
        variant: &PyVariant,
        ts: Linspace,
        tolerance: Tolerance,
        part_num: Index,
    ) -> Self {
        Self {
            inner: Config::new(
                engine.inner.clone(),
                method.inner.clone(),
                variant.inner.clone(),
                ts,
                tolerance,
                part_num,
            ),
        }
    }
    fn run<'py>(
        &self,
        py: Python<'py>,
        py_meal: &Bound<'py, PyList>,
        py_istates: &Bound<'py, PyList>,
    ) -> Bound<'py, PyList> {
        // ----- Parse recipes: List[List[PyRecipe]] -----
        let mut meal: shared::Meal = [[shared::Recipe::default();shared::MAX_RECIPES];shared::MAX_COURSES]; 
        let _ = Self::parse_py_meal(py_meal,&mut meal);

        // ----- Parse arrays: List[np.ndarray] -----


        let mut istates: shared::IStates = [[0.0; shared::ILENGTH]; shared::MAX_STATES];
        let _ = Self::parse_py_istates(py_istates,&mut istates);

        // ----- Run your Rust core logic -----
        let results = self.inner.run(meal, istates);
        
    	let py_results = PyList::new(py, results.iter().map(|a| a.to_vec()));

        // Return as owned Py<PyList>
        py_results.expect("REASON")
        
    }
}


#[pyclass]
#[derive(Clone)]
struct Modern {
    flags: ModernFlags,
}

#[pymethods]
impl Modern {
    #[new]
    fn new() -> Self {
        Self {
            flags: ModernFlags::NONE,
        }
    }

    fn add(&mut self, value: Index) {
        if let Some(flag) = ModernFlags::from_bits(value) {
            self.flags.set(flag);
        }
    }

    fn has(&self, value: Index) -> bool {
        ModernFlags::from_bits(value)
            .map(|f| self.flags.contains(f))
            .unwrap_or(false)
    }

    fn bits(&self) -> Index {
        self.flags.bits()
    }

    fn __repr__(&self) -> String {
        format!("Modern(bits={:#06b})", self.flags.bits())
    }
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
