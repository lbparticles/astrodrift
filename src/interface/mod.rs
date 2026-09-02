use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule, PyTuple};

mod container;
mod engine;
mod flag;
mod method;
mod recipe;
mod variant;

pub use container::Container;
pub use engine::PyEngine;
pub use flag::Modern;
pub use method::PyMethod;
pub use recipe::PyRecipe;
use shared::{Linspace,Tolerance,Real,Model,Config,Index};
pub use variant::PyVariant;
use crate::dispatch::PotSpec;
use crate::integrators::run_integration;
use crate::state::InputFrame;
use crate::tree::AdjacencyMatrix;

#[derive(Default, Clone, Debug)]
pub struct BoundLinspace(pub Linspace);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundLinspace {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // --- Case 1: (start, end, num) tuple ---
        if let Ok(tup) = obj.cast::<PyTuple>() {
            if tup.len() != 3 {
                return Err(PyValueError::new_err(
                    "Linspace tuple must have 3 elements: (start, end, steps)",
                ));
            }
            let start: f64 = tup.get_item(0)?.extract()?;
            let end: f64 = tup.get_item(1)?.extract()?;
            let steps: usize = tup.get_item(2)?.extract()?;
            return Ok(BoundLinspace(Linspace{start, end, steps}));
        }

        // --- Case 2: NumPy array ---
        if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
            let slice = unsafe { arr.as_slice_mut()? };
            let n = slice.len();

            if n < 2 {
                return Err(PyValueError::new_err(
                    "NumPy linspace must contain at least two points",
                ));
            }

            let start = slice.first().copied().unwrap_or(0.0);
            let end = slice.last().copied().unwrap_or(start);
            let steps = n;
            return Ok(BoundLinspace(Linspace{start, end, steps}));
        }
        if let Ok(seq) = obj.extract::<Vec<f64>>() {
            if seq.len() < 2 {
                return Err(PyValueError::new_err(
                    "List must have at least two elements to form Linspace",
                ));
            }
            let start = seq[0];
            let end = *seq.last().unwrap();
            let steps = seq.len();
            return Ok(BoundLinspace(Linspace{start, end, steps}));
        }

        Err(PyValueError::new_err(
            "Expected (start, end, steps) tuple or 1D numpy.linspace array",
        ))
    }
}

#[derive(Default, Clone, Debug)]
pub struct BoundTolerance(pub Tolerance);
impl<'a, 'py> FromPyObject<'a, 'py> for BoundTolerance {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(tup) = obj.cast::<PyTuple>()
            && tup.len() == 2
        {
            let rtol: f64 = tup.get_item(0)?.extract()?;
            let atol: f64 = tup.get_item(1)?.extract()?;
            return Ok(BoundTolerance(Tolerance{rtol, atol}));
        }

        // Accept single float for convenience
        if let Ok(val) = obj.extract::<f64>() {
            return Ok(BoundTolerance(Tolerance{rtol: val, atol: val}));
        }

        Err(PyValueError::new_err(
            "Expected (rtol, atol) tuple or single float tolerance value",
        ))
    }
}

#[pyclass(name = "Config")]
#[derive(Debug)]
pub struct PyConfig {
    inner: Config,
    adjacency_matrix: AdjacencyMatrix,
}

impl PyConfig {
    fn build_tree(&self,
         containers: Vec<Box<Container>>) -> (Model, InputFrame) {
        let input = vec_to_option_array_11(containers);
        let (x,y) = self.adjacency_matrix.build(input);
        // println!("{:?}",x);
        // println!("{:?}",y);
        (x,y)
    }
}


fn vec_to_option_array_11(v: Vec<Box<Container>>) -> Box<[Option<Box<Container>>; 11]> {
    // Slot containers by their dependency_label: the AdjacencyMatrix is keyed
    // by label, so run-order positioning silently mismatched labels whenever
    // more than one Config was created in a process (the global label counter
    // never resets) and whole integration stages were skipped.
    let mut out: Box<[Option<Box<Container>>; 11]> = Box::new([None,None,None,None,None,None,None,None,None,None,None]);
    for item in v {
        let label = item.dependency_label;
        if label < 11 {
            out[label] = Some(item);
        } else {
            eprintln!("Error!!!! Container label {label} out of range (max 11)")
        }
    }
    out
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (engine=None,method=None,variant=None,flags=None,ts=None,tolerance=None))]
    fn new(
        engine: Option<PyEngine>,
        method: Option<PyMethod>,
        variant: Option<PyVariant>,
        flags: Option<Modern>,
        ts: Option<BoundLinspace>,
        tolerance: Option<BoundTolerance>,
    ) -> Self {
        let thing = Self {
            inner: Config::new(
                engine.unwrap_or_default().inner,
                method.unwrap_or_default().inner,
                variant.unwrap_or_default().inner,
                flags.unwrap_or_default().inner,
                ts.unwrap_or_default().0,
                tolerance.unwrap_or_default().0,
            ),
            adjacency_matrix: AdjacencyMatrix(0),
        };
        thing
    }

    #[pyo3(signature = (*args))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
    ) 
    -> PyResult<Bound<'py, PyList>> 
    {
        let mut containers: Vec<Box<Container>> = Vec::new();
        let mut pot: Option<PotSpec> = None;
        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            // First Bovy background container defines the MW2014 potential
            // spec (bulge force LUT + geometry; see dispatch::PotSpec).
            if pot.is_none() {
                if let (Some(recipe), Some(table)) = (&container.recipe, &container.table) {
                    if matches!(recipe.inner, shared::Recipe::Bovy(_)) {
                        pot = Some(PotSpec {
                            fparams: [table.r_min, table.dr, 0.0, 0.0, 0.0, 0.0],
                            uparams: [0, table.values.len(), 0, 0, 0, 0],
                            supertable: table.values.clone(),
                            annulus: None,
                        });
                    }
                }
            }
            // Test-particle container may carry the annulus perturber stack
            // (quintic-origin coefficients); attach it to the potential spec.
            if container.annulus.is_some() {
                match &mut pot {
                    Some(spec) => {
                        spec.annulus = container.annulus.clone().map(|a| {
                            crate::dispatch::AnnulusSpec {
                                coeffs: a.coeffs.clone(),
                                n_gmc: a.n_gmc,
                                division: a.division,
                                final_time: a.final_time,
                                plummer_amp: a.plummer_amp,
                                plummer_b: a.plummer_b,
                            }
                        })
                    }
                    None => panic!(
                        "annulus stack requires a Bovy/MW2014 background \
                         container in the same run"
                    ),
                }
            }
            containers.push(Box::new(container.clone()));
        }
        let (meal, istates) = self.build_tree(containers);
        let mut results = run_integration(self.inner, meal, istates, pot.as_ref()).unwrap();
        let nt = self.inner.settings.ts.steps;
        let items: Vec<Py<PyAny>> = results
            .0
            .iter_mut()
            .filter_map(|opt| opt.as_mut())
            .map(|arr| {
                // Hand results back as an (nt, n, 6) float64 array per
                // integrated container (zero-copy from the host buffer).
                let n_states = if nt > 0 { arr.data.len() / (nt * 6) } else { 0 };
                let flat = PyArray1::from_vec(py, std::mem::take(&mut arr.data));
                flat.reshape([nt, n_states, 6])
                    .expect("reshape integration output")
                    .into_any()
                    .unbind()
            })
            .collect::<Vec<_>>();
        PyList::new(py, items)
    }

    #[pyo3(signature = (node,*args))]
    fn dependency<'py>(
        &mut self,
        _py: Python<'py>,
        node: Container,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<()> {
        let mut dep: Vec<Index> = Vec::new();

        for i in 0..args.len() {
            let obj = args.get_item(i)?;
            let container: PyRef<Container> = obj.extract()?;
            dep.push(container.dependency_label);
        }
        for x in dep.iter() {
            self.adjacency_matrix
                .set(x.clone(), node.dependency_label, true);
        }
        Ok(())
    }
    #[pyo3(signature = ())]
    fn info(&self) -> () {
        println!("{:?}", self);
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
    m.add_class::<PyRecipe>()?;
    m.add_class::<Modern>()?;
    m.add_class::<Container>()?;
    m.add_function(wrap_pyfunction!(container::test_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::part_group, m)?)?;
    m.add_function(wrap_pyfunction!(container::bg_feature, m)?)?;

    // Define enum.Flag in Python
    let locals = PyDict::new(py);
    py.run(
        c_str!(
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
"#
        ),
        None,
        Some(&locals),
    )?;

    let py_enum = locals.get_item("ModernFlag").unwrap();
    m.add("ModernFlag", py_enum)?;

    Ok(())
}
