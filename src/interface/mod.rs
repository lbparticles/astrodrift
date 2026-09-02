use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use std::os::raw::c_int;
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


thread_local! {
    /// (bulge force LUT, r_min, dr) stashed for the CPU RHS callback.
    static CPU_MW_LUT: std::cell::RefCell<Option<(Vec<f64>, f64, f64)>> =
        std::cell::RefCell::new(None);
}

/// Bind the bulge force LUT for the CPU DOP853 RHS (call once per
/// process/table; mirrors the GPU's bg_feature pattern).
#[pyfunction]
#[pyo3(signature = (ar_table, r_min, dr))]
fn set_cpu_mw_lut(ar_table: Vec<f64>, r_min: f64, dr: f64) -> PyResult<()> {
    if ar_table.len() < 2 {
        return Err(PyValueError::new_err("ar_table needs at least 2 entries"));
    }
    crate::integrators::dop853_cpu::set_mw_cpu_context(
        crate::integrators::dop853_cpu::MwCpuContext {
            lut: ar_table,
            r_min,
            dr,
            annulus: None,
        },
    );
    Ok(())
}

/// RHS evaluation count since the last reset (diagnostics).
#[pyfunction]
#[pyo3(signature = (reset = false))]
fn cpu_mw_rhs_evals(reset: bool) -> u64 {
    if reset {
        crate::integrators::dop853_cpu::reset_mw_cpu_rhs_evals();
        0
    } else {
        crate::integrators::dop853_cpu::mw_cpu_rhs_evals()
    }
}

/// Integrate ONE particle in MW2014 (bulge LUT + MN + NFW) with the
/// CPU DOP853 integrator. Requires `set_cpu_mw_lut` first.
#[pyfunction]
#[pyo3(signature = (state0, times, rtol, atol))]
fn dop853_mw2014_cpu(
    state0: Vec<f64>,
    times: Vec<f64>,
    rtol: f64,
    atol: f64,
) -> PyResult<Vec<f64>> {
    use crate::integrators::dop853_cpu;
    let ctx = dop853_cpu::mw_cpu_context()
        .ok_or_else(|| PyValueError::new_err("call set_cpu_mw_lut first"))?;
    if state0.len() != 6 || times.len() < 2 {
        return Err(PyValueError::new_err(
            "state0 must have 6 entries; times needs at least 2",
        ));
    }
    let nt = times.len();
    let mut result = vec![0.0_f64; nt * 6];
    let mut err: i32 = 0;
    unsafe {
        dop853_cpu::dop853(
            Some(dop853_cpu::mw2014_cpu_rhs),
            6,
            state0.as_ptr() as *mut f64,
            nt as i32,
            -9999.99,
            times.as_ptr() as *mut f64,
            0,
            ctx as *const dop853_cpu::MwCpuContext as *mut std::ffi::c_void,
            rtol,
            atol,
            result.as_mut_ptr(),
            &mut err,
        );
    }
    if err != 0 {
        return Err(PyValueError::new_err(format!("dop853 reported err={err}")));
    }
    Ok(result)
}

/// Batch: integrate N particles in MW2014 (+ optional GMC annulus stack)
/// with the CPU DOP853 integrator (rayon-parallel over particles). States:
/// (n, 6) float64. Returns a flat (nt, n, 6) buffer. Requires
/// `set_cpu_mw_lut` first; pass the annulus kwargs for pot_type=2 physics
/// (quintic-origin stack, same layout as the GPU path).
#[pyfunction]
#[pyo3(signature = (states, times, rtol, atol, annulus_coeffs=None,
                    n_gmc=None, division=None, final_time=None,
                    plummer_amp=None, plummer_b=None))]
#[allow(clippy::too_many_arguments)]
fn dop853_mw2014_cpu_batch<'py>(
    py: Python<'py>,
    states: &Bound<'py, numpy::PyArray2<f64>>,
    times: Vec<f64>,
    rtol: f64,
    atol: f64,
    annulus_coeffs: Option<Vec<f64>>,
    n_gmc: Option<usize>,
    division: Option<usize>,
    final_time: Option<f64>,
    plummer_amp: Option<f64>,
    plummer_b: Option<f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use crate::integrators::dop853_cpu;
    use numpy::PyArrayMethods;
    let ro = states.readonly();
    let flat = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("states must be contiguous"))?;
    if flat.len() % 6 != 0 {
        return Err(PyValueError::new_err("states must be (n, 6)"));
    }
    let _ = py; // GIL intentionally held (see set_cpu_mw_lut race note)
    let ctx = dop853_cpu::mw_cpu_context()
        .ok_or_else(|| PyValueError::new_err("call set_cpu_mw_lut first"))?;
    let nt = times.len();
    let n = flat.len() / 6;

    let annulus = match (
        annulus_coeffs,
        n_gmc,
        division,
        final_time,
        plummer_amp,
        plummer_b,
    ) {
        (None, None, None, None, None, None) => None,
        (Some(coeffs), Some(n_gmc), Some(division), Some(final_time),
         Some(plummer_amp), Some(plummer_b)) => {
            let expected = 18 * n_gmc * division;
            if coeffs.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "annulus_coeffs length {} != 18 * n_gmc * division = {expected}",
                    coeffs.len()
                )));
            }
            Some(dop853_cpu::CpuAnnulusCtx {
                coeffs,
                n_gmc,
                division,
                final_time,
                amp: plummer_amp,
                b: plummer_b,
            })
        }
        _ => {
            return Err(PyValueError::new_err(
                "annulus requires all of: annulus_coeffs, n_gmc, division,                  final_time, plummer_amp, plummer_b",
            ))
        }
    };

    let ctx = dop853_cpu::MwCpuContext {
        lut: ctx.lut.clone(),
        r_min: ctx.r_min,
        dr: ctx.dr,
        annulus,
    };

    // The GIL is held on purpose: set_cpu_mw_lut re-binding while a batch is
    // in flight would race the RHS context pointer. (rayon workers do not
    // need the GIL; blocking other Python threads for the batch duration is
    // the accepted trade-off.)
    let out = dop853_cpu::dop853_mw2014_batch(flat, &times, rtol, atol, &ctx);
    Ok(numpy::PyArray1::from_vec(py, out))
}

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
    m.add_function(wrap_pyfunction!(dop853_mw2014_cpu, m)?);
    m.add_function(wrap_pyfunction!(dop853_mw2014_cpu_batch, m)?);
    m.add_function(wrap_pyfunction!(set_cpu_mw_lut, m)?);
    m.add_function(wrap_pyfunction!(cpu_mw_rhs_evals, m)?);

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
