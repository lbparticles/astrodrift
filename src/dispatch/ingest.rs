use tree::tree_constructor;
use tree::Container;
use pyo3::prelude::*;
use numpy::PyArray1;
use engine::dopr54_adaptive;

const MAX_INTEGRATIONS = 3;
const MAX_TIMESTEPS = 1000;
const STATE_DIM = 11;
type EngineOutput<'a> = PyResult<[Bound<'a, PyArray1<f64>>;MAX_INTEGRATIONS]>;


#[pyclass]
#[derive(Clone)]
struct InterfaceContainer<py'> {
    potentialID: usize,
    state0: PyReadonlyArray1<py',f64>
}

fn method<T>(in:Container<T>){
    dopr54_adaptive(in)
}

#[pyfunction]
pub fn ingest<'py>(
    py: Python<'py>,
    containers: PyArray<py',f64>,
    engine:usize,
)->EngineOutput<'py>{
    let method = parse(engine);
    let translation = containers.as_array();
    let stages = tree_constructor(translation); 
    state_out = [[0.0f64; STATE_DIM*MAX_TIMESTEPS*MAX_PARTICLES];MAX_INTEGRATIONS]
    // GIL held for entire function
    for (i,stage) in stages.iter().enumerate().take(MAX_INTEGRATIONS){
        state_out[i] = method(stage))};        
    }
    // // create table
    // let bulge_amp = 0.029994597188218296;
    // let bulge_alpha = 1.8;
    // let bulge_r1 = 1.0;
    // let bulge_rc = 1.9 / 8.0;

    // let (ar_table_host, r_min, dr) =
    //     build_sphericalcutoff_force_table(bulge_amp, bulge_alpha, bulge_r1, bulge_rc);
    // // let (ar_table_eval_host, r_min, dr) =
    // //     build_sphericalcutoff_eval_table(bulge_amp, bulge_alpha, bulge_rc);

    // t, x, y, z, vx, vy, vz, ax, ay, az, potential_energy
    let flat = PyArray1::from_vec(py, state_out);

    Ok([flat,flat,flat])
}
