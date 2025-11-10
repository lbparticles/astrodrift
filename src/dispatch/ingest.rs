use tree::tree_constructor;
use tree::Container;
use pyo3::prelude::*;
use numpy::PyArray1;
use engine::dopr54_adaptive;
use shared::{PotentialEnum,MNPotential,NFWPotential,PlummerPotential,MW2014Potential,SphericalcutoffPotential};
use crate::consts::{MAX_INTEGRATIONS,MAX_TIMESTEPS,STATE_DIM,MAX_PARTICLES};

type EngineOutput = PyResult<Vec<Py<PyArray1<f64>>>>;


#[pyclass]
#[derive(Clone)]
struct InterfaceContainer<'py> {
    potential_id: usize,
    params: Option<PyReadonlyArray1<'py,f64>>,
    state0: PyReadonlyArray1<'py,f64>
}
// The imported Container
// pub struct Container {
//     potential: Box<dyn Potential>,
//     istate: mut *f64,
//     dependencies: Vec<*mut Container<T>>,
// }
fn construct_nodes(containers:Vec<InterfaceContainer>)->Vec<Container>{
    let mut output= Vec<Container>::new();
    for c in containers.iter(){      
    	let params_vec = c
    	    .params
    	    .as_ref()
    	    .map(|a| a.as_slice().unwrap().to_vec())
    	    .unwrap_or_default();
	
    	let state_vec = c.state0.as_slice().unwrap().to_vec();
        let potential = match c.potential_id {
            // PotentialEnum::Custom=>CustomPotential, 
            PotentialEnum::Bovy14=>MW2014Potential{params_vec}, 
            // PotentialEnum::SpiralArm=>, 
            // PotentialEnum::Bar=>, 
            PotentialEnum::Plummer=>PlummerPotential{params_vec}, 
            // PotentialEnum::Point=>, 
            PotentialEnum::NFW=>NFWPotential{params_vec}, 
            PotentialEnum::Sphericalcutoff=>SphericalcutoffPotential{params},             
            PotentialEnum::MN=>MNPotential{params_vec},
        }
        output.push(Container{potential:Box::new(potential),istate:state_vec,dependencies: Vec::new(),})
    }
    output
}

#[pyfunction]
pub fn ingest<'py>(
    py: Python<'py>,
    containers: Vec<InterfaceContainer>,
    engine:usize,
)->EngineOutput{
    let nodes = construct_nodes(containers);
    let stages = tree_constructor(nodes); 
    let method = match engine {
        0 => dopr54_adaptive,
        _ => dopr54_adaptive,
    };
    let results = py.allow_threads(|| {
        stages.iter()
            .take(MAX_INTEGRATIONS)
            .map(|stage| dopr54_adaptive(stage))
            .collect::<Vec<_>>()
    });
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
    let output = results
        .into_iter()
        .map(|r| PyArray1::from_vec(py, r).to_owned())
        .collect();
    Ok(output)
}
