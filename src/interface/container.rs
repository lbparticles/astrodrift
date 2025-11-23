use pyo3::prelude::*;
use crate::interface::recipe::PyRecipe;
use crate::interface::potential::PyPotential;
#[pyclass]
#[derive(Clone)]
pub struct Container{
    pub recipe: Option<PyRecipe>,
    pub state: Option<shared::IState>,
}

#[pyfunction]
#[pyo3(signature = ())]
pub fn test_group<'py>(
    _py: Python<'py>,
)->Container{
    Container{recipe:None,state:None}
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn part_group<'py>(
    _py: Python<'py>,
    potential: PyPotential,
)->Container{
    Container{recipe:Some(potential.into()),state:None}
}

#[pyfunction]
#[pyo3(signature = (potential))]
pub fn bg_feature<'py>(
    _py: Python<'py>,
    potential: PyPotential,
)->Container{
    Container{recipe:Some(potential.into()),state:None}
}
// #[pymethods]
// impl IntegratorContainer{
//     #[staticmethod]
//     fn part_group<'py>(
//         _py: Python<'py>,
//         potential: PyPotential,
//     )->Self{
//         Self{recipe:Some(potential.into()),state:None}
//     }
//     #[staticmethod]
//     fn bg_feature(potential:PyPotential)->Self{
//         Self{recipe:Some(potential.into()),state:None}
//     }
//     #[staticmethod]
//     #[pyo3(signature = ())]
//     // fn test_group(
//     fn test_group<'py>(
//      _py: Python<'py>,
//     )->Self{
//         Self{recipe:None,state:None}
//     }
// }
