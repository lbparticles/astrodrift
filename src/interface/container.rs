use pyo3::prelude::*;
use crate::interface::recipe::PyRecipe;
#[pyclass]
#[derive(Clone)]
pub struct IntegratorContainer{
    pub recipe: Option<PyRecipe>,
    pub state: Option<shared::IState>,
}
#[pymethods]
impl IntegratorContainer{
    #[staticmethod]
    fn part_group<'py>(
        _py: Python<'py>,
    )->Self{
        Self{recipe:None,state:None}
    }
    #[staticmethod]
    fn bg_feature()->Self{
        Self{recipe:None,state:None}
    }
    #[staticmethod]
    fn test_group<'py>(
        _py: Python<'py>,
    )->Self{
        Self{recipe:None,state:None}
    }
}
