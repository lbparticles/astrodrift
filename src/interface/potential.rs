use pyo3::prelude::*;

#[pyclass(name = "Potential", subclass)]
#[derive(Default, Clone)]
pub struct PyPotential {
    pub inner: shared::RecipeEnum,
}

#[pymethods]
impl PyPotential {
    #[staticmethod]
    fn kepler(amp: Option<shared::Real>) -> Self {
        Self {
            inner: shared::RecipeEnum::Kepler(shared::KeplerRecipe {
                name: shared::PotentialName::Kepler,
                amp: amp.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn plummer(amp: Option<shared::Real>, radius: Option<shared::Real>) -> Self {
        Self {
            inner: shared::RecipeEnum::Plummer(shared::PlummerRecipe {
                name: shared::PotentialName::Plummer,
                amp: amp.unwrap_or_default(),
                radius: radius.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn bovy() -> Self {
        Self {
            inner: shared::RecipeEnum::Bovy(shared::BovyRecipe {
                name: shared::PotentialName::Bovy,
            }),
        }
    }
}
