use pyo3::prelude::*;

#[pyclass(name = "Potential", subclass)]
#[derive(Default, Clone)]
pub struct PyPotential {
    pub inner: shared::PotentialEnum,
}

#[pymethods]
impl PyPotential {
    #[staticmethod]
    fn kepler(amp: Option<shared::Real>) -> Self {
        Self {
            inner: shared::PotentialEnum::Kepler(shared::KeplerPotential {
                name: shared::PotentialName::Kepler,
                amp: amp.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn plummer(amp: Option<shared::Real>, radius: Option<shared::Real>) -> Self {
        Self {
            inner: shared::PotentialEnum::Plummer(shared::PlummerPotential {
                name: shared::PotentialName::Plummer,
                amp: amp.unwrap_or_default(),
                radius: radius.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn bovy() -> Self {
        Self {
            inner: shared::PotentialEnum::Bovy(shared::BovyPotential {
                name: shared::PotentialName::Bovy,
            }),
        }
    }
}
