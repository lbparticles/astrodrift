use pyo3::prelude::*;
#[pyclass(name = "Potential", subclass)]
#[derive(Default, Debug, Clone, Copy)]
pub struct PyRecipe {
    pub inner: shared::Recipe,
}

#[pymethods]
impl PyRecipe {
    /// Point-mass potential. Units follow the codebase convention (G = 1).
    ///
    /// :param amp: Total mass of the point mass. Required; there is no
    ///     physically meaningful default.
    #[staticmethod]
    fn kepler(amp: shared::Real) -> Self {
        Self {
            inner: shared::Recipe::Kepler(shared::KeplerRecipe {
                name: shared::PotentialName::Kepler,
                amp,
            }),
        }
    }

    /// Plummer sphere: a softened non-singular mass distribution.
    ///
    /// :param amp: Total mass of the sphere.
    /// :param radius: Plummer scale radius.
    #[staticmethod]
    fn plummer(amp: shared::Real, radius: shared::Real) -> Self {
        Self {
            inner: shared::Recipe::Plummer(shared::PlummerRecipe {
                name: shared::PotentialName::Plummer,
                amp,
                radius,
            }),
        }
    }

    #[staticmethod]
    fn bovy() -> Self {
        Self {
            inner: shared::Recipe::Bovy(shared::BovyRecipe {
                name: shared::PotentialName::Bovy,
            }),
        }
    }
}
