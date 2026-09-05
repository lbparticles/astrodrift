use pyo3::prelude::*;

/// A gravitational potential definition.
///
/// Construct potentials with :meth:`Potential.kepler`,
/// :meth:`Potential.plummer`, or :meth:`Potential.bovy`.
#[pyclass(name = "Potential", subclass, from_py_object)]
#[derive(Default, Debug, Clone)]
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

    /// Construct the built-in composite background potential.
    #[staticmethod]
    fn bovy() -> Self {
        Self {
            inner: shared::Recipe::Bovy(shared::BovyRecipe {
                name: shared::PotentialName::Bovy,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PyRecipe;
    use shared::Recipe;

    #[test]
    fn kepler_amplitude_is_preserved() {
        let Recipe::Kepler(recipe) = PyRecipe::kepler(2.5).inner else {
            panic!("expected a Kepler recipe");
        };

        assert_eq!(recipe.amp, 2.5);
    }

    #[test]
    fn plummer_parameters_are_preserved() {
        let Recipe::Plummer(recipe) = PyRecipe::plummer(3.5, 0.25).inner else {
            panic!("expected a Plummer recipe");
        };

        assert_eq!(recipe.amp, 3.5);
        assert_eq!(recipe.radius, 0.25);
    }
}
