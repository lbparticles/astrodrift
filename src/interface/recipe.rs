use pyo3::prelude::*;
#[pyclass(name = "Potential",subclass)]
#[derive(Default, Debug, Clone, Copy)]
pub struct PyRecipe {
    pub inner: shared::Recipe,
}

#[pymethods]
impl PyRecipe {
    #[staticmethod]
    fn kepler(amp: Option<shared::Real>) -> Self {
        Self {
            inner: shared::Recipe::Kepler(shared::KeplerRecipe {
                name: shared::PotentialName::Kepler,
                amp: amp.unwrap_or_default(),
            }),
        }
    }

    #[staticmethod]
    fn plummer(amp: Option<shared::Real>, radius: Option<shared::Real>) -> Self {
        Self {
            inner: shared::Recipe::Plummer(shared::PlummerRecipe {
                name: shared::PotentialName::Plummer,
                amp: amp.unwrap_or_default(),
                radius: radius.unwrap_or_default(),
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

// #[pymethods]
// imp PyRecipe {
//     fn force(&self,positions:Vec<f64>,t:f64)->Vec<f64>{
//         for p in postiions{
//             let (ax,ay,az) = self.inner.construct().force(t,p[0],p[1],p[2])
//         }
//     }
//     fn evaluate(&self,positions:Vec<f64>,t:f64)->f64{
//         for p in postiions{
//             let (ax,ay,az) = self.inner.construct().force(t,p[0],p[1],p[2])
//         }
//     }
// }
