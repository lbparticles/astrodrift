
pub struct Course(pub [Option<Recipe>; MAX_RECIPES]);
pub struct Meal(pub Box<[Option<Course>; MAX_COURSES]>);

pub type IndexParams = (Index, Index, Index, Index, Index, Index);
pub type RealParams = (Real, Real, Real, Real, Real, Real);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    pub real_params: RealParams,
    pub index_params: IndexParams,
    pub potential: PotentialName,
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            index_params: (0, 0, 0, 0, 0, 0),
            potential: PotentialName::Kepler,
        }
    }
}
impl From<PotentialEnum> for Recipe {
    fn from(pot: PotentialEnum) -> Self {
        match pot {
            PotentialEnum::Kepler(p) => Self {
                real_params: (p.amp, 0.0, 0.0, 0.0, 0.0, 0.0),
                index_params: (0, 0, 0, 0, 0, 0),
                potential: PotentialName::Kepler,
            },
            PotentialEnum::Plummer(p) => Self {
                real_params: (p.amp, p.radius, 0.0, 0.0, 0.0, 0.0),
                index_params: (0, 0, 0, 0, 0, 0),
                potential: PotentialName::Plummer,
            },
            PotentialEnum::Bovy(_p) => Self {
                real_params: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                index_params: (0, 0, 0, 0, 0, 0),
                potential: PotentialName::Bovy,
            },
        }
    }
}



impl Display for Meal{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        let mut first_outer = true;

        for outer_opt in self.0.iter() {
            let Some(inner_arr) = outer_opt else { continue };

            // Filter only present recipes
            let mut inner_iter = inner_arr.iter().filter_map(|opt| opt.as_ref());

            // Skip this outer slot if it would be empty after filtering
            if inner_iter.clone().next().is_none() {
                continue;
            }

            if !first_outer {
                write!(f, ", ")?;
            }
            first_outer = false;

            write!(f, "[")?;
            let mut first_inner = true;
            for recipe in inner_iter {
                if !first_inner {
                    write!(f, ", ")?;
                }
                first_inner = false;
                write!(f, "{:?}", recipe)?;
            }
            write!(f, "]")?;
        }

        write!(f, "]")
    }
}


impl From<[Option<[Option<Recipe>; 11]>; 11]> for Meal {
    fn from(arr: [Option<[Option<Recipe>; 11]>; 11]) -> Self {
        Meal(Box::new(arr))
    }
}

impl From<Box<[Option<[Option<Recipe>; 11]>; 11]>> for Meal {
    fn from(b: Box<[Option<[Option<Recipe>; 11]>; 11]>) -> Self {
        Meal(b)
    }
}
