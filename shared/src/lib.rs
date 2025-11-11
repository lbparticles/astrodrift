mod potentials;



struct PotentialRecipe {
    potential_id: PotentialEnum,
    fparams: [f64;6],
    uparams: [usize;6],
    lutInfo: Option<LookUpTable>,
}

struct LookUpTable {
    id: usize,
    offset: f64,
    length: usize,
}


enum PotentialEnum {
    Bovy14,
    Plummer,
    MN,
    NFW,
    SphCutoff,
}
