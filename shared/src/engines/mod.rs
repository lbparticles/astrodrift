mod dop54;
pub use dop54::DormandPrince54;

pub trait ButcherTableau {
    const ORDER: usize;
    const STAGES: usize;
    const A: [[f64; 7]; 7];
    const B: [f64; 7];
    const B_HAT: [f64; 7]; // FIXME: Is this what they are called?
    const C: [f64; 7];
}
