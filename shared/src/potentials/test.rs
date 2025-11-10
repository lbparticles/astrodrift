use super::*;
use crate::combine_potentials;
use super::ConstPotential;
#[test]
fn combine_potentials_sums_correctly() {
    let a = ConstPotential { value: 1.0 };
    let b = ConstPotential { value: 2.0 };
    let c = ConstPotential { value: 3.0 };

    // Combine using the macro
    let combined = combine_potentials!(a, b, c);

    // Evaluate and force should both sum the component values
    assert_eq!(combined.evaluate(0.0, 0.0, 0.0, 0.0), 6.0);

    let (fx, fy, fz) = combined.force(0.0, 0.0, 0.0, 0.0);
    assert_eq!((fx, fy, fz), (6.0, 6.0, 6.0));
}

#[test]
fn combine_potentials_two_terms() {
    let a = ConstPotential { value: 10.0 };
    let b = ConstPotential { value: -2.0 };

    let combined = combine_potentials!(a, b);
    assert_eq!(combined.evaluate(0.0, 0.0, 0.0, 0.0), 8.0);

    let (fx, fy, fz) = combined.force(0.0, 0.0, 0.0, 0.0);
    assert_eq!((fx, fy, fz), (8.0, 8.0, 8.0));
}
