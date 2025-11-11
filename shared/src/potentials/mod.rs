use core::ops::Add;
use libm::{atan2, cos, floor, log, pow, sin, sqrt};

#[macro_export]
macro_rules! unimplemented {
    () => {
        $crate::panicking::panic("not implemented")
    };
    ($($arg:tt)+) => {
        $crate::panic!("not implemented: {}", $crate::format_args!($($arg)+))
    };
}

pub trait Potential {
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64;
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64);
}

// let references to a potential also be a Potential (so &T works)
impl<T: Potential + ?Sized> Potential for &T {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        unimplemented!();
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (*self).force(t, x, y, z)
    }
}

// can hold owned values or references
#[derive(Clone, Copy)]
pub struct Sum<P, Q> {
    pub p: P,
    pub q: Q,
}

impl<P: Potential, Q: Potential> Potential for Sum<P, Q> {
    #[inline(always)]
    fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
        self.p.evaluate(t, x, y, z) + self.q.evaluate(t, x, y, z)
    }
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (px, py, pz) = self.p.force(t, x, y, z);
        let (qx, qy, qz) = self.q.force(t, x, y, z);
        (px + qx, py + qy, pz + qz)
    }
}

// instead of working around the orphan rule we will just use a macro
#[macro_export]
macro_rules! combine_potentials {
    ($first:expr $(, $rest:expr)+ $(,)?) => {{
        let acc = $first;
        $( let acc = $crate::potential::Sum { p: acc, q: $rest }; )+
        acc
    }};
}
