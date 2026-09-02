use libm::{floor, log, sqrt};

#[derive(Clone, Copy)]
pub enum PotentialEnum {
    Bovy(BovyPotential),
    Plummer(PlummerPotential),
    CustomPlummer(CustomOrigin<PlummerPotential>),
    Kepler(KeplerPotential),
    CustomKepler(CustomOrigin<KeplerPotential>),
    // NOTE: nothing constructs the Custom* variants yet (the recipe
    // `Construct` impls still return the origin-less potentials); they exist
    // so the moving-potential machinery has a home on the host side.
    // MNPotential(MNPotential),
    // NFWPotential(NFWPotential),
    // SphericalcutoffPotential(SphericalcutoffPotential),
}

impl Potential for PotentialEnum {
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        match self {
            PotentialEnum::Bovy(p) => p.force(t, x, y, z),
            PotentialEnum::Plummer(p) => p.force(t, x, y, z),
            PotentialEnum::Kepler(p) => p.force(t, x, y, z),
            PotentialEnum::CustomKepler(p) => p.force(t, x, y, z),
            PotentialEnum::CustomPlummer(p) => p.force(t, x, y, z),
            // PotentialEnum::MNPotential(p) => p.force(t, x, y, z),
            // PotentialEnum::NFWPotential(p) => p.force(t, x, y, z),
            // PotentialEnum::SphericalcutoffPotential(p) => p.force(t, x, y, z),
        }
    }

    // Optional if you uncomment in the trait:
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     match self {
    //         PotentialEnum::MW2014Potential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::MNPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::NFWPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::PlummerPotential(p) => p.evaluate(t, x, y, z),
    //         PotentialEnum::SphericalcutoffPotential(p) => p.evaluate(t, x, y, z),
    //     }
    // }
}

/// Piecewise-quintic origin trajectory table (single source of truth for
/// every consumer: the GPU kernel's annulus stack, host-side `CustomOrigin`
/// potentials, and the Python/numpy mirror -- see
/// `benchmarks/annulus_mirror.py` and `quintic_coeffs` in
/// `benchmarks/annulus_run.py`, which must all use this convention).
///
/// One trajectory per object, split into `division` equal time divisions of
/// width `dt = final_time / division`. Object `i`'s position at time `t` is
/// a quintic in local time
///
/// ```text
/// tau = t - t0 * dt   in [0, dt],  t0 = floor(t / dt) (clamped)
/// ```
///
/// with six coefficients per cartesian axis, stored highest power first:
///
/// ```text
/// origin_a(t) = sum_{k=0..5} coeff[k] * tau^(5-k)
/// ```
///
/// Coefficient `k` of axis `a` lives at table element
///
/// ```text
/// 18 * (i * division + t0) + 6 * a + k
/// ```
///
/// (particle-major layout: all divisions of object i are contiguous). The
/// coefficients are normally the quintic Hermite fit of the object's stage-1
/// trajectory over each division, matching the builder's row order
/// [p(h), p'(h), p''(h), p(0), p'(0), p''(0)].
///
/// NOTE: an earlier, idealised version of this machinery (the load_data-era
/// `CustomOrigin`) evaluated `quintic_interp(t - t0)` -- local time in
/// *division-index* units -- and indexed with stride `18 * (i * n_objects +
/// t0)`. Both conventions only coincide with this table when dt == 1 and
/// division == n_objects; any consumer still using that convention produces
/// O(1) origin errors and must be migrated (see tests).
#[derive(Debug, Clone, Copy)]
pub struct QuinticOriginTable {
    pub table: *const f64,
    pub n_objects: usize,
    pub division: usize,
    pub final_time: f64,
}

#[derive(Clone, Copy)]
struct QuinticCoeff {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

#[inline(always)]
fn quintic_interp(t: f64, coeff: QuinticCoeff) -> f64 {
    let t2 = t * t;
    let t3 = t * t2;
    let t4 = t * t3;
    let t5 = t * t4;
    coeff.a * t5 + coeff.b * t4 + coeff.c * t3 + coeff.d * t2 + coeff.e * t + coeff.f
}

impl QuinticOriginTable {
    /// Division index t0 and local time tau = t - t0*dt for absolute time t.
    /// t0 is clamped to [0, division-1] so that t in [0, final_time] maps to
    /// a valid slice (t == final_time evaluates the last slice at tau = dt).
    #[inline(always)]
    fn division_start(&self, t: f64) -> (usize, f64) {
        let dt = self.final_time / (self.division as f64);
        let mut t0 = floor(t / dt);
        if t0 < 0.0 {
            t0 = 0.0;
        }
        if t0 >= self.division as f64 {
            t0 = (self.division - 1) as f64;
        }
        let t0 = t0 as usize;
        (t0, t - t0 as f64 * dt)
    }

    /// Origin of object i at absolute time t (see the struct docs for the
    /// layout and local-time conventions).
    ///
    /// # Safety
    /// `table` must point to at least `18 * n_objects * division` readable
    /// f64 elements for the lifetime of the call.
    #[inline(always)]
    pub unsafe fn origin(&self, t: f64, i: usize) -> [f64; 3] {
        let (t0, tau) = self.division_start(t);
        let p0 = 18 * (i * self.division + t0);
        let c = |k: usize| QuinticCoeff {
            a: *self.table.add(p0 + k),
            b: *self.table.add(p0 + k + 1),
            c: *self.table.add(p0 + k + 2),
            d: *self.table.add(p0 + k + 3),
            e: *self.table.add(p0 + k + 4),
            f: *self.table.add(p0 + k + 5),
        };
        let xc = c(0);
        let yc = c(6);
        let zc = c(12);
        [
            quintic_interp(tau, xc),
            quintic_interp(tau, yc),
            quintic_interp(tau, zc),
        ]
    }
}

pub trait Potential {
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64;
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64);
}

// let references to a potential also be a Potential (so &T works)
impl<T: Potential + ?Sized> Potential for &T {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    // }
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
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     self.p.evaluate(t, x, y, z) + self.q.evaluate(t, x, y, z)
    // }
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

#[derive(Clone, Copy)]
pub struct BovyPotential {
    bulge: SphericalcutoffPotential,
    disk: MNPotential,
    halo: NFWPotential,
}

impl BovyPotential {
    pub const fn new(ar_table: *const f64, r_min: f64, dr: f64, n_ar: usize) -> Self {
        let bulge = SphericalcutoffPotential {
            ar_table,
            r_min,
            dr,
            n_ar,
        };

        let disk = MNPotential {
            amp: 0.7574802019,
            a: 3.0 / 8.0,
            b: 0.28 / 8.0,
        };

        let halo = NFWPotential {
            amp: 4.852230533528,
            a: 16.0 / 8.0,
        };
        Self {
            bulge: bulge,
            disk: disk,
            halo: halo,
        }
    }
}

impl Potential for BovyPotential {
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     combine_potentials!(&self.bulge, &self.disk, &self.halo).evaluate(t, x, y, z)
    // }
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        combine_potentials!(&self.bulge, &self.disk, &self.halo).force(_t, x, y, z)
    }
}

#[derive(Clone, Copy)]
pub struct MNPotential {
    pub amp: f64,
    pub a: f64,
    pub b: f64,
}
impl Potential for MNPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     0.
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        // NOTE: no libm::pow here -- the annulus kernel evaluates this once
        // per (GMC, stage, substep); a generic pow costs 50-200 cycles of
        // FP64 vs ~3 ops for the explicit form. Measured ~10x on the stack
        // loop (see benchmarks/drift_stage2_breakdown.py).
        let r2 = x * x + y * y;
        let r = sqrt(r2);
        let z2 = z * z;
        let b2 = self.b * self.b;
        let sqrtz2b2 = sqrt(z2 + b2);
        let a1 = self.a + sqrtz2b2;
        let pyth = a1 * a1;
        let denom = (pyth + r2) * sqrt(pyth + r2);
        let ar = -self.amp * (r / denom);
        let ax = ar * (x / r);
        let ay = ar * (y / r);
        let az = -self.amp * (z * a1) / (sqrtz2b2 * denom);
        (ax, ay, az)
    }
}

#[derive(Clone, Copy)]
pub struct NFWPotential {
    pub amp: f64,
    pub a: f64,
}
impl Potential for NFWPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     0.
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = x * x + y * y + z * z;
        let r = sqrt(r2);

        let ar = -self.amp * (log(1. + r / self.a) - r / (self.a + r)) / r2;

        let ax = ar * (x / r);
        let ay = ar * (y / r);
        let az = ar * (z / r);
        (ax, ay, az)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SphericalcutoffPotential {
    pub ar_table: *const f64,
    pub r_min: f64,
    pub dr: f64,
    pub n_ar: usize,
}
impl SphericalcutoffPotential {
    #[inline(always)]
    fn radial_force_table(&self, r: f64) -> f64 {
        let t = (r - self.r_min) / self.dr;
        let i = floor(t) as usize;
        let f = t - i as f64;

        // linear interpolation
        let i0 = i.min((self.n_ar - 2) as usize);
        let (ar0, ar1) = unsafe { (*self.ar_table.add(i0), *self.ar_table.add(i0 + 1)) };
        (1.0 - f) * ar0 + f * ar1
    }
}
impl Potential for SphericalcutoffPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = x * x + y * y + z * z;
        if r2 == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let r = sqrt(r2);
        let ar = self.radial_force_table(r);
        let ax = ar * x / r;
        let ay = ar * y / r;
        let az = ar * z / r;
        (ax, ay, az)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KeplerPotential {
    pub amp: f64,
}

impl Potential for KeplerPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
    //     return -self.amp / sqrt(r2 + pow(self.b, 2.0));
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let r2 = x * x + y * y + z * z;
        let r = sqrt(r2);
        let inv_r3 = 1.0 / (r2 * r);
        let ar = -self.amp * inv_r3;
        let ax = ar * x;
        let ay = ar * y;
        let az = ar * z;
        (ax, ay, az)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlummerPotential {
    pub amp: f64,
    pub b: f64,
}

impl Potential for PlummerPotential {
    // #[inline(always)]
    // fn evaluate(&self, t: f64, x: f64, y: f64, z: f64) -> f64 {
    //     let r2 = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
    //     return -self.amp / sqrt(r2 + pow(self.b, 2.0));
    // }
    #[inline(always)]
    fn force(&self, _t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        // (d^2 + b^2)^{-3/2} via sqrt: see the Plummer pow note above; this
        // sits in the annulus kernel's 2000-term inner loop.
        let r2 = x * x + y * y + z * z;
        let r2b2 = r2 + self.b * self.b;
        let ar = -self.amp / (r2b2 * sqrt(r2b2));
        let ax = ar * x;
        let ay = ar * y;
        let az = ar * z;
        (ax, ay, az)
    }
}

/// A potential whose (single) origin moves along a piecewise-quintic
/// trajectory (see [`QuinticOriginTable`]). The potential is evaluated at
/// (x, y, z) - origin(t); for a stack of movers, sum `CustomOrigin` forces
/// or index the table directly (the GPU annulus kernel does the latter).
#[derive(Debug, Clone, Copy)]
pub struct CustomOrigin<P: Potential + Copy> {
    pub origins: QuinticOriginTable,
    pub potential: P,
}

impl<P: Potential + Copy> Potential for CustomOrigin<P> {
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let p = unsafe { self.origins.origin(t, 0) };
        self.potential.force(t, x - p[0], y - p[1], z - p[2])
    }
}

/// Sum of `CustomOrigin` forces over a whole stack (host-side reference for
/// the kernel's annulus loop).
#[derive(Debug, Clone, Copy)]
pub struct CustomOriginStack<P: Potential + Copy> {
    pub origins: QuinticOriginTable,
    pub potential: P,
}

impl<P: Potential + Copy> Potential for CustomOriginStack<P> {
    #[inline(always)]
    fn force(&self, t: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let mut total_force = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..self.origins.n_objects {
            let p = unsafe { self.origins.origin(t, i) };
            let (f1, f2, f3) = self.potential.force(t, x - p[0], y - p[1], z - p[2]);
            total_force.0 += f1;
            total_force.1 += f2;
            total_force.2 += f3;
        }
        total_force
    }
}
