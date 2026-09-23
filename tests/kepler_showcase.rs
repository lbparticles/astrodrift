//! A closed showcase: the two-set Kepler simulation rewritten in Rust on the
//! CPU.
//!
//! Set 1 (sources) orbits a Kepler point mass at the origin. Set 2 (tracers) is
//! then integrated through that same central mass *plus* the summed potentials
//! of set 1, each source carried along its own integrated trajectory. That is
//! exactly the shape of the reference simulation — sources first, then tracers
//! in the field of the sources — and it is carried here by the ported CPU
//! DOP853 and nothing else. Each source trajectory is splined for the tracers
//! with the same natural cubic spline galpy uses for a moving object.
//!
//! Nothing is configurable. There is no dispatch layer, no Python binding, no
//! fixture and no input: the initial conditions, masses, output grid and
//! tolerances are constants in this file. The only claim being made is that the
//! Rust port can carry the whole simulation, moving perturbers included.
//!
//! `cargo test --test kepler_showcase -- --nocapture` prints the run.
//! `cargo test --test kepler_showcase -- --ignored` instead writes the same
//! trajectories to `tmp/rust_moving_kepler.csv`, which is how the run is
//! compared against galpy's implementation of the same simulation.

use drift_rs::integrators::galpy::LogTolerance;
use drift_rs::integrators::galpy::dop853::{dop853, integrate_kepler, potentialArg};
use libc::{c_double, c_int};
use shared::{Real, Tolerance};

/// Cartesian phase space: x, y, z, vx, vy, vz.
const DIM: usize = 6;
/// The only thing at the origin.
const CENTRAL_AMP: Real = 1.0;
/// Mass of each source particle. Small enough that the tracers stay bound.
const SOURCE_AMP: Real = 1.0e-3;

/// Output grid.
const START: Real = 0.0;
const END: Real = 20.0;
const STEPS: usize = 401;

/// galpy's "derive the initial step yourself" sentinel; the port ignores `_dt`.
const INITIAL_STEP_SENTINEL: Real = -9999.99;

type Vector = [Real; 3];
type State = [Real; DIM];

/// Set 1: three equal point masses 120° apart on one circular orbit, so they
/// never collide and the tracers see a genuinely time-dependent field.
const SOURCE_RADIUS: Real = 1.0;
const SOURCE_ANGLES: [Real; 3] = [
    0.0,
    core::f64::consts::TAU / 3.0,
    2.0 * core::f64::consts::TAU / 3.0,
];

/// Set 2: one tracer inside the source ring, one outside it.
const TRACER_RADII: [Real; 2] = [0.8, 1.6];
const TRACER_ANGLES: [Real; 2] = [0.0, core::f64::consts::PI];

/// A circular orbit about the central mass, `radius` out at `angle`.
fn circular_orbit(radius: Real, angle: Real) -> State {
    let speed = (CENTRAL_AMP / radius).sqrt();
    let (sin, cos) = angle.sin_cos();
    [
        radius * cos,
        radius * sin,
        0.0,
        -speed * sin,
        speed * cos,
        0.0,
    ]
}

fn output_times() -> Vec<Real> {
    let step = (END - START) / (STEPS - 1) as Real;
    (0..STEPS)
        .map(|index| START + index as Real * step)
        .collect()
}

fn tolerance() -> LogTolerance {
    LogTolerance::from_linear(Tolerance::new(1.0e-10, 1.0e-12))
}

/// Kepler acceleration from a point mass of `amplitude` displaced by `offset`.
fn point_mass(amplitude: Real, offset: Vector) -> Vector {
    let radius_squared = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
    let inverse_cubed = 1.0 / (radius_squared * radius_squared.sqrt());
    [
        -amplitude * offset[0] * inverse_cubed,
        -amplitude * offset[1] * inverse_cubed,
        -amplitude * offset[2] * inverse_cubed,
    ]
}

/// The summed potential of set 1, sampled on the output grid.
///
/// The sources move, so each trajectory is interpolated in time. That spline is
/// the one galpy uses for a moving object: a natural cubic spline, which is what
/// `initMovingObjectSplines` builds with `gsl_interp_cspline`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MovingSources<'a> {
    /// `[step][source][x, y, z]` samples, row-major.
    table: &'a [Real],
    /// Natural cubic spline second derivatives for `table`, same layout.
    second_derivatives: &'a [Real],
    count: usize,
    steps: usize,
    start: Real,
    step: Real,
    mass: Real,
}

impl MovingSources<'_> {
    /// Natural cubic interpolation of one source position at `time`.
    fn origin(&self, time: Real, source: usize) -> Vector {
        let position = ((time - self.start) / self.step).clamp(0.0, (self.steps - 1) as Real);
        let lower = (position.floor() as usize).min(self.steps - 2);
        let fraction = position - lower as Real;
        let sample = |component: usize| {
            let first = (lower * self.count + source) * 3 + component;
            let second = ((lower + 1) * self.count + source) * 3 + component;
            let value = self.table[first];
            let next_value = self.table[second];
            let curvature = self.second_derivatives[first];
            let next_curvature = self.second_derivatives[second];
            // The samples are equally spaced, so the abscissa spacing is one
            // index unit and the cubic reduces to this.
            let slope = (next_value - value) - (2.0 * curvature + next_curvature) / 6.0;
            let quadratic = curvature / 2.0;
            let cubic = (next_curvature - curvature) / 6.0;
            value + fraction * (slope + fraction * (quadratic + fraction * cubic))
        };
        [sample(0), sample(1), sample(2)]
    }

    /// Summed acceleration of set 1 at `position` and `time`.
    fn acceleration(&self, time: Real, position: Vector) -> Vector {
        let mut total = [0.0; 3];
        for source in 0..self.count {
            let origin = self.origin(time, source);
            let contribution = point_mass(
                self.mass,
                [
                    position[0] - origin[0],
                    position[1] - origin[1],
                    position[2] - origin[2],
                ],
            );
            for (total_component, contribution_component) in total.iter_mut().zip(contribution) {
                *total_component += contribution_component;
            }
        }
        total
    }
}

/// Natural cubic spline second derivatives for every series of a
/// `[step][series][component]` table.
///
/// This is the interpolant galpy uses for a moving object — `gsl_interp_cspline`
/// is a cubic spline with natural boundary conditions — so the second
/// derivatives are zero at both ends and, at interior nodes,
///
/// ```text
/// m[i-1] + 4 m[i] + m[i+1] = 6 (y[i+1] - 2 y[i] + y[i-1])
/// ```
///
/// because the samples are equally spaced and the abscissa spacing is therefore
/// one index unit. The system is solved with a Thomas sweep.
fn natural_cubic_second_derivatives(table: &[Real], count: usize, steps: usize) -> Vec<Real> {
    assert!(
        steps >= 3,
        "a natural cubic spline needs at least three samples, got {steps}"
    );

    let stride = count * 3;
    let mut second_derivatives = vec![0.0; table.len()];
    // Modified super-diagonal and right-hand side, one entry per interior node.
    let mut super_diagonal = vec![0.0; steps];
    let mut right_hand_side = vec![0.0; steps];

    for series in 0..stride {
        let sample = |step: usize| table[step * stride + series];

        // Forward sweep over the interior unknowns.
        for interior in 0..steps - 2 {
            let node = interior + 1;
            let value = 6.0 * (sample(node + 1) - 2.0 * sample(node) + sample(node - 1));
            // The first row has no lower coefficient: the boundary second
            // derivative it would multiply is fixed at zero.
            let denominator = if interior == 0 {
                4.0
            } else {
                4.0 - super_diagonal[interior - 1]
            };
            super_diagonal[interior] = 1.0 / denominator;
            right_hand_side[interior] = if interior == 0 {
                value / denominator
            } else {
                (value - right_hand_side[interior - 1]) / denominator
            };
        }

        // Back substitution. The last row's upper coefficient is likewise
        // dropped, which the zero seed does for free.
        let mut following = 0.0;
        for interior in (0..steps - 2).rev() {
            let value = right_hand_side[interior] - super_diagonal[interior] * following;
            second_derivatives[(interior + 1) * stride + series] = value;
            following = value;
        }
    }

    second_derivatives
}

/// Set 1 integrated through the central mass alone, one trajectory per source.
fn integrate_sources(times: &[Real]) -> Vec<Vec<Real>> {
    let tolerance = tolerance();
    SOURCE_ANGLES
        .iter()
        .map(|angle| {
            integrate_kepler(
                circular_orbit(SOURCE_RADIUS, *angle),
                times,
                INITIAL_STEP_SENTINEL,
                0,
                tolerance.rtol,
                tolerance.atol,
            )
        })
        .collect()
}

/// Packs per-source trajectories into the `[step][source][x, y, z]` table the
/// right-hand side interpolates.
fn source_position_table(trajectories: &[Vec<Real>], steps: usize) -> Vec<Real> {
    let count = trajectories.len();
    let mut table = vec![0.0; steps * count * 3];
    for (source, trajectory) in trajectories.iter().enumerate() {
        let (states, remainder) = trajectory.as_chunks::<DIM>();
        debug_assert!(remainder.is_empty());
        for (step, state) in states.iter().enumerate() {
            let base = (step * count + source) * 3;
            table[base..base + 3].copy_from_slice(&state[..3]);
        }
    }
    table
}

/// The right-hand side of set 2: the central mass plus the summed moving
/// potentials of set 1.
extern "C" fn central_plus_sources(
    time: c_double,
    q: *mut c_double,
    a: *mut c_double,
    _nargs: c_int,
    sources: *mut potentialArg,
) {
    // SAFETY: dop853 passes back the buffers and argument pointer given to it.
    let (state, acceleration, sources) = unsafe {
        (
            core::slice::from_raw_parts(q, DIM),
            core::slice::from_raw_parts_mut(a, DIM),
            &*(sources as *const MovingSources<'_>),
        )
    };

    let position: Vector = [state[0], state[1], state[2]];
    let mut force = point_mass(CENTRAL_AMP, position);
    let from_sources = sources.acceleration(time, position);
    for (component, contribution) in force.iter_mut().zip(from_sources) {
        *component += contribution;
    }

    acceleration[..3].copy_from_slice(&[state[3], state[4], state[5]]);
    acceleration[3..].copy_from_slice(&force);
}

/// Integrates one tracer through the central mass plus set 1.
fn integrate_tracer(
    initial_state: State,
    sources: &MovingSources<'_>,
    times: &[Real],
) -> Vec<Real> {
    let tolerance = tolerance();
    let mut initial_state = initial_state;
    let mut times = times.to_vec();
    let mut result = vec![0.0; times.len() * DIM];
    let mut error = 0;

    // SAFETY: the callback reads only `sources`, which outlives the call.
    unsafe {
        dop853(
            Some(central_plus_sources),
            DIM as c_int,
            initial_state.as_mut_ptr(),
            times.len() as c_int,
            INITIAL_STEP_SENTINEL,
            times.as_mut_ptr(),
            0,
            sources as *const MovingSources<'_> as *mut potentialArg,
            tolerance.rtol,
            tolerance.atol,
            result.as_mut_ptr(),
            &mut error,
        );
    }

    assert_eq!(error, 0, "dop853 reported err={error}");
    result
}

fn radius(state: &[Real]) -> Real {
    (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt()
}

fn worst_difference(actual: &[Real], expected: &[Real]) -> Real {
    assert_eq!(actual.len(), expected.len(), "trajectory length mismatch");
    let mut worst: Real = 0.0;
    for (actual, expected) in actual.iter().zip(expected) {
        worst = worst.max((actual - expected).abs());
    }
    worst
}

/// Everything one run of the showcase produces.
struct Simulation {
    times: Vec<Real>,
    /// `[source][step][x, y, z, vx, vy, vz]`.
    source_states: Vec<Vec<Real>>,
    /// `[tracer][step][x, y, z, vx, vy, vz]`, the central mass alone.
    unheated: Vec<Vec<Real>>,
    /// The same tracers through the central mass plus set 1.
    heated: Vec<Vec<Real>>,
    /// Worst gap between an empty source ring and the unheated run.
    massless_error: Real,
}

fn simulate() -> Simulation {
    let times = output_times();
    let tolerance = tolerance();
    let source_states = integrate_sources(&times);
    let source_table = source_position_table(&source_states, times.len());
    let second_derivatives =
        natural_cubic_second_derivatives(&source_table, SOURCE_ANGLES.len(), times.len());
    let sources = MovingSources {
        table: &source_table,
        second_derivatives: &second_derivatives,
        count: SOURCE_ANGLES.len(),
        steps: times.len(),
        start: START,
        step: (END - START) / (STEPS - 1) as Real,
        mass: SOURCE_AMP,
    };

    // Set 2, unheated: the central mass alone, straight through the ported
    // integrator with no perturbers at all.
    let unheated: Vec<Vec<Real>> = TRACER_RADII
        .iter()
        .zip(TRACER_ANGLES)
        .map(|(radius, angle)| {
            integrate_kepler(
                circular_orbit(*radius, angle),
                &times,
                INITIAL_STEP_SENTINEL,
                0,
                tolerance.rtol,
                tolerance.atol,
            )
        })
        .collect();

    // Set 2, heated: the same tracers through the central mass plus set 1.
    let heated: Vec<Vec<Real>> = TRACER_RADII
        .iter()
        .zip(TRACER_ANGLES)
        .map(|(radius, angle)| integrate_tracer(circular_orbit(*radius, angle), &sources, &times))
        .collect();

    // An empty source ring must reproduce the unheated run: the only thing the
    // tracers feel beyond the central mass is set 1 itself.
    let massless = MovingSources {
        mass: 0.0,
        ..sources
    };
    let massless_tracer = integrate_tracer(
        circular_orbit(TRACER_RADII[0], TRACER_ANGLES[0]),
        &massless,
        &times,
    );
    let massless_error = worst_difference(&massless_tracer, &unheated[0]);

    Simulation {
        times,
        source_states,
        unheated,
        heated,
        massless_error,
    }
}

#[test]
fn kepler_two_set_showcase() {
    let simulation = simulate();
    let times = &simulation.times;

    // Zero-mass sources reproduce the unheated run.
    assert!(
        simulation.massless_error <= 1.0e-9,
        "zero-mass sources must reproduce the unheated run: {}",
        simulation.massless_error
    );

    // Set 1 stays on its circular orbit, i.e. set 1 really did orbit the
    // central mass and nothing else.
    let mut source_error: Real = 0.0;
    for trajectory in &simulation.source_states {
        let (states, remainder) = trajectory.as_chunks::<DIM>();
        debug_assert!(remainder.is_empty());
        for state in states {
            source_error = source_error.max((radius(state) - SOURCE_RADIUS).abs());
        }
    }
    assert!(
        source_error <= 1.0e-6,
        "sources left their circular orbit by {source_error}"
    );

    // Set 2 on the unheated run stays circular for the same reason.
    let mut unheated_error: Real = 0.0;
    for (trajectory, radius_expected) in simulation.unheated.iter().zip(TRACER_RADII) {
        let (states, remainder) = trajectory.as_chunks::<DIM>();
        debug_assert!(remainder.is_empty());
        for state in states {
            unheated_error = unheated_error.max((radius(state) - radius_expected).abs());
        }
    }
    assert!(
        unheated_error <= 1.0e-6,
        "unheated tracers left their circular orbit by {unheated_error}"
    );

    // And set 2 does feel set 1.
    let displacement = worst_difference(&simulation.heated[0], &simulation.unheated[0]);
    assert!(
        displacement > 1.0e-3,
        "the moving sources did not perturb the tracer: {displacement}"
    );

    println!(
        "kepler two-set showcase | central={CENTRAL_AMP} source={SOURCE_AMP} \
         sources={} tracers={} {START}..{END} x{STEPS}",
        SOURCE_ANGLES.len(),
        TRACER_RADII.len(),
    );
    println!(
        "  sources deviate from r={SOURCE_RADIUS} by at most {source_error:.3e}; \
         unheated tracers from circular by at most {unheated_error:.3e}"
    );
    println!(
        "  empty ring vs unheated: {:.3e}",
        simulation.massless_error
    );
    for (index, (radius_expected, angle)) in TRACER_RADII.iter().zip(TRACER_ANGLES).enumerate() {
        println!(
            "  tracer {index} (r={radius_expected}, angle={angle}): \
             max |heated - unheated| = {:.3e}",
            worst_difference(&simulation.heated[index], &simulation.unheated[index]),
        );
    }
    println!("  t        x_unheated  y_unheated  x_heated    y_heated");
    for step in (0..STEPS).step_by(STEPS / 4) {
        let state = &simulation.heated[0][step * DIM..step * DIM + DIM];
        let baseline = &simulation.unheated[0][step * DIM..step * DIM + DIM];
        println!(
            "  {:>7.3}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            times[step], baseline[0], baseline[1], state[0], state[1],
        );
    }
}

/// Writes the run to `tmp/rust_moving_kepler.csv` so it can be compared with
/// galpy's implementation of the same simulation.
///
/// `cargo test --test kepler_showcase -- --ignored --nocapture`
#[test]
#[ignore = "writes a dump file instead of asserting"]
fn dump_trajectories() {
    let simulation = simulate();
    let mut lines = String::from("kind,index,t,x,y,z,vx,vy,vz\n");

    for (kind, trajectories) in [
        ("source", &simulation.source_states),
        ("unheated", &simulation.unheated),
        ("heated", &simulation.heated),
    ] {
        for (index, trajectory) in trajectories.iter().enumerate() {
            let (states, remainder) = trajectory.as_chunks::<DIM>();
            debug_assert!(remainder.is_empty());
            for (step, state) in states.iter().enumerate() {
                let values: Vec<String> =
                    state.iter().map(|value| format!("{value:.17e}")).collect();
                lines.push_str(&format!(
                    "{kind},{index},{:.17e},{}\n",
                    simulation.times[step],
                    values.join(",")
                ));
            }
        }
    }

    let path = std::path::Path::new("tmp/rust_moving_kepler.csv");
    std::fs::write(path, lines).expect("could not write the trajectory dump");
    println!("wrote {}", path.display());
}
