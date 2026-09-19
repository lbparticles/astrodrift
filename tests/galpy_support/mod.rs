use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use libc::c_int;
use shared::{INPUT_STATE_DIM, Real};

const CORPUS_CASES: usize = 100;

/// Finds the 100 seeded native-galpy dumps for one integrator in stable order.
pub fn corpus_fixture_paths(directory: &str, integrator: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("could not read {directory}: {error}"))
        .map(|entry| {
            entry
                .expect("could not read fixture directory entry")
                .path()
        })
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("case_") && name.ends_with(".fixture"))
        })
        .collect();
    paths.sort();

    assert_eq!(
        paths.len(),
        CORPUS_CASES,
        "expected {CORPUS_CASES} native galpy {integrator} fixtures",
    );

    paths
}

/// One parsed dump from the instrumented native galpy implementation.
///
/// Galpy records logarithmic tolerances, exact input bits, and, for corpus
/// fixtures, every expected output bit. The canonical DOPR54 reference predates
/// full trajectory dumps, so its expected output is intentionally empty and its
/// tests compare against a separately recorded final state.
pub struct GalpyFixture {
    pub dim: usize,
    pub nt: usize,
    pub dt_one: Real,
    pub rtol: Real,
    pub atol: Real,
    pub times: Vec<Real>,
    pub initial_state: Vec<Real>,
    pub nargs: c_int,
    pub expected_state_bits: Vec<u64>,
}

pub fn assert_tail_bits(case_name: &str, result: &[Real], expected: [u64; INPUT_STATE_DIM]) {
    let tail = &result[result.len() - expected.len()..];

    for (index, (&actual, expected)) in tail.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected,
            "{case_name}: mismatch in tail element {index}: got 0x{:016x}, expected 0x{expected:016x}",
            actual.to_bits(),
        );
    }
}

pub fn assert_all_state_bits(case_name: &str, result: &[Real], fixture: &GalpyFixture) {
    assert_eq!(
        fixture.expected_state_bits.len(),
        result.len(),
        "{case_name}: expected state length mismatch"
    );

    for (index, (&actual, &expected)) in result
        .iter()
        .zip(fixture.expected_state_bits.iter())
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected,
            "{case_name}: mismatch at step {}, component {}: got 0x{:016x}, expected 0x{expected:016x}",
            index / fixture.dim,
            index % fixture.dim,
            actual.to_bits(),
        );
    }
}

/// Accumulates the host/device differences caused by CUDA device math.
#[derive(Default)]
pub struct ErrorSummary {
    compared: usize,
    mismatched: usize,
    max_abs: Real,
    max_rel: Real,
    max_ulp: u64,
    max_abs_case: String,
    max_abs_step: usize,
    max_abs_component: usize,
    max_abs_actual_bits: u64,
    max_abs_expected_bits: u64,
}

impl ErrorSummary {
    pub fn for_fixture(case_name: &str, result: &[Real], fixture: &GalpyFixture) -> Self {
        let mut summary = Self::default();
        summary.observe(case_name, result, fixture);
        summary
    }

    pub fn compared(&self) -> usize {
        self.compared
    }

    pub fn mismatched(&self) -> usize {
        self.mismatched
    }

    pub fn max_abs(&self) -> Real {
        self.max_abs
    }

    pub fn observe(&mut self, case_name: &str, result: &[Real], fixture: &GalpyFixture) {
        assert_eq!(
            fixture.expected_state_bits.len(),
            result.len(),
            "{case_name}: expected state length mismatch"
        );

        for (index, (&actual, &expected_bits)) in result
            .iter()
            .zip(fixture.expected_state_bits.iter())
            .enumerate()
        {
            self.compared += 1;

            let actual_bits = actual.to_bits();
            if actual_bits != expected_bits {
                self.mismatched += 1;
            }

            let expected = Real::from_bits(expected_bits);
            let absolute_error = (actual - expected).abs();
            let relative_error = if expected == 0.0 {
                absolute_error
            } else {
                absolute_error / expected.abs()
            };

            self.max_rel = self.max_rel.max(relative_error);
            self.max_ulp = self.max_ulp.max(ulp_distance(actual_bits, expected_bits));

            if absolute_error > self.max_abs {
                self.max_abs = absolute_error;
                self.max_abs_case = case_name.to_string();
                self.max_abs_step = index / fixture.dim;
                self.max_abs_component = index % fixture.dim;
                self.max_abs_actual_bits = actual_bits;
                self.max_abs_expected_bits = expected_bits;
            }
        }
    }
}

impl std::fmt::Display for ErrorSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "compared={} mismatched={} max_abs={} at {} step {} component {} got=0x{:016x} expected=0x{:016x} max_rel={} max_ulp={}",
            self.compared,
            self.mismatched,
            self.max_abs,
            self.max_abs_case,
            self.max_abs_step,
            self.max_abs_component,
            self.max_abs_actual_bits,
            self.max_abs_expected_bits,
            self.max_rel,
            self.max_ulp,
        )
    }
}

impl GalpyFixture {
    pub fn initial_state_array(&self) -> [Real; INPUT_STATE_DIM] {
        self.initial_state
            .as_slice()
            .try_into()
            .expect("fixture state dimension is not INPUT_STATE_DIM")
    }

    /// Parses a dump and validates the dimensions needed by the integrator tests.
    pub fn load_and_validate(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        Self::read(path)
            .unwrap_or_else(|error| panic!("could not parse {}: {error}", path.display()))
    }

    fn read(path: &Path) -> io::Result<Self> {
        let mut text = String::new();
        File::open(path)?.read_to_string(&mut text)?;

        let mut dim = None;
        let mut nt = None;
        let mut dt_one = None;
        let mut rtol = None;
        let mut atol = None;
        let mut times = Vec::new();
        let mut initial_state = Vec::new();
        let mut nargs = 0;
        let mut expected_state_bits = Vec::new();

        for line in text.lines() {
            let mut parts = line.split_whitespace();
            let Some(key) = parts.next() else {
                continue;
            };

            match key {
                "dim" => dim = parts.next().map(parse_number::<usize>),
                "nt" => nt = parts.next().map(parse_number::<usize>),
                "dt_one" => dt_one = parts.next().map(parse_number::<Real>),
                "rtol" => rtol = parts.next().map(parse_number::<Real>),
                "atol" => atol = parts.next().map(parse_number::<Real>),
                // Older DOPR54 references include decimal rows; generated
                // fixtures also include exact hexadecimal rows, which win.
                "t" if times.is_empty() => times.extend(parts.map(parse_number::<Real>)),
                "t_hex" => {
                    times.clear();
                    times.extend(parts.map(parse_hex_real));
                }
                "yo" if initial_state.is_empty() => {
                    initial_state.extend(parts.map(parse_number::<Real>));
                }
                "yo_hex" => {
                    initial_state.clear();
                    initial_state.extend(parts.map(parse_hex_real));
                }
                "nargs" => {
                    nargs = parse_number::<c_int>(parts.next().expect("nargs has no value"));
                }
                "state_hex" => {
                    let _step = parts.next();
                    expected_state_bits.extend(parts.map(parse_hex_bits));
                }
                _ => {}
            }
        }

        let fixture = Self {
            dim: dim.expect("dim missing in fixture"),
            nt: nt.expect("nt missing in fixture"),
            dt_one: dt_one.expect("dt_one missing in fixture"),
            rtol: rtol.expect("rtol missing in fixture"),
            atol: atol.expect("atol missing in fixture"),
            times,
            initial_state,
            nargs,
            expected_state_bits,
        };
        fixture.validate(path)?;
        Ok(fixture)
    }

    fn validate(&self, path: &Path) -> io::Result<()> {
        let invalid = |message| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {message}", path.display()),
            )
        };

        if self.dim != INPUT_STATE_DIM {
            return Err(invalid(format!(
                "state dimension {} != {INPUT_STATE_DIM}",
                self.dim
            )));
        }
        if self.times.len() != self.nt {
            return Err(invalid(format!(
                "{} output times != nt {}",
                self.times.len(),
                self.nt
            )));
        }
        if self.initial_state.len() != self.dim {
            return Err(invalid(format!(
                "initial state length {} != dimension {}",
                self.initial_state.len(),
                self.dim
            )));
        }
        // Empty output is valid only for an input-only reference. Any present
        // output must contain the complete time-major trajectory.
        if !self.expected_state_bits.is_empty()
            && self.expected_state_bits.len() != self.nt * self.dim
        {
            return Err(invalid(format!(
                "partial expected state: length {} != nt * dimension {}",
                self.expected_state_bits.len(),
                self.nt * self.dim
            )));
        }
        Ok(())
    }
}

fn parse_number<T: std::str::FromStr>(value: &str) -> T
where
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .unwrap_or_else(|error| panic!("failed to parse '{value}': {error}"))
}

fn parse_hex_real(value: &str) -> Real {
    Real::from_bits(parse_hex_bits(value))
}

fn parse_hex_bits(value: &str) -> u64 {
    u64::from_str_radix(value, 16)
        .unwrap_or_else(|error| panic!("failed to parse hex f64 '{value}': {error}"))
}

fn ulp_distance(a: u64, b: u64) -> u64 {
    ordered_real_bits(a).abs_diff(ordered_real_bits(b))
}

fn ordered_real_bits(bits: u64) -> u64 {
    if bits & (1 << 63) == 0 {
        bits | (1 << 63)
    } else {
        !bits
    }
}
