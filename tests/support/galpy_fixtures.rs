use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use drift_rs::dispatch::DispatchError;
use drift_rs::integrators::galpy::LogTolerance;
use drift_rs::state::{InputState, OutputState};
use libc::c_int;
use shared::{INPUT_STATE_DIM, Index, ModelComponent, OutputGrid, Real};

const CORPUS_CASES: usize = 100;

type CpuIntegrator = fn(&GalpyFixture) -> Vec<Real>;
type GpuLauncher = fn(
    &ModelComponent,
    &InputState,
    LogTolerance,
    OutputGrid,
    Option<Vec<Real>>,
) -> Result<OutputState, DispatchError>;

#[derive(Clone, Copy)]
pub struct GalpyFixtureSuite {
    name: &'static str,
    fixture_dir: &'static str,
    gpu_dump_env: &'static str,
    cpu_integrator: CpuIntegrator,
    gpu_launcher: GpuLauncher,
}

impl GalpyFixtureSuite {
    pub const fn new(
        name: &'static str,
        fixture_dir: &'static str,
        gpu_dump_env: &'static str,
        cpu_integrator: CpuIntegrator,
        gpu_launcher: GpuLauncher,
    ) -> Self {
        Self {
            name,
            fixture_dir,
            gpu_dump_env,
            cpu_integrator,
            gpu_launcher,
        }
    }

    pub fn reference(self) -> GalpyFixture {
        let path = Path::new(self.fixture_dir).join("reference.fixture");
        GalpyFixture::read(&path)
            .unwrap_or_else(|error| panic!("could not parse {}: {error}", path.display()))
    }

    pub fn integrate_cpu(self, fixture: &GalpyFixture) -> Vec<Real> {
        (self.cpu_integrator)(fixture)
    }

    pub fn integrate_gpu(self, fixture: &GalpyFixture) -> Vec<Real> {
        let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
        let output = OutputGrid {
            start: fixture.times[0],
            end: *fixture.times.last().expect("fixture has no output times"),
            steps: fixture.nt as Index,
        };

        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = fixture.initial_state.len() as Index / fixture.dim as Index;
        input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);

        let model_component = ModelComponent(core::array::from_fn(|_| None));
        let output_state = (self.gpu_launcher)(
            &model_component,
            &input_state,
            tolerance,
            output,
            Some(fixture.times.clone()),
        )
        .unwrap_or_else(|error| panic!("{} kernel launch failed: {error}", self.name));

        output_state.data[..fixture.nt * fixture.dim].to_vec()
    }

    pub fn assert_cpu_corpus(self) {
        self.for_each_corpus_case(|case_name, fixture| {
            let result = self.integrate_cpu(fixture);
            assert_all_state_bits(case_name, &result, fixture);
        });
    }

    pub fn assert_gpu_corpus(self) {
        self.for_each_corpus_case(|case_name, fixture| {
            let result = self.integrate_gpu(fixture);
            assert_all_state_bits(case_name, &result, fixture);
        });
    }

    pub fn report_gpu_corpus_errors(self) {
        let mut summary = ErrorSummary::default();
        let mut expected_comparisons = 0;
        let mut dump = std::env::var_os(self.gpu_dump_env).map(|path| {
            BufWriter::new(
                File::create(&path)
                    .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display())),
            )
        });

        self.for_each_corpus_case(|case_name, fixture| {
            let result = self.integrate_gpu(fixture);
            expected_comparisons += fixture.expected_state_bits.len();

            if let Some(writer) = dump.as_mut() {
                for value in &result {
                    writer
                        .write_all(&value.to_bits().to_le_bytes())
                        .expect("failed to write GPU output dump");
                }
            }

            summary.observe(case_name, &result, fixture);
        });

        if let Some(mut writer) = dump {
            writer.flush().expect("failed to flush GPU output dump");
        }

        println!("gpu/native galpy summary: {summary}");
        assert_eq!(summary.compared, expected_comparisons);
        assert!(summary.max_abs.is_finite());
    }

    fn for_each_corpus_case(self, mut test: impl FnMut(&str, &GalpyFixture)) {
        for path in self.corpus_paths() {
            let case_name = path
                .file_stem()
                .expect("fixture path has no file name")
                .to_string_lossy();
            let fixture = GalpyFixture::read(&path)
                .unwrap_or_else(|error| panic!("could not parse {}: {error}", path.display()));
            test(&case_name, &fixture);
        }
    }

    fn corpus_paths(self) -> Vec<PathBuf> {
        let mut paths: Vec<_> = fs::read_dir(self.fixture_dir)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", self.fixture_dir))
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
            "expected {CORPUS_CASES} native galpy {} fixtures",
            self.name,
        );
        paths
    }
}

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

impl GalpyFixture {
    pub fn initial_state_array(&self) -> [Real; INPUT_STATE_DIM] {
        self.initial_state
            .as_slice()
            .try_into()
            .expect("fixture state dimension is not INPUT_STATE_DIM")
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

    pub fn mismatched(&self) -> usize {
        self.mismatched
    }

    pub fn max_abs(&self) -> Real {
        self.max_abs
    }

    fn observe(&mut self, case_name: &str, result: &[Real], fixture: &GalpyFixture) {
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
