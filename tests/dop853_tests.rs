#[cfg(test)]
mod tests {
    use drift_rs::dispatch::gpu::launch_dop853_kernel;
    use drift_rs::integrators::dop853_cpu::{dop853, potentialArg};
    use drift_rs::state::InputState;
    use libc::{c_double, c_int};
    use shared::{Config, Index, ModelComponent, Tolerance};
    use std::fs::{self, File};
    use std::io::{self, Read};
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::ptr;

    const GALPY_NATIVE_FIXTURE_DIR: &str = "tests/fixtures/dop853_galpy_native";

    #[link(name = "m")]
    unsafe extern "C" {
        fn pow(x: c_double, y: c_double) -> c_double;
        fn sqrt(x: c_double) -> c_double;
    }

    extern "C" fn galpy_kepler_rhs(
        _t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        _nargs: c_int,
        _potential_args: *mut potentialArg,
    ) {
        unsafe {
            let x = *q.add(0);
            let y = *q.add(1);
            let z = *q.add(2);
            let vx = *q.add(3);
            let vy = *q.add(4);
            let vz = *q.add(5);

            *a.add(0) = vx;
            *a.add(1) = vy;
            *a.add(2) = vz;

            let radius = sqrt(x * x + y * y);
            let sin_phi = y / radius;
            let cos_phi = x / radius;
            let radius_squared = radius * radius + z * z;
            let radial_force = -radius * pow(radius_squared, -1.5);
            let phi_torque = 0.0;

            *a.add(3) = cos_phi * radial_force - 1.0 / radius * sin_phi * phi_torque;
            *a.add(4) = sin_phi * radial_force + 1.0 / radius * cos_phi * phi_torque;
            *a.add(5) = -z * pow(radius_squared, -1.5);
        }
    }

    #[test]
    fn dop853_cpu_matches_native_galpy_dump() {
        let dump = parse_dump("dop853_init_dump.txt").expect("could not parse DOP853 dump");
        let result = integrate_cpu(&dump);

        assert_all_state_bits("dop853_init_dump.txt", &result, &dump);
    }

    // two differing libdevice pow(x, 1/8) results in this run. Substituting those bits makes all
    // 6,006 GPU state bits match galpy exactly.
    #[test]
    #[ignore = "CUDA device math first differs from native galpy by 1 ULP at step 3, component 3"]
    fn dop853_gpu_matches_native_galpy_dump() {
        let dump = parse_dump("dop853_init_dump.txt").expect("could not parse DOP853 dump");
        let result = integrate_gpu(&dump);

        assert_all_state_bits("dop853_init_dump.txt", &result, &dump);
    }

    #[test]
    fn dop853_gpu_tracks_native_galpy_dump() {
        let dump = parse_dump("dop853_init_dump.txt").expect("could not parse DOP853 dump");
        let result = integrate_gpu(&dump);
        let mut mismatched = 0;
        let mut max_absolute_error = 0.0_f64;

        for (&actual, &expected_bits) in result.iter().zip(&dump.expected_state_bits) {
            mismatched += usize::from(actual.to_bits() != expected_bits);
            max_absolute_error =
                max_absolute_error.max((actual - f64::from_bits(expected_bits)).abs());
        }

        assert!(mismatched > 0, "expected host/device math to differ");
        assert!(
            max_absolute_error < 1.0e-12,
            "GPU/native galpy max absolute error was {max_absolute_error}"
        );
    }

    #[test]
    #[ignore = "local generated native galpy fixture corpus"]
    fn dop853_cpu_matches_native_galpy_fixtures() {
        for path in galpy_native_fixture_paths() {
            let case_name = path.file_stem().unwrap().to_string_lossy();
            let dump = parse_dump(&path).expect("could not parse galpy fixture");
            let result = integrate_cpu(&dump);

            assert_all_state_bits(&case_name, &result, &dump);
        }
    }

    #[test]
    #[ignore = "native galpy C/libm may not be bit-identical to CUDA device math"]
    fn dop853_gpu_matches_native_galpy_fixtures() {
        for path in galpy_native_fixture_paths() {
            let case_name = path.file_stem().unwrap().to_string_lossy();
            let dump = parse_dump(&path).expect("could not parse galpy fixture");
            let result = integrate_gpu(&dump);

            assert_all_state_bits(&case_name, &result, &dump);
        }
    }

    #[test]
    #[ignore = "diagnostic report for host libm versus CUDA device math drift"]
    fn dop853_gpu_native_galpy_fixture_error_summary() {
        let mut summary = ErrorSummary::default();
        let mut output_dump = std::env::var_os("ASTRODRIFT_DOP853_GPU_DUMP").map(|path| {
            BufWriter::new(
                File::create(&path)
                    .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display())),
            )
        });

        for path in galpy_native_fixture_paths() {
            let case_name = path.file_stem().unwrap().to_string_lossy();
            let dump = parse_dump(&path).expect("could not parse galpy fixture");
            let result = integrate_gpu(&dump);

            if let Some(writer) = output_dump.as_mut() {
                for value in &result {
                    writer
                        .write_all(&value.to_bits().to_le_bytes())
                        .expect("failed to write GPU output dump");
                }
            }

            summary.observe_fixture(&case_name, &result, &dump);
        }

        if let Some(mut writer) = output_dump {
            writer.flush().expect("failed to flush GPU output dump");
        }

        println!(
            "gpu/native galpy summary: compared={} mismatched={} max_abs={} at {} step {} component {} got=0x{:016x} expected=0x{:016x} max_rel={} max_ulp={}",
            summary.compared,
            summary.mismatched,
            summary.max_abs,
            summary.max_abs_case,
            summary.max_abs_step,
            summary.max_abs_component,
            summary.max_abs_actual_bits,
            summary.max_abs_expected_bits,
            summary.max_rel,
            summary.max_ulp,
        );

        assert_eq!(summary.compared, 100 * 1000 * 6);
        assert!(summary.max_abs.is_finite());
    }

    fn integrate_cpu(dump: &DumpData) -> Vec<c_double> {
        assert_eq!(dump.t.len(), dump.nt as usize, "t length != nt");
        assert_eq!(dump.y0.len(), dump.dim as usize, "y0 length != dim");
        assert_eq!(
            dump.expected_state_bits.len(),
            dump.nt as usize * dump.dim as usize,
            "native state length mismatch"
        );

        let mut t = dump.t.clone();
        let mut y0 = dump.y0.clone();
        let mut result = vec![0.0; dump.nt as usize * dump.dim as usize];
        let mut err = 0;

        unsafe {
            dop853(
                Some(galpy_kepler_rhs),
                dump.dim,
                y0.as_mut_ptr(),
                dump.nt,
                dump.dt_one,
                t.as_mut_ptr(),
                dump.nargs,
                ptr::null_mut(),
                dump.rtol,
                dump.atol,
                result.as_mut_ptr(),
                &mut err,
            );
        }
        assert_eq!(err, 0, "dop853 returned err={err}");

        result
    }

    fn integrate_gpu(dump: &DumpData) -> Vec<c_double> {
        assert_eq!(dump.t.len(), dump.nt as usize, "t length != nt");
        assert_eq!(dump.y0.len(), dump.dim as usize, "y0 length != dim");

        let mut config = Config::default();
        config.settings.tolerance = Tolerance {
            rtol: dump.rtol,
            atol: dump.atol,
        };
        config.settings.ts.end = *dump.t.last().unwrap();
        config.settings.ts.steps = dump.nt as Index;

        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = dump.y0.len() as Index / dump.dim as Index;
        for (index, value) in dump.y0.iter().enumerate() {
            input_state.data[index] = *value;
        }

        let model_component = ModelComponent(core::array::from_fn(|_| None));
        let output_state = launch_dop853_kernel(
            &model_component,
            &input_state,
            config.flags,
            config.settings.tolerance,
            config.settings.ts,
            Some(dump.t.clone()),
            &[0], // single-GPU: match the device list the dispatch layer uses
        )
        .expect("kernel launch failed");

        let len = dump.nt as usize * dump.dim as usize;
        output_state.data[..len].to_vec()
    }

    fn assert_all_state_bits(case_name: &str, result: &[c_double], dump: &DumpData) {
        for (index, (&actual, &expected)) in result
            .iter()
            .zip(dump.expected_state_bits.iter())
            .enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected,
                "{case_name}: mismatch at step {}, component {}: got 0x{:016x}, expected 0x{:016x}",
                index / dump.dim as usize,
                index % dump.dim as usize,
                actual.to_bits(),
                expected,
            );
        }
    }

    #[derive(Default)]
    struct ErrorSummary {
        compared: usize,
        mismatched: usize,
        max_abs: f64,
        max_rel: f64,
        max_ulp: u64,
        max_abs_case: String,
        max_abs_step: usize,
        max_abs_component: usize,
        max_abs_actual_bits: u64,
        max_abs_expected_bits: u64,
    }

    impl ErrorSummary {
        fn observe_fixture(&mut self, case_name: &str, result: &[c_double], dump: &DumpData) {
            assert_eq!(
                dump.expected_state_bits.len(),
                result.len(),
                "{case_name}: expected state length mismatch"
            );

            for (index, (&actual, &expected_bits)) in result
                .iter()
                .zip(dump.expected_state_bits.iter())
                .enumerate()
            {
                self.compared += 1;

                let actual_bits = actual.to_bits();
                if actual_bits != expected_bits {
                    self.mismatched += 1;
                }

                let expected = f64::from_bits(expected_bits);
                let absolute_error = (actual - expected).abs();
                let relative_error = if expected == 0.0 {
                    absolute_error
                } else {
                    absolute_error / expected.abs()
                };
                let ulp_error = ulp_distance(actual_bits, expected_bits);

                self.max_rel = self.max_rel.max(relative_error);
                self.max_ulp = self.max_ulp.max(ulp_error);

                if absolute_error > self.max_abs {
                    self.max_abs = absolute_error;
                    self.max_abs_case = case_name.to_string();
                    self.max_abs_step = index / dump.dim as usize;
                    self.max_abs_component = index % dump.dim as usize;
                    self.max_abs_actual_bits = actual_bits;
                    self.max_abs_expected_bits = expected_bits;
                }
            }
        }
    }

    fn ulp_distance(a: u64, b: u64) -> u64 {
        ordered_f64_bits(a).abs_diff(ordered_f64_bits(b))
    }

    fn ordered_f64_bits(bits: u64) -> u64 {
        if bits & (1 << 63) == 0 {
            bits | (1 << 63)
        } else {
            !bits
        }
    }

    fn galpy_native_fixture_paths() -> Vec<PathBuf> {
        let mut paths: Vec<_> = fs::read_dir(GALPY_NATIVE_FIXTURE_DIR)
            .unwrap_or_else(|error| panic!("could not read {GALPY_NATIVE_FIXTURE_DIR}: {error}"))
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "fixture")
            })
            .collect();
        paths.sort();

        assert_eq!(
            paths.len(),
            100,
            "expected 100 native galpy DOP853 fixtures"
        );
        paths
    }

    struct DumpData {
        dim: c_int,
        nt: c_int,
        dt_one: c_double,
        rtol: c_double,
        atol: c_double,
        t: Vec<c_double>,
        y0: Vec<c_double>,
        nargs: c_int,
        expected_state_bits: Vec<u64>,
    }

    fn parse_dump<P: AsRef<Path>>(path: P) -> io::Result<DumpData> {
        let mut text = String::new();
        File::open(path)?.read_to_string(&mut text)?;

        let mut dim = None;
        let mut nt = None;
        let mut dt_one = None;
        let mut rtol = None;
        let mut atol = None;
        let mut t = Vec::new();
        let mut y0 = Vec::new();
        let mut nargs = 0;
        let mut expected_state_bits = Vec::new();

        for line in text.lines() {
            let mut parts = line.split_whitespace();
            let Some(key) = parts.next() else {
                continue;
            };

            match key {
                "dim" => dim = parts.next().map(|value| value.parse().unwrap()),
                "nt" => nt = parts.next().map(|value| value.parse().unwrap()),
                "dt_one" => dt_one = parts.next().map(|value| value.parse().unwrap()),
                "rtol" => rtol = parts.next().map(|value| value.parse().unwrap()),
                "atol" => atol = parts.next().map(|value| value.parse().unwrap()),
                "t_hex" => t.extend(parts.map(parse_hex_f64)),
                "yo_hex" => y0.extend(parts.map(parse_hex_f64)),
                "nargs" => nargs = parts.next().unwrap().parse().unwrap(),
                "state_hex" => {
                    let _step = parts.next();
                    expected_state_bits.extend(parts.map(parse_hex_bits));
                }
                _ => {}
            }
        }

        Ok(DumpData {
            dim: dim.expect("dim missing in dump"),
            nt: nt.expect("nt missing in dump"),
            dt_one: dt_one.expect("dt_one missing in dump"),
            rtol: rtol.expect("rtol missing in dump"),
            atol: atol.expect("atol missing in dump"),
            t,
            y0,
            nargs,
            expected_state_bits,
        })
    }

    fn parse_hex_f64(value: &str) -> c_double {
        c_double::from_bits(parse_hex_bits(value))
    }

    fn parse_hex_bits(value: &str) -> u64 {
        u64::from_str_radix(value, 16)
            .unwrap_or_else(|error| panic!("failed to parse hex f64 '{value}': {error}"))
    }
}
