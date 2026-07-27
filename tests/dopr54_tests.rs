#[cfg(test)]
mod tests {
    use drift_rs::dispatch::gpu::launch_kernel;
    use drift_rs::integrators::dopr54_cpu::*;
    use drift_rs::state::InputState;
    use libc::{self, c_double, c_int};
    use libm::sqrt;
    use shared::{Config, Index, ModelComponent, Tolerance};
    #[cfg(feature = "galpy-kepler-reference")]
    use std::fs;
    use std::fs::File;
    use std::io::{self, Read};
    #[cfg(feature = "galpy-kepler-reference")]
    use std::io::{BufWriter, Write};
    use std::path::Path;
    #[cfg(feature = "galpy-kepler-reference")]
    use std::path::PathBuf;
    use std::ptr;

    extern "C" fn kepler_rhs(
        _t: c_double,
        q: *mut c_double,
        a: *mut c_double,
        _nargs: c_int,
        _pot_args: *mut potentialArg,
    ) {
        unsafe {
            let x = *q.add(0);
            let y = *q.add(1);
            let z = *q.add(2);
            let vx = *q.add(3);
            let vy = *q.add(4);
            let vz = *q.add(5);

            let (ax, ay, az) = reference_kepler_force(x, y, z);

            *a.add(0) = vx;
            *a.add(1) = vy;
            *a.add(2) = vz;

            *a.add(3) = ax;
            *a.add(4) = ay;
            *a.add(5) = az;
        }
    }

    #[cfg(not(feature = "galpy-kepler-reference"))]
    fn reference_kepler_force(
        x: c_double,
        y: c_double,
        z: c_double,
    ) -> (c_double, c_double, c_double) {
        let r2 = x * x + y * y + z * z;
        let r2_safe = if r2 == 0.0 { 1e-16 } else { r2 };
        let r = sqrt(r2_safe);
        let inv_r3 = 1.0 / (r2_safe * r);

        (-x * inv_r3, -y * inv_r3, -z * inv_r3)
    }

    #[cfg(feature = "galpy-kepler-reference")]
    fn reference_kepler_force(
        x: c_double,
        y: c_double,
        z: c_double,
    ) -> (c_double, c_double, c_double) {
        let r = sqrt(x * x + y * y);
        let sinphi = y / r;
        let cosphi = x / r;
        let r2 = r * r + z * z;
        let rforce = -r * r2.powf(-1.5);
        let phitorque = 0.0;
        let ax = cosphi * rforce - 1.0 / r * sinphi * phitorque;
        let ay = sinphi * rforce + 1.0 / r * cosphi * phitorque;
        let az = -z * r2.powf(-1.5);

        (ax, ay, az)
    }

    #[cfg(not(feature = "galpy-kepler-reference"))]
    fn expected_tail_bits() -> [u64; 6] {
        [
            0xbfead9ac890cbf34,
            0xbfe1689ef5f2f595,
            0x0000000000000000,
            0x3fe1689ef5f2da49,
            0xbfead9ac890ca8be,
            0x0000000000000000,
        ]
    }

    #[cfg(feature = "galpy-kepler-reference")]
    fn expected_tail_bits() -> [u64; 6] {
        [
            0xbfead9ac890cbf36,
            0xbfe1689ef5f2f535,
            0x0000000000000000,
            0x3fe1689ef5f2da37,
            0xbfead9ac890ca90a,
            0x0000000000000000,
        ]
    }

    #[cfg(feature = "galpy-kepler-reference")]
    const GALPY_NATIVE_FIXTURE_DIR: &str = "tests/fixtures/dopr54_galpy_native";

    #[test]
    fn dopr54_cpu_matches_reference() {
        let init = parse_dopr54_dump("dopr54_init_dump.txt").expect("could not parse init dump");
        let result = integrate_cpu(&init);

        assert_tail_bits("dopr54_init_dump.txt", &result, expected_tail_bits());
    }

    #[test]
    fn dopr54_gpu_matches_reference() {
        let init = parse_dopr54_dump("dopr54_init_dump.txt").expect("could not parse init dump");
        let result = integrate_gpu(&init);

        assert_tail_bits("dopr54_init_dump.txt", &result, expected_tail_bits());
    }

    #[cfg(feature = "galpy-kepler-reference")]
    #[test]
    #[ignore = "local generated native galpy fixture corpus"]
    fn dopr54_cpu_matches_native_galpy_fixtures() {
        for path in galpy_native_fixture_paths() {
            let case_name = path.file_stem().unwrap().to_string_lossy();
            let init = parse_dopr54_dump(&path).expect("could not parse galpy fixture");
            let result = integrate_cpu(&init);

            assert_all_state_bits(&case_name, &result, &init);
        }
    }

    #[cfg(feature = "galpy-kepler-reference")]
    // First observed mismatch in the shorter fixture corpus was case_00, step 35,
    // component 5 by 1 ULP.
    #[test]
    #[ignore = "native galpy C/libm is not bit-identical to CUDA device math for non-planar fixtures"]
    fn dopr54_gpu_matches_native_galpy_fixtures() {
        for path in galpy_native_fixture_paths() {
            let case_name = path.file_stem().unwrap().to_string_lossy();
            let init = parse_dopr54_dump(&path).expect("could not parse galpy fixture");
            let result = integrate_gpu(&init);

            assert_all_state_bits(&case_name, &result, &init);
        }
    }

    #[cfg(feature = "galpy-kepler-reference")]
    #[test]
    #[ignore = "diagnostic report for host libm versus CUDA device math drift"]
    fn dopr54_gpu_native_galpy_fixture_error_summary() {
        let mut summary = ErrorSummary::default();
        let mut dump = std::env::var_os("ASTRODRIFT_DOPR54_GPU_DUMP").map(|path| {
            BufWriter::new(
                File::create(&path)
                    .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display())),
            )
        });

        for path in galpy_native_fixture_paths() {
            let case_name = path.file_stem().unwrap().to_string_lossy();
            let init = parse_dopr54_dump(&path).expect("could not parse galpy fixture");
            let result = integrate_gpu(&init);

            if let Some(writer) = dump.as_mut() {
                for value in &result {
                    writer
                        .write_all(&value.to_bits().to_le_bytes())
                        .expect("failed to write GPU output dump");
                }
            }

            summary.observe_fixture(&case_name, &result, &init);
        }

        if let Some(mut writer) = dump {
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

    #[cfg(feature = "galpy-kepler-reference")]
    fn galpy_native_fixture_paths() -> Vec<PathBuf> {
        let mut paths: Vec<_> = fs::read_dir(GALPY_NATIVE_FIXTURE_DIR)
            .unwrap_or_else(|err| panic!("could not read {GALPY_NATIVE_FIXTURE_DIR}: {err}"))
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "fixture"))
            .collect();
        paths.sort();

        assert_eq!(
            paths.len(),
            100,
            "expected 100 galpy native DOPR54 fixtures"
        );

        paths
    }

    #[cfg(feature = "galpy-kepler-reference")]
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

    #[cfg(feature = "galpy-kepler-reference")]
    impl ErrorSummary {
        fn observe_fixture(&mut self, case_name: &str, result: &[c_double], init: &DumpData) {
            assert_eq!(
                init.expected_state_bits.len(),
                result.len(),
                "{case_name}: expected state length mismatch"
            );

            for (i, (&actual_f64, &expected_bits)) in result
                .iter()
                .zip(init.expected_state_bits.iter())
                .enumerate()
            {
                self.compared += 1;

                let actual_bits = actual_f64.to_bits();
                if actual_bits != expected_bits {
                    self.mismatched += 1;
                }

                let expected_f64 = f64::from_bits(expected_bits);
                let abs = (actual_f64 - expected_f64).abs();
                let rel = if expected_f64 == 0.0 {
                    abs
                } else {
                    abs / expected_f64.abs()
                };
                let ulp = ulp_distance(actual_bits, expected_bits);

                self.max_rel = self.max_rel.max(rel);
                self.max_ulp = self.max_ulp.max(ulp);

                if abs > self.max_abs {
                    self.max_abs = abs;
                    self.max_abs_case = case_name.to_string();
                    self.max_abs_step = i / init.dim as usize;
                    self.max_abs_component = i % init.dim as usize;
                    self.max_abs_actual_bits = actual_bits;
                    self.max_abs_expected_bits = expected_bits;
                }
            }
        }
    }

    #[cfg(feature = "galpy-kepler-reference")]
    fn ulp_distance(a: u64, b: u64) -> u64 {
        let a = ordered_f64_bits(a);
        let b = ordered_f64_bits(b);

        a.abs_diff(b)
    }

    #[cfg(feature = "galpy-kepler-reference")]
    fn ordered_f64_bits(bits: u64) -> u64 {
        if bits & (1 << 63) == 0 {
            bits | (1 << 63)
        } else {
            !bits
        }
    }

    struct DumpData {
        dim: c_int,
        nt: c_int,
        dt_one: c_double,
        rtol: c_double,
        atol: c_double,
        t: Vec<c_double>,
        yo: Vec<c_double>,
        nargs: c_int,
        expected_state_bits: Vec<u64>,
    }

    fn integrate_cpu(init: &DumpData) -> Vec<c_double> {
        let dim = init.dim;
        let nt = init.nt;
        let nargs = init.nargs;
        assert_eq!(init.t.len(), nt as usize, "t length != nt");
        assert_eq!(init.yo.len(), dim as usize, "yo length != dim");

        let mut t = init.t.clone();
        let mut yo = init.yo.clone();
        let mut result = vec![0.0 as c_double; (nt as usize) * (dim as usize)];
        let mut err: c_int = 0;
        let pot_ptr: *mut potentialArg = ptr::null_mut();

        unsafe {
            dopr54(
                Some(kepler_rhs),
                dim,
                yo.as_mut_ptr(),
                nt,
                init.dt_one,
                t.as_mut_ptr(),
                nargs,
                pot_ptr,
                init.rtol,
                init.atol,
                result.as_mut_ptr(),
                &mut err,
            );
        }
        assert_eq!(err, 0, "dopr54 returned err={err}");

        result
    }

    fn integrate_gpu(init: &DumpData) -> Vec<c_double> {
        assert_eq!(init.t.len(), init.nt as usize, "t length != nt");
        assert_eq!(init.yo.len(), init.dim as usize, "yo length != dim");

        let mut config = Config::default();
        config.settings.tolerance = Tolerance {
            rtol: init.rtol,
            atol: init.atol,
        };
        config.settings.ts.end = *init.t.last().unwrap();
        config.settings.ts.steps = init.nt as Index;

        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = init.yo.len() as Index / init.dim as Index;
        for (i, v) in init.yo.iter().enumerate() {
            input_state.data[i] = *v;
        }

        let model_component = ModelComponent(core::array::from_fn(|_| None));
        let output_state = launch_kernel(
            &model_component,
            &input_state,
            config.flags,
            config.settings.tolerance,
            config.settings.ts,
            Some(init.t.clone()),
        )
        .expect("kernel launch failed");

        let len = (init.nt as usize) * (init.dim as usize);
        output_state.data[..len].to_vec()
    }

    fn assert_tail_bits(case_name: &str, result: &[c_double], expected: [u64; 6]) {
        let tail = &result[result.len() - expected.len()..];

        for (i, &actual_f64) in tail.iter().enumerate() {
            let actual_bits = actual_f64.to_bits();
            let expected_bits = expected[i];

            assert_eq!(
                actual_bits, expected_bits,
                "{case_name}: mismatch in tail element {i}: got 0x{:016x}, expected 0x{:016x}",
                actual_bits, expected_bits
            );
        }
    }

    fn assert_all_state_bits(case_name: &str, result: &[c_double], init: &DumpData) {
        assert_eq!(
            init.expected_state_bits.len(),
            result.len(),
            "{case_name}: expected state length mismatch"
        );

        for (i, (&actual_f64, &expected_bits)) in result
            .iter()
            .zip(init.expected_state_bits.iter())
            .enumerate()
        {
            let actual_bits = actual_f64.to_bits();

            assert_eq!(
                actual_bits,
                expected_bits,
                "{case_name}: mismatch at step {}, component {}: got 0x{:016x}, expected 0x{:016x}",
                i / init.dim as usize,
                i % init.dim as usize,
                actual_bits,
                expected_bits
            );
        }
    }

    fn parse_dopr54_dump<P: AsRef<Path>>(path: P) -> io::Result<DumpData> {
        let mut text = String::new();
        File::open(path)?.read_to_string(&mut text)?;
        parse_dopr54_dump_text(&text)
    }

    fn parse_dopr54_dump_text(text: &str) -> io::Result<DumpData> {
        let mut dim: Option<c_int> = None;
        let mut nt: Option<c_int> = None;
        let mut dt_one: Option<c_double> = None;
        let mut rtol: Option<c_double> = None;
        let mut atol: Option<c_double> = None;
        let mut t: Vec<c_double> = Vec::new();
        let mut yo: Vec<c_double> = Vec::new();
        let mut nargs: c_int = 0;
        let mut args: Vec<c_double> = Vec::new();
        let mut expected_state_bits: Vec<u64> = Vec::new();

        for line in text.lines() {
            let mut parts = line.split_whitespace();
            let key = match parts.next() {
                Some(k) => k,
                None => continue,
            };
            match key {
                "dim" => {
                    if let Some(v) = parts.next() {
                        dim = Some(v.parse().unwrap());
                    }
                }
                "nt" => {
                    if let Some(v) = parts.next() {
                        nt = Some(v.parse().unwrap());
                    }
                }
                "dt_one" => {
                    if let Some(v) = parts.next() {
                        dt_one = Some(v.parse().unwrap());
                    }
                }
                "rtol" => {
                    if let Some(v) = parts.next() {
                        rtol = Some(v.parse().unwrap());
                    }
                }
                "atol" => {
                    if let Some(v) = parts.next() {
                        atol = Some(v.parse().unwrap());
                    }
                }
                "t_hex" => {
                    t.clear();
                    for v in parts {
                        t.push(f64::from_bits(parse_hex_bits(v)));
                    }
                }
                "yo" => {
                    for v in parts {
                        yo.push(v.parse().unwrap());
                    }
                }
                "yo_hex" => {
                    yo.clear();
                    for v in parts {
                        yo.push(f64::from_bits(parse_hex_bits(v)));
                    }
                }
                "nargs" => {
                    if let Some(v) = parts.next() {
                        nargs = v.parse().unwrap();
                    }
                }
                "args" => {
                    for v in parts {
                        args.push(v.parse().unwrap());
                    }
                }
                "state_hex" => {
                    let _step = parts.next();
                    for v in parts {
                        expected_state_bits.push(parse_hex_bits(v));
                    }
                }
                "states_hex" => {}
                _ => {}
            }
        }

        Ok(DumpData {
            dim: dim.expect("dim missing in dump"),
            nt: nt.expect("nt missing in dump"),
            dt_one: dt_one.unwrap_or(-9999.99),
            rtol: rtol.expect("rtol missing in dump"),
            atol: atol.expect("atol missing in dump"),
            t,
            yo,
            nargs,
            expected_state_bits,
        })
    }

    fn parse_hex_bits(s: &str) -> u64 {
        u64::from_str_radix(s, 16)
            .unwrap_or_else(|e| panic!("failed to parse hex f64 '{}': {}", s, e))
    }
}
