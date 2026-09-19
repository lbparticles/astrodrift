#[cfg(test)]
mod galpy_support;

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::Path;

    use drift_rs::dispatch::gpu::{launch_galpy_dop853, launch_galpy_dopr54};
    use drift_rs::integrators::galpy::{LogTolerance, dop853, dopr54};
    use drift_rs::state::InputState;
    use shared::{Index, ModelComponent, OutputGrid};

    use crate::galpy_support::{
        ErrorSummary, GalpyFixture, assert_all_state_bits, assert_tail_bits, corpus_fixture_paths,
    };

    const DOPR54_FIXTURE_DIR: &str = "tests/fixtures/dopr54_galpy_native";
    const DOPR54_REFERENCE: &str = "tests/fixtures/dopr54_galpy_native/reference.fixture";
    const DOPR54_GPU_DUMP_ENV: &str = "DRIFT_DOPR54_GPU_DUMP";
    const DOP853_FIXTURE_DIR: &str = "tests/fixtures/dop853_galpy_native";
    const DOP853_REFERENCE: &str = "tests/fixtures/dop853_galpy_native/reference.fixture";
    const DOP853_GPU_DUMP_ENV: &str = "DRIFT_DOP853_GPU_DUMP";

    // Previous Cartesian Kepler force tail:
    // [
    //     0xbfead9ac890cbf34,
    //     0xbfe1689ef5f2f595,
    //     0x0000000000000000,
    //     0x3fe1689ef5f2da49,
    //     0xbfead9ac890ca8be,
    //     0x0000000000000000,
    // ]
    const DOPR54_EXPECTED_TAIL_BITS: [u64; 6] = [
        0xbfead9ac890cbf36,
        0xbfe1689ef5f2f535,
        0x0000000000000000,
        0x3fe1689ef5f2da37,
        0xbfead9ac890ca90a,
        0x0000000000000000,
    ];

    // Runs the exact CPU DOPR54 port from the canonical galpy input and checks
    // its final phase-space state against the separately recorded reference.
    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dopr54_cpu_matches_reference() {
        // The old reference records the input but not the full trajectory.
        let fixture = GalpyFixture::load_and_validate(DOPR54_REFERENCE);
        assert!(fixture.expected_state_bits.is_empty());

        // Run the direct CPU port, without the general dispatch layer.
        let result = dopr54::integrate_kepler(
            fixture.initial_state_array(),
            &fixture.times,
            fixture.dt_one,
            fixture.nargs,
            fixture.rtol,
            fixture.atol,
        )
        .unwrap_or_else(|error| panic!("DOPR54 returned err={error}"));

        // Require bit-for-bit agreement in the recorded final state.
        assert_tail_bits(DOPR54_REFERENCE, &result, DOPR54_EXPECTED_TAIL_BITS);
    }

    // Runs the DOPR54 GPU kernel from the same canonical input and checks that
    // it reaches the same recorded final phase-space state.
    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dopr54_gpu_matches_reference() {
        // Load the input-only reference and make its expectation explicit.
        let fixture = GalpyFixture::load_and_validate(DOPR54_REFERENCE);
        assert!(fixture.expected_state_bits.is_empty());

        // Translate the fixture into the direct GPU launch inputs.
        let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
        let output_grid = OutputGrid {
            start: fixture.times[0],
            end: *fixture.times.last().expect("fixture has no output times"),
            steps: fixture.nt as Index,
        };
        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = fixture.initial_state.len() / fixture.dim;
        input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
        let model_component = ModelComponent(core::array::from_fn(|_| None));

        // Launch the concrete DOPR54 kernel, without the general dispatch layer.
        let output_state = launch_galpy_dopr54(
            &model_component,
            &input_state,
            tolerance,
            output_grid,
            Some(fixture.times.clone()),
        )
        .expect("DOPR54 kernel launch failed");
        let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

        // Require bit-for-bit agreement in the recorded final state.
        assert_tail_bits(DOPR54_REFERENCE, &result, DOPR54_EXPECTED_TAIL_BITS);
    }

    // Runs the exact CPU DOPR54 port over all 100 seeded native-galpy dumps and
    // requires every requested output state to match galpy bit for bit.
    #[test]
    #[ignore = "local generated native galpy fixture corpus"]
    fn dopr54_cpu_matches_native_galpy_fixtures() {
        let fixture_paths = corpus_fixture_paths(DOPR54_FIXTURE_DIR, "DOPR54");

        for path in fixture_paths {
            // Read and validate one complete galpy trajectory.
            let case_name = fixture_case_name(&path);
            let fixture = GalpyFixture::load_and_validate(&path);
            assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

            // Run the direct CPU port for the saved initial conditions and times.
            let result = dopr54::integrate_kepler(
                fixture.initial_state_array(),
                &fixture.times,
                fixture.dt_one,
                fixture.nargs,
                fixture.rtol,
                fixture.atol,
            )
            .unwrap_or_else(|error| panic!("DOPR54 returned err={error}"));

            // Check the complete time-major trajectory, including exact bits.
            assert_all_state_bits(&case_name, &result, &fixture);
        }
    }

    // First observed mismatch in the shorter fixture corpus was case_00, step 35,
    // component 5 by 1 ULP. This exact test documents that native libm and CUDA
    // device math are not generally bit-identical for the non-planar cases.
    #[test]
    #[ignore = "native galpy C/libm is not bit-identical to CUDA device math for non-planar fixtures"]
    fn dopr54_gpu_matches_native_galpy_fixtures() {
        let fixture_paths = corpus_fixture_paths(DOPR54_FIXTURE_DIR, "DOPR54");

        for path in fixture_paths {
            // Read and validate one complete galpy trajectory.
            let case_name = fixture_case_name(&path);
            let fixture = GalpyFixture::load_and_validate(&path);
            assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

            // Build and launch the concrete DOPR54 GPU kernel inputs.
            let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
            let output_grid = OutputGrid {
                start: fixture.times[0],
                end: *fixture.times.last().expect("fixture has no output times"),
                steps: fixture.nt as Index,
            };
            let mut input_state = InputState::new_zeroed();
            input_state.num_particles = fixture.initial_state.len() / fixture.dim;
            input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
            let model_component = ModelComponent(core::array::from_fn(|_| None));
            let output_state = launch_galpy_dopr54(
                &model_component,
                &input_state,
                tolerance,
                output_grid,
                Some(fixture.times.clone()),
            )
            .expect("DOPR54 kernel launch failed");
            let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

            // Deliberately require exact bits so this diagnostic stops at the
            // first host/device transcendental difference.
            assert_all_state_bits(&case_name, &result, &fixture);
        }
    }

    // Measures the known native-libm versus CUDA-device-math drift over all 100
    // DOPR54 cases and optionally writes raw GPU values for backend comparison.
    #[test]
    #[ignore = "diagnostic report for host libm versus CUDA device math drift"]
    fn dopr54_gpu_native_galpy_fixture_error_summary() {
        let fixture_paths = corpus_fixture_paths(DOPR54_FIXTURE_DIR, "DOPR54");
        let mut summary = ErrorSummary::default();
        let mut expected_comparisons = 0;
        let mut dump = std::env::var_os(DOPR54_GPU_DUMP_ENV).map(|path| {
            BufWriter::new(
                File::create(&path)
                    .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display())),
            )
        });

        for path in fixture_paths {
            // Read and validate one complete galpy trajectory.
            let case_name = fixture_case_name(&path);
            let fixture = GalpyFixture::load_and_validate(&path);
            assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

            // Run the concrete DOPR54 GPU kernel for this fixture.
            let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
            let output_grid = OutputGrid {
                start: fixture.times[0],
                end: *fixture.times.last().expect("fixture has no output times"),
                steps: fixture.nt as Index,
            };
            let mut input_state = InputState::new_zeroed();
            input_state.num_particles = fixture.initial_state.len() / fixture.dim;
            input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
            let model_component = ModelComponent(core::array::from_fn(|_| None));
            let output_state = launch_galpy_dopr54(
                &model_component,
                &input_state,
                tolerance,
                output_grid,
                Some(fixture.times.clone()),
            )
            .expect("DOPR54 kernel launch failed");
            let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

            // Preserve raw results when requested and include every value in the
            // aggregate host/device error report.
            if let Some(writer) = dump.as_mut() {
                for value in &result {
                    writer
                        .write_all(&value.to_bits().to_le_bytes())
                        .expect("failed to write GPU output dump");
                }
            }
            expected_comparisons += fixture.expected_state_bits.len();
            summary.observe(&case_name, &result, &fixture);
        }

        if let Some(mut writer) = dump {
            writer.flush().expect("failed to flush GPU output dump");
        }

        println!("gpu/native galpy summary: {summary}");
        assert_eq!(summary.compared(), expected_comparisons);
        assert!(summary.max_abs().is_finite());
    }

    // Runs the exact CPU DOP853 port from the canonical native-galpy dump and
    // requires the complete requested trajectory to match bit for bit.
    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dop853_cpu_matches_native_galpy_dump() {
        // Load and validate the complete canonical galpy trajectory.
        let fixture = GalpyFixture::load_and_validate(DOP853_REFERENCE);
        assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

        // Run the direct CPU port, without the general dispatch layer.
        let result = dop853::integrate_kepler(
            fixture.initial_state_array(),
            &fixture.times,
            fixture.dt_one,
            fixture.nargs,
            fixture.rtol,
            fixture.atol,
        );

        // Require exact agreement at every requested output time.
        assert_all_state_bits(DOP853_REFERENCE, &result, &fixture);
    }

    // Runs the DOP853 GPU kernel against the canonical native-galpy dump. Two
    // libdevice pow(x, 1/8) results differ; replacing those bits makes all 6,006
    // GPU state bits match galpy exactly.
    #[test]
    #[ignore = "CUDA device math first differs from native galpy by 1 ULP at step 3, component 3"]
    fn dop853_gpu_matches_native_galpy_dump() {
        // Load and validate the complete canonical galpy trajectory.
        let fixture = GalpyFixture::load_and_validate(DOP853_REFERENCE);
        assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

        // Build and launch the concrete DOP853 GPU kernel inputs.
        let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
        let output_grid = OutputGrid {
            start: fixture.times[0],
            end: *fixture.times.last().expect("fixture has no output times"),
            steps: fixture.nt as Index,
        };
        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = fixture.initial_state.len() / fixture.dim;
        input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
        let model_component = ModelComponent(core::array::from_fn(|_| None));
        let output_state = launch_galpy_dop853(
            &model_component,
            &input_state,
            tolerance,
            output_grid,
            Some(fixture.times.clone()),
        )
        .expect("DOP853 kernel launch failed");
        let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

        // Deliberately require exact bits to identify the first math divergence.
        assert_all_state_bits(DOP853_REFERENCE, &result, &fixture);
    }

    // Confirms that the known DOP853 host/device differences remain small even
    // though the exact-bit diagnostic above intentionally fails at the first one.
    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dop853_gpu_tracks_native_galpy_dump() {
        // Load and validate the complete canonical galpy trajectory.
        let fixture = GalpyFixture::load_and_validate(DOP853_REFERENCE);
        assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

        // Build and launch the concrete DOP853 GPU kernel inputs.
        let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
        let output_grid = OutputGrid {
            start: fixture.times[0],
            end: *fixture.times.last().expect("fixture has no output times"),
            steps: fixture.nt as Index,
        };
        let mut input_state = InputState::new_zeroed();
        input_state.num_particles = fixture.initial_state.len() / fixture.dim;
        input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
        let model_component = ModelComponent(core::array::from_fn(|_| None));
        let output_state = launch_galpy_dop853(
            &model_component,
            &input_state,
            tolerance,
            output_grid,
            Some(fixture.times.clone()),
        )
        .expect("DOP853 kernel launch failed");
        let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

        // Measure all differences and enforce the expected device-math bound.
        let summary = ErrorSummary::for_fixture(DOP853_REFERENCE, &result, &fixture);
        assert!(
            summary.mismatched() > 0,
            "expected host/device math to differ"
        );
        assert!(
            summary.max_abs() < 1.0e-12,
            "GPU/native galpy max absolute error was {}",
            summary.max_abs(),
        );
    }

    // Runs the exact CPU DOP853 port over all 100 seeded native-galpy dumps and
    // requires every requested output state to match galpy bit for bit.
    #[test]
    #[ignore = "local generated native galpy fixture corpus"]
    fn dop853_cpu_matches_native_galpy_fixtures() {
        let fixture_paths = corpus_fixture_paths(DOP853_FIXTURE_DIR, "DOP853");

        for path in fixture_paths {
            // Read and validate one complete galpy trajectory.
            let case_name = fixture_case_name(&path);
            let fixture = GalpyFixture::load_and_validate(&path);
            assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

            // Run the direct CPU port for the saved initial conditions and times.
            let result = dop853::integrate_kepler(
                fixture.initial_state_array(),
                &fixture.times,
                fixture.dt_one,
                fixture.nargs,
                fixture.rtol,
                fixture.atol,
            );

            // Check the complete time-major trajectory, including exact bits.
            assert_all_state_bits(&case_name, &result, &fixture);
        }
    }

    // Runs the DOP853 GPU kernel over all 100 native-galpy dumps and deliberately
    // asks for exact bits to expose any host/device transcendental differences.
    #[test]
    #[ignore = "native galpy C/libm may not be bit-identical to CUDA device math"]
    fn dop853_gpu_matches_native_galpy_fixtures() {
        let fixture_paths = corpus_fixture_paths(DOP853_FIXTURE_DIR, "DOP853");

        for path in fixture_paths {
            // Read and validate one complete galpy trajectory.
            let case_name = fixture_case_name(&path);
            let fixture = GalpyFixture::load_and_validate(&path);
            assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

            // Build and launch the concrete DOP853 GPU kernel inputs.
            let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
            let output_grid = OutputGrid {
                start: fixture.times[0],
                end: *fixture.times.last().expect("fixture has no output times"),
                steps: fixture.nt as Index,
            };
            let mut input_state = InputState::new_zeroed();
            input_state.num_particles = fixture.initial_state.len() / fixture.dim;
            input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
            let model_component = ModelComponent(core::array::from_fn(|_| None));
            let output_state = launch_galpy_dop853(
                &model_component,
                &input_state,
                tolerance,
                output_grid,
                Some(fixture.times.clone()),
            )
            .expect("DOP853 kernel launch failed");
            let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

            // Check the complete time-major trajectory, including exact bits.
            assert_all_state_bits(&case_name, &result, &fixture);
        }
    }

    // Measures native-libm versus CUDA-device-math drift over all 100 DOP853
    // cases and optionally writes raw GPU values for backend comparison.
    #[test]
    #[ignore = "diagnostic report for host libm versus CUDA device math drift"]
    fn dop853_gpu_native_galpy_fixture_error_summary() {
        let fixture_paths = corpus_fixture_paths(DOP853_FIXTURE_DIR, "DOP853");
        let mut summary = ErrorSummary::default();
        let mut expected_comparisons = 0;
        let mut dump = std::env::var_os(DOP853_GPU_DUMP_ENV).map(|path| {
            BufWriter::new(
                File::create(&path)
                    .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display())),
            )
        });

        for path in fixture_paths {
            // Read and validate one complete galpy trajectory.
            let case_name = fixture_case_name(&path);
            let fixture = GalpyFixture::load_and_validate(&path);
            assert_eq!(fixture.expected_state_bits.len(), fixture.nt * fixture.dim);

            // Run the concrete DOP853 GPU kernel for this fixture.
            let tolerance = LogTolerance::from_logarithmic(fixture.rtol, fixture.atol);
            let output_grid = OutputGrid {
                start: fixture.times[0],
                end: *fixture.times.last().expect("fixture has no output times"),
                steps: fixture.nt as Index,
            };
            let mut input_state = InputState::new_zeroed();
            input_state.num_particles = fixture.initial_state.len() / fixture.dim;
            input_state.data[..fixture.initial_state.len()].copy_from_slice(&fixture.initial_state);
            let model_component = ModelComponent(core::array::from_fn(|_| None));
            let output_state = launch_galpy_dop853(
                &model_component,
                &input_state,
                tolerance,
                output_grid,
                Some(fixture.times.clone()),
            )
            .expect("DOP853 kernel launch failed");
            let result = output_state.data[..fixture.nt * fixture.dim].to_vec();

            // Preserve raw results when requested and include every value in the
            // aggregate host/device error report.
            if let Some(writer) = dump.as_mut() {
                for value in &result {
                    writer
                        .write_all(&value.to_bits().to_le_bytes())
                        .expect("failed to write GPU output dump");
                }
            }
            expected_comparisons += fixture.expected_state_bits.len();
            summary.observe(&case_name, &result, &fixture);
        }

        if let Some(mut writer) = dump {
            writer.flush().expect("failed to flush GPU output dump");
        }

        println!("gpu/native galpy summary: {summary}");
        assert_eq!(summary.compared(), expected_comparisons);
        assert!(summary.max_abs().is_finite());
    }

    fn fixture_case_name(path: &Path) -> String {
        path.file_stem()
            .expect("fixture path has no file name")
            .to_string_lossy()
            .into_owned()
    }
}
