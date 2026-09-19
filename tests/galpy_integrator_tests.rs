#[cfg(test)]
mod support;

#[cfg(test)]
mod tests {
    use drift_rs::dispatch::gpu::{launch_galpy_dop853, launch_galpy_dopr54};
    use drift_rs::integrators::galpy::{dop853, dopr54};
    use shared::Real;

    use crate::support::galpy_fixtures::{
        ErrorSummary, GalpyFixture, GalpyFixtureSuite, assert_all_state_bits, assert_tail_bits,
    };

    const DOPR54: GalpyFixtureSuite = GalpyFixtureSuite::new(
        "DOPR54",
        "tests/fixtures/dopr54_galpy_native",
        "DRIFT_DOPR54_GPU_DUMP",
        integrate_dopr54_cpu,
        launch_galpy_dopr54,
    );
    const DOP853: GalpyFixtureSuite = GalpyFixtureSuite::new(
        "DOP853",
        "tests/fixtures/dop853_galpy_native",
        "DRIFT_DOP853_GPU_DUMP",
        integrate_dop853_cpu,
        launch_galpy_dop853,
    );

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

    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dopr54_cpu_matches_reference() {
        let fixture = DOPR54.reference();
        let result = DOPR54.integrate_cpu(&fixture);

        assert_tail_bits("DOPR54 reference", &result, DOPR54_EXPECTED_TAIL_BITS);
    }

    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dopr54_gpu_matches_reference() {
        let fixture = DOPR54.reference();
        let result = DOPR54.integrate_gpu(&fixture);

        assert_tail_bits("DOPR54 reference", &result, DOPR54_EXPECTED_TAIL_BITS);
    }

    #[test]
    #[ignore = "local generated native galpy fixture corpus"]
    fn dopr54_cpu_matches_native_galpy_fixtures() {
        DOPR54.assert_cpu_corpus();
    }

    // First observed mismatch in the shorter fixture corpus was case_00, step 35,
    // component 5 by 1 ULP.
    #[test]
    #[ignore = "native galpy C/libm is not bit-identical to CUDA device math for non-planar fixtures"]
    fn dopr54_gpu_matches_native_galpy_fixtures() {
        DOPR54.assert_gpu_corpus();
    }

    #[test]
    #[ignore = "diagnostic report for host libm versus CUDA device math drift"]
    fn dopr54_gpu_native_galpy_fixture_error_summary() {
        DOPR54.report_gpu_corpus_errors();
    }

    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dop853_cpu_matches_native_galpy_dump() {
        let fixture = DOP853.reference();
        let result = DOP853.integrate_cpu(&fixture);

        assert_all_state_bits("DOP853 reference", &result, &fixture);
    }

    // Two differing libdevice pow(x, 1/8) results occur in this run. Substituting
    // those bits makes all 6,006 GPU state bits match galpy exactly.
    #[test]
    #[ignore = "CUDA device math first differs from native galpy by 1 ULP at step 3, component 3"]
    fn dop853_gpu_matches_native_galpy_dump() {
        let fixture = DOP853.reference();
        let result = DOP853.integrate_gpu(&fixture);

        assert_all_state_bits("DOP853 reference", &result, &fixture);
    }

    #[test]
    #[ignore = "requires ./scripts/generate_galpy_fixtures.py reference"]
    fn dop853_gpu_tracks_native_galpy_dump() {
        let fixture = DOP853.reference();
        let result = DOP853.integrate_gpu(&fixture);
        let summary = ErrorSummary::for_fixture("DOP853 reference", &result, &fixture);

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

    #[test]
    #[ignore = "local generated native galpy fixture corpus"]
    fn dop853_cpu_matches_native_galpy_fixtures() {
        DOP853.assert_cpu_corpus();
    }

    #[test]
    #[ignore = "native galpy C/libm may not be bit-identical to CUDA device math"]
    fn dop853_gpu_matches_native_galpy_fixtures() {
        DOP853.assert_gpu_corpus();
    }

    #[test]
    #[ignore = "diagnostic report for host libm versus CUDA device math drift"]
    fn dop853_gpu_native_galpy_fixture_error_summary() {
        DOP853.report_gpu_corpus_errors();
    }

    fn integrate_dopr54_cpu(fixture: &GalpyFixture) -> Vec<Real> {
        dopr54::integrate_kepler(
            fixture.initial_state_array(),
            &fixture.times,
            fixture.dt_one,
            fixture.nargs,
            fixture.rtol,
            fixture.atol,
        )
        .unwrap_or_else(|error| panic!("DOPR54 returned err={error}"))
    }

    fn integrate_dop853_cpu(fixture: &GalpyFixture) -> Vec<Real> {
        dop853::integrate_kepler(
            fixture.initial_state_array(),
            &fixture.times,
            fixture.dt_one,
            fixture.nargs,
            fixture.rtol,
            fixture.atol,
        )
    }
}
