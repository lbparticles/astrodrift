//! Tests for the method registry and the mirror dispatch: catalog
//! consistency, name parsing across all four upstream spellings, and the
//! stub dispatch contract (`NotImplemented` for stubs, unchanged behaviour
//! for the implemented methods).
#[cfg(test)]
mod tests {
    use drift_rs::methods::registry::{self, Family, Status};
    use drift_rs::methods::run_integration;
    use shared::{Config, Engine, Linspace, Method, Model, Tolerance, Variant};

    /// An integration model with no components: stub dispatch never touches
    /// it, so an empty one is enough to exercise the dispatch contract.
    fn empty_model() -> Model {
        Model(std::array::from_fn(|_| None))
    }

    fn compatible_config(engine: Engine, method: Method) -> Config {
        Config::new(
            engine,
            method,
            Variant::Compatible,
            shared::ModernFlags::NONE,
            Linspace::default(),
            Tolerance::default(),
        )
    }

    #[test]
    fn every_method_has_a_catalog_row() {
        for method in Method::ALL {
            let row = registry::spec(method);
            assert_eq!(row.method, method);
            assert!(!row.order.is_empty());
            assert!(!row.mirrors.is_empty());
        }
    }

    #[test]
    fn implemented_methods_are_exactly_the_dispatched_ones() {
        for row in registry::CATALOG {
            let dispatched = matches!(row.method, Method::DOPR54 | Method::DOP853);
            assert_eq!(
                row.status == Status::Implemented,
                dispatched,
                "{} row/dispatch mismatch",
                row.method.canonical_name()
            );
        }
    }

    #[test]
    fn families_map_to_mirror_folders() {
        assert_eq!(registry::spec(Method::DOP853).family, Family::ExplicitRk);
        assert_eq!(registry::spec(Method::LEAPFROG).family, Family::Symplectic);
        assert_eq!(registry::spec(Method::IAS15).family, Family::Implicit);
        assert_eq!(registry::spec(Method::MERCURIUS).family, Family::Hybrid);
    }

    #[test]
    fn stubs_report_not_implemented_on_both_engines() {
        for method in Method::ALL {
            if registry::spec(method).status == Status::Implemented {
                continue;
            }
            for engine in [Engine::GPU, Engine::CPU] {
                let config = compatible_config(engine, method);
                let err = run_integration(&config, &empty_model(), &empty_frame())
                    .expect_err("stub must fail");
                assert!(
                    err.to_string().contains(method.canonical_name()),
                    "error should name the method: {err}"
                );
            }
        }
    }

    #[test]
    fn implemented_methods_still_dispatch_on_compatible() {
        // CPU dispatch is a documented placeholder that always succeeds;
        // GPU would require a device, so only the CPU path is asserted here.
        for method in [Method::DOPR54, Method::DOP853] {
            let config = compatible_config(Engine::CPU, method);
            let result = run_integration(&config, &empty_model(), &empty_frame());
            assert!(result.is_ok(), "{method:?} CPU path must not error");
        }
    }

    #[test]
    fn modern_variant_stays_a_noop_placeholder_for_every_method() {
        for method in Method::ALL {
            let mut config = compatible_config(Engine::GPU, method);
            config.variant = Variant::Modern;
            let result = run_integration(&config, &empty_model(), &empty_frame());
            assert!(result.is_ok(), "{method:?} modern placeholder must not error");
        }
    }

    #[test]
    fn parse_accepts_canonical_and_library_spellings() {
        let cases = [
            ("DOPR54", Method::DOPR54),
            ("dopr54_c", Method::DOPR54),
            ("RK45", Method::DOPR54),
            ("DOP853", Method::DOP853),
            ("DOPRI853Integrator", Method::DOP853),
            ("RK23", Method::RK23),
            ("rk4_c", Method::RK4),
            ("RK5Integrator", Method::RK5),
            ("rk6_c", Method::RK6),
            ("LEAPFROG", Method::LEAPFROG),
            ("LeapfrogIntegrator", Method::LEAPFROG),
            ("symplec4_c", Method::SYMPLEC4),
            ("Ruth4Integrator", Method::SYMPLEC4),
            ("symplec6_c", Method::SYMPLEC6),
            ("WHFast", Method::WHFAST),
            ("WHFast512", Method::WHFAST512),
            ("SEI", Method::SEI),
            ("SABA", Method::SABA),
            ("SABA(10,6,4)", Method::SABA),
            ("EOS", Method::EOS),
            ("IAS15", Method::IAS15),
            ("ias15_c", Method::IAS15),
            ("JANUS", Method::JANUS),
            ("Radau", Method::RADAU),
            ("BDF", Method::BDF),
            ("LSODA", Method::LSODA),
            ("odeint", Method::LSODA),
            ("vode", Method::VODE),
            ("BS", Method::BS),
            ("MERCURIUS", Method::MERCURIUS),
            ("HERMES", Method::MERCURIUS),
            ("TRACE", Method::TRACE),
        ];
        for (name, expected) in cases {
            assert_eq!(
                registry::parse_name(name),
                Some(expected),
                "spelling {name:?}"
            );
        }
        assert_eq!(registry::parse_name("NOT_A_METHOD"), None);
    }

    fn empty_frame() -> drift_rs::state::InputFrame {
        drift_rs::state::InputFrame(std::array::from_fn(|_| None))
    }
}
