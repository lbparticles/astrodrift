//! Method registry: the single catalog of every integration method drift
//! knows about.
//!
//! Each [`MethodSpec`] row records the algorithm family, the upstream
//! libraries and identifiers the drift implementation mirrors (galpy, scipy,
//! REBOUND, gala), the integration order, the per-step force-evaluation count
//! feeding the throughput model, and the implementation status. Python-side
//! introspection (`method_catalog()`) and the `Method` string parser are both
//! driven from here, so adding a method means adding one enum variant, one
//! row, and one stub module.
//!
//! Mirror spellings were verified against upstream sources:
//! - galpy `Orbits.check_integrator` (galpy/orbit/Orbits.py)
//! - scipy `solve_ivp`/`ode` solver names (scipy/integrate)
//! - REBOUND `REB_BUILTIN_INTEGRATORS` X-macro list (src/rebound.h) and the
//!   SABA/EOS subtype tables (`src/integrator_saba.h`, `src/integrator_eos.h`)
//! - gala `gala.integrate.pyintegrators` exports (DOPRI853, Leapfrog, RK5,
//!   Ruth4) and the matching cyintegrators

use shared::Method;

/// Algorithm family a method belongs to; mirrors live in one folder per
/// family under `src/methods/`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Explicit Runge-Kutta schemes with adaptive step or dense output.
    ExplicitRk,
    /// Symplectic splitting schemes (drift/kick compositions).
    Symplectic,
    /// Implicit or extrapolative non-symplectic schemes.
    Implicit,
    /// Hybrid schemes switching between integrators per particle/step.
    Hybrid,
}

impl Family {
    /// Folder name under `src/methods/` holding this family's mirrors.
    #[must_use]
    pub const fn folder(self) -> &'static str {
        match self {
            Family::ExplicitRk => "rk",
            Family::Symplectic => "symplectic",
            Family::Implicit => "implicit",
            Family::Hybrid => "hybrid",
        }
    }
}

/// Upstream library a mirrored method comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Galpy,
    Scipy,
    Rebound,
    Gala,
}

impl Origin {
    /// Lowercase library identifier used in the Python-facing catalog.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Origin::Galpy => "galpy",
            Origin::Scipy => "scipy",
            Origin::Rebound => "rebound",
            Origin::Gala => "gala",
        }
    }
}

/// Implementation status of a method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Dispatched to a real integration loop (device kernels and/or the
    /// reference transliteration on host).
    Implemented,
    /// Dispatch is wired and the module documents the implementation plan,
    /// but the integration loop itself is not written yet; running it
    /// returns `GPUDispatchError::NotImplemented`.
    Stub,
}

impl Status {
    /// Status label used in the Python-facing catalog.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Status::Implemented => "implemented",
            Status::Stub => "stub",
        }
    }
}

/// One catalog row per method.
#[derive(Debug, Clone, Copy)]
pub struct MethodSpec {
    /// The method this row describes.
    pub method: Method,
    /// Algorithm family (selects the mirror folder).
    pub family: Family,
    /// Integration order, using the usual `p(e)` error notation for
    /// embedded pairs and dense-output notes where applicable.
    pub order: &'static str,
    /// Force evaluations per step feeding the analytic throughput model
    /// (`None` = no model yet; stubs have none until calibrated).
    pub stages: Option<usize>,
    /// Implementation status.
    pub status: Status,
    /// Upstream mirrors as `(library, identifier spelled as there)` pairs.
    pub mirrors: &'static [(&'static Origin, &'static str)],
}

/// The catalog: one row per [`Method::ALL`] entry, same order (enforced by
/// test).
pub const CATALOG: &[MethodSpec; Method::ALL.len()] = &[
    MethodSpec {
        method: Method::DOPR54,
        family: Family::ExplicitRk,
        order: "5(4), dense output",
        stages: Some(6),
        status: Status::Implemented,
        mirrors: &[
            (&Origin::Galpy, "dopr54_c"),
            (&Origin::Scipy, "RK45"),
            (&Origin::Scipy, "dopri5"),
        ],
    },
    MethodSpec {
        method: Method::DOP853,
        family: Family::ExplicitRk,
        order: "8(5,3), dense output",
        stages: Some(12),
        status: Status::Implemented,
        mirrors: &[
            (&Origin::Galpy, "dop853_c"),
            (&Origin::Galpy, "dop853"),
            (&Origin::Scipy, "DOP853"),
            (&Origin::Gala, "DOPRI853Integrator"),
        ],
    },
    MethodSpec {
        method: Method::RK23,
        family: Family::ExplicitRk,
        order: "3(2)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Scipy, "RK23")],
    },
    MethodSpec {
        method: Method::RK4,
        family: Family::ExplicitRk,
        order: "4, fixed step",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Galpy, "rk4_c")],
    },
    MethodSpec {
        method: Method::RK5,
        family: Family::ExplicitRk,
        order: "5, fixed step",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Gala, "RK5Integrator")],
    },
    MethodSpec {
        method: Method::RK6,
        family: Family::ExplicitRk,
        order: "6, fixed step",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Galpy, "rk6_c")],
    },
    MethodSpec {
        method: Method::LEAPFROG,
        family: Family::Symplectic,
        order: "2 (symplectic)",
        stages: None,
        status: Status::Stub,
        mirrors: &[
            (&Origin::Galpy, "leapfrog_c"),
            (&Origin::Galpy, "leapfrog"),
            (&Origin::Rebound, "LEAPFROG"),
            (&Origin::Gala, "LeapfrogIntegrator"),
        ],
    },
    MethodSpec {
        method: Method::SYMPLEC4,
        family: Family::Symplectic,
        order: "4 (symplectic)",
        stages: None,
        status: Status::Stub,
        mirrors: &[
            (&Origin::Galpy, "symplec4_c"),
            (&Origin::Gala, "Ruth4Integrator"),
        ],
    },
    MethodSpec {
        method: Method::SYMPLEC6,
        family: Family::Symplectic,
        order: "6 (symplectic)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Galpy, "symplec6_c")],
    },
    MethodSpec {
        method: Method::WHFAST,
        family: Family::Symplectic,
        order: "2 (symplectic), correctors up to 11th",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "WHFast")],
    },
    MethodSpec {
        method: Method::WHFAST512,
        family: Family::Symplectic,
        order: "2 (symplectic, SIMD lanes)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "WHFast512")],
    },
    MethodSpec {
        method: Method::SEI,
        family: Family::Symplectic,
        order: "2 (symplectic, shearing sheet)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "SEI")],
    },
    MethodSpec {
        method: Method::SABA,
        family: Family::Symplectic,
        order: "family 2-6, default SABA(10,6,4)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "SABA")],
    },
    MethodSpec {
        method: Method::EOS,
        family: Family::Symplectic,
        order: "family 2-8 (embedded splitting)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "EOS")],
    },
    MethodSpec {
        method: Method::IAS15,
        family: Family::Implicit,
        order: "15 (adaptive Gauss-Radau)",
        stages: None,
        status: Status::Stub,
        mirrors: &[
            (&Origin::Rebound, "IAS15"),
            (&Origin::Galpy, "ias15_c"),
        ],
    },
    MethodSpec {
        method: Method::JANUS,
        family: Family::Implicit,
        order: "4 (implicit symplectic, bit-reversible)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "JANUS")],
    },
    MethodSpec {
        method: Method::RADAU,
        family: Family::Implicit,
        order: "5 (implicit Radau IIA)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Scipy, "Radau")],
    },
    MethodSpec {
        method: Method::BDF,
        family: Family::Implicit,
        order: "1-5 (implicit, variable order)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Scipy, "BDF")],
    },
    MethodSpec {
        method: Method::LSODA,
        family: Family::Implicit,
        order: "adaptive Adams/BDF switching",
        stages: None,
        status: Status::Stub,
        mirrors: &[
            (&Origin::Scipy, "LSODA"),
            (&Origin::Galpy, "odeint"),
        ],
    },
    MethodSpec {
        method: Method::VODE,
        family: Family::Implicit,
        order: "adaptive Adams/BDF",
        stages: None,
        status: Status::Stub,
        // scipy's complex-valued twin `zvode` is out of scope: drift states
        // are real f64.
        mirrors: &[(&Origin::Scipy, "vode")],
    },
    MethodSpec {
        method: Method::BS,
        family: Family::Implicit,
        order: "variable (Gragg-Bulirsch-Stoer extrapolation)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "BS")],
    },
    MethodSpec {
        method: Method::MERCURIUS,
        family: Family::Hybrid,
        order: "2 + 15 (WHFast / IAS15 switch)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "MERCURIUS")],
    },
    MethodSpec {
        method: Method::TRACE,
        family: Family::Hybrid,
        order: "hybrid (Lu, Hernandez & Rein 2024)",
        stages: None,
        status: Status::Stub,
        mirrors: &[(&Origin::Rebound, "TRACE")],
    },
];

/// Catalog row for a method. Total over `Method::ALL` (enforced by test:
/// row order matches `Method::ALL`).
#[must_use]
pub const fn spec(method: Method) -> &'static MethodSpec {
    // A direct match (not a scan) keeps this usable from const contexts;
    // the catalog-order alignment is asserted by `catalog_covers_every_method_in_order`.
    match method {
        Method::DOPR54 => &CATALOG[0],
        Method::DOP853 => &CATALOG[1],
        Method::RK23 => &CATALOG[2],
        Method::RK4 => &CATALOG[3],
        Method::RK5 => &CATALOG[4],
        Method::RK6 => &CATALOG[5],
        Method::LEAPFROG => &CATALOG[6],
        Method::SYMPLEC4 => &CATALOG[7],
        Method::SYMPLEC6 => &CATALOG[8],
        Method::WHFAST => &CATALOG[9],
        Method::WHFAST512 => &CATALOG[10],
        Method::SEI => &CATALOG[11],
        Method::SABA => &CATALOG[12],
        Method::EOS => &CATALOG[13],
        Method::IAS15 => &CATALOG[14],
        Method::JANUS => &CATALOG[15],
        Method::RADAU => &CATALOG[16],
        Method::BDF => &CATALOG[17],
        Method::LSODA => &CATALOG[18],
        Method::VODE => &CATALOG[19],
        Method::BS => &CATALOG[20],
        Method::MERCURIUS => &CATALOG[21],
        Method::TRACE => &CATALOG[22],
    }
}

/// Upstream-native alias table for [`parse_name`]. Matched
/// case-insensitively after the galpy `_c` suffix strip.
const ALIASES: &[(&str, Method)] = &[
    // scipy solve_ivp "RK45" and ode "dopri5" are Dormand-Prince 5(4)
    ("RK45", Method::DOPR54),
    ("DOPRI5", Method::DOPR54),
    // gala python class names
    ("DOPRI853INTEGRATOR", Method::DOP853),
    ("LEAPFROGINTEGRATOR", Method::LEAPFROG),
    ("RUTH4INTEGRATOR", Method::SYMPLEC4),
    ("RK5INTEGRATOR", Method::RK5),
    // galpy routes its scipy fallback "odeint" to LSODA
    ("ODEINT", Method::LSODA),
    // REBOUND: HERMES is the pre-rename MERCURIUS, WH the legacy Wisdom-
    // Holman spelling (WHFast family), SABA(10,6,4) the SABA default
    ("HERMES", Method::MERCURIUS),
    ("WH", Method::WHFAST),
    ("SABA(10,6,4)", Method::SABA),
];

/// Parse a method name in any accepted spelling:
/// - canonical drift names (`"DOP853"`, `"LEAPFROG"`, ...), any case
/// - galpy spellings, including the `_c` compiled-suffix (`"dopr54_c"`,
///   `"symplec4_c"`, `"ias15_c"`, `"odeint"`, ...)
/// - scipy solver names (`"RK45"`, `"Radau"`, `"BDF"`, `"LSODA"`, ...)
/// - REBOUND integrator names (`"IAS15"`, `"WHFast"`, `"MERCURIUS"`,
///   `"HERMES"`, ...)
/// - gala integrator class names (`"DOPRI853Integrator"`,
///   `"Ruth4Integrator"`, ...)
///
/// Returns `None` for unknown names; the caller decides how to report it.
#[must_use]
pub fn parse_name(raw: &str) -> Option<Method> {
    // galpy marks its compiled flavours with a trailing `_c`; the algorithm
    // is identical, so fold the suffix away before matching.
    let stripped = raw.strip_suffix("_c").unwrap_or(raw);
    for method in Method::ALL {
        if stripped.eq_ignore_ascii_case(method.canonical_name()) {
            return Some(method);
        }
    }
    for (alias, method) in ALIASES {
        if stripped.eq_ignore_ascii_case(alias) {
            return Some(*method);
        }
    }
    None
}

/// Human-readable list of every accepted spelling, for error messages.
#[must_use]
pub fn accepted_names() -> String {
    let mut joined = String::new();
    for method in Method::ALL {
        if !joined.is_empty() {
            joined.push_str(", ");
        }
        joined.push_str(method.canonical_name());
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_covers_every_method_in_order() {
        assert_eq!(CATALOG.len(), Method::ALL.len());
        for (row, method) in CATALOG.iter().zip(Method::ALL) {
            assert_eq!(row.method, method);
        }
    }

    #[test]
    fn canonical_names_round_trip() {
        for method in Method::ALL {
            assert_eq!(parse_name(method.canonical_name()), Some(method));
        }
    }

    #[test]
    fn canonical_names_are_unique() {
        for (i, a) in Method::ALL.iter().enumerate() {
            for b in &Method::ALL[i + 1..] {
                assert_ne!(a.canonical_name(), b.canonical_name());
            }
        }
    }

    #[test]
    fn library_spellings_parse() {
        for name in [
            "dopr54_c",
            "dop853_c",
            "dop853",
            "rk4_c",
            "rk6_c",
            "leapfrog",
            "leapfrog_c",
            "symplec4_c",
            "symplec6_c",
            "ias15_c",
            "odeint",
        ] {
            assert!(parse_name(name).is_some(), "galpy spelling {name:?}");
        }
        for name in ["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA", "vode"] {
            assert!(parse_name(name).is_some(), "scipy spelling {name:?}");
        }
        for name in [
            "IAS15", "WHFast", "WHFast512", "SEI", "LEAPFROG", "JANUS", "MERCURIUS", "HERMES",
            "SABA", "SABA(10,6,4)", "EOS", "BS", "TRACE",
        ] {
            assert!(parse_name(name).is_some(), "rebound spelling {name:?}");
        }
        for name in [
            "DOPRI853Integrator",
            "LeapfrogIntegrator",
            "RK5Integrator",
            "Ruth4Integrator",
        ] {
            assert!(parse_name(name).is_some(), "gala spelling {name:?}");
        }
    }

    #[test]
    fn unknown_names_rejected() {
        assert!(parse_name("NOPE").is_none());
        assert!(parse_name("").is_none());
        assert!(parse_name("dopri9_c").is_none());
    }

    #[test]
    fn implemented_methods_have_stage_counts() {
        for row in CATALOG {
            if row.status == Status::Implemented {
                assert!(row.stages.is_some(), "{}", row.method.canonical_name());
            }
        }
    }
}
