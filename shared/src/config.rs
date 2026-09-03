use core::f64::consts::PI;

use crate::ModernFlags;
use crate::{MIN_RTOL,MIN_ATOL,Real,Index};

/// Maximum number of GPUs usable by a single dispatch. Fixed bound on
/// purpose (JPL rule 2: every loop has a compile-time upper bound).
pub const MAX_DEVICES: usize = 8;

#[derive(Debug, Clone, Copy, Default)]
pub struct Config {
    pub engine: Engine,
    pub method: Method,
    pub variant: Variant,
    pub flags: ModernFlags,
    pub settings: Settings,
    /// GPU ordinals used by `Engine::GPU` dispatch. `num_devices == 0`
    /// means "unset" and is treated as device 0 (backwards compatible).
    pub devices: [usize; MAX_DEVICES],
    pub num_devices: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Settings {
    pub ts: Linspace,
    pub tolerance: Tolerance,
}

impl Config {
    pub fn new(
        engine: Engine,
        method: Method,
        variant: Variant,
        flags: ModernFlags,
        ts: Linspace,
        tolerance: Tolerance,
    ) -> Self {
        Self {
            engine,
            method,
            variant,
            flags,
            settings: Settings { ts, tolerance },
            devices: [0; MAX_DEVICES],
            num_devices: 0, // unset -> devices_slice() yields device 0
        }
    }
    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }

    /// Selects the GPUs used by `Engine::GPU` dispatch. Count and duplicates
    /// are validated here; ordinal validity is checked by the driver at
    /// context creation (failures propagate as `GPUDispatchError`).
    pub fn set_devices(&mut self, devices: &[usize]) -> Result<(), &'static str> {
        if devices.is_empty() {
            return Err("devices must contain at least one GPU ordinal");
        }
        if devices.len() > MAX_DEVICES {
            return Err("too many devices");
        }
        for (i, device) in devices.iter().enumerate() {
            if devices[..i].contains(device) {
                return Err("duplicate device ordinals");
            }
            self.devices[i] = *device;
        }
        self.num_devices = devices.len();
        Ok(())
    }

    /// The effective device list (device 0 when unset).
    pub fn devices_slice(&self) -> &[usize] {
        if self.num_devices == 0 {
            &[0]
        } else {
            &self.devices[..self.num_devices]
        }
    }
}


#[derive(Clone, Copy, Debug)]
pub struct Linspace{
    pub start: Real, 
    pub end: Real, 
    pub steps: Index,
}
impl Default for Linspace {
    fn default() -> Self {
        Self{
            start: 0.0, 
            end: 2. * PI, 
            steps:100
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Tolerance{
    pub atol: Real, 
    pub rtol: Real
}
impl Default for Tolerance {
    fn default() -> Self {
        Self{
            atol: MIN_ATOL, 
            rtol: MIN_RTOL
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Engine {
    #[default]
    GPU,
    CPU,
}

/// Integration method selector.
///
/// Variant spellings are the canonical drift names; the catalog in
/// `src/methods/registry.rs` maps every method to the upstream libraries it
/// mirrors (galpy, scipy, REBOUND, gala) and its implementation status.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    // -- explicit Runge-Kutta family (src/methods/rk/) --------------------
    #[default]
    DOPR54,
    DOP853,
    RK23,
    RK4,
    RK5,
    RK6,
    // -- symplectic splitting family (src/methods/symplectic/) ------------
    LEAPFROG,
    SYMPLEC4,
    SYMPLEC6,
    WHFAST,
    WHFAST512,
    SEI,
    SABA,
    EOS,
    // -- implicit and adaptive non-symplectic family
    //    (src/methods/implicit/) -----------------------------------------
    IAS15,
    JANUS,
    RADAU,
    BDF,
    LSODA,
    VODE,
    BS,
    // -- hybrid splitting family (src/methods/hybrid/) --------------------
    MERCURIUS,
    TRACE,
}

impl Method {
    /// Every method known to the dispatcher, in catalog order. The method
    /// registry (`src/methods/registry.rs`) carries one spec row per entry;
    /// the two arrays are kept aligned by test.
    pub const ALL: [Method; 23] = [
        Method::DOPR54,
        Method::DOP853,
        Method::RK23,
        Method::RK4,
        Method::RK5,
        Method::RK6,
        Method::LEAPFROG,
        Method::SYMPLEC4,
        Method::SYMPLEC6,
        Method::WHFAST,
        Method::WHFAST512,
        Method::SEI,
        Method::SABA,
        Method::EOS,
        Method::IAS15,
        Method::JANUS,
        Method::RADAU,
        Method::BDF,
        Method::LSODA,
        Method::VODE,
        Method::BS,
        Method::MERCURIUS,
        Method::TRACE,
    ];

    /// Canonical drift spelling of this method (the Python-facing name and
    /// the registry/catalog key).
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Method::DOPR54 => "DOPR54",
            Method::DOP853 => "DOP853",
            Method::RK23 => "RK23",
            Method::RK4 => "RK4",
            Method::RK5 => "RK5",
            Method::RK6 => "RK6",
            Method::LEAPFROG => "LEAPFROG",
            Method::SYMPLEC4 => "SYMPLEC4",
            Method::SYMPLEC6 => "SYMPLEC6",
            Method::WHFAST => "WHFAST",
            Method::WHFAST512 => "WHFAST512",
            Method::SEI => "SEI",
            Method::SABA => "SABA",
            Method::EOS => "EOS",
            Method::IAS15 => "IAS15",
            Method::JANUS => "JANUS",
            Method::RADAU => "RADAU",
            Method::BDF => "BDF",
            Method::LSODA => "LSODA",
            Method::VODE => "VODE",
            Method::BS => "BS",
            Method::MERCURIUS => "MERCURIUS",
            Method::TRACE => "TRACE",
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Compatible,
    #[default]
    Modern,
}

