use crate::Index;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModernFlags(Index);

const MAX_DISTINCT_FLAGS: Index = 4;

impl ModernFlags {
    pub const NONE: Self = Self(0);
    pub const DEBUG: Self = Self(1);
    pub const INTERPOLATION: Self = Self(1 << 1);
    pub const POTENTIAL_EVAL: Self = Self(1 << 2);
    pub const PRINT_INFO: Self = Self(1 << 3);
    pub const PRINT_WARN: Self = Self(1 << 4);
    pub const PRINT_ERROR: Self = Self(1 << 5);
    pub const ACCELERATION: Self = Self(1 << 6);
    pub const PARTIAL_STEPS: Self = Self(1 << 7);
    pub const REVERSE: Self = Self(1 << 8);
    pub const HEX_RESULT: Self = Self(1 << 9);
    pub const BCC: Self = Self(1 << 10);

    // ======Composite Flags======

    pub const INFO: Self = Self(Self::PRINT_INFO.0);
    pub const WARN: Self = Self(Self::PRINT_INFO.0 | Self::PRINT_WARN.0);
    pub const ERROR: Self = Self(Self::PRINT_INFO.0 | Self::PRINT_WARN.0 | Self::PRINT_ERROR.0);
    pub const RECOMMENDED: Self =
        Self(Self::ERROR.0 | Self::INTERPOLATION.0 | Self::POTENTIAL_EVAL.0 | Self::ACCELERATION.0);
    pub const DEBUG_SUITE: Self = Self(Self::DEBUG.0 | Self::HEX_RESULT.0 | Self::ERROR.0);
    pub const ALL: Self = Self((1 << MAX_DISTINCT_FLAGS) - 1);

    #[must_use]
    pub fn from_bits(bits: Index) -> Option<Self> {
        let all_bits = Self::ALL.0;
        if bits & !all_bits == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    #[must_use]
    pub fn contains(&self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }

    pub fn set(&mut self, flag: Self) {
        self.0 |= flag.0;
    }

    pub fn clear(&mut self, flag: Self) {
        self.0 &= !flag.0;
    }

    #[must_use]
    pub fn bits(&self) -> Index {
        self.0
    }
}
impl Default for ModernFlags {
    fn default() -> Self {
        ModernFlags::RECOMMENDED
    }
}
