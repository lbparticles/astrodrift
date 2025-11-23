use pyo3::prelude::*;
#[pyclass]
#[derive(Clone)]
pub struct Modern {
    pub flags: shared::ModernFlags,
}

#[pymethods]
impl Modern {
    #[new]
    fn new() -> Self {
        Self {
            flags: shared::ModernFlags::NONE,
        }
    }

    fn add(&mut self, value: shared::Index) {
        if let Some(flag) = shared::ModernFlags::from_bits(value) {
            self.flags.set(flag);
        }
    }

    fn has(&self, value: shared::Index) -> bool {
        shared::ModernFlags::from_bits(value)
            .map(|f| self.flags.contains(f))
            .unwrap_or(false)
    }

    fn bits(&self) -> shared::Index {
        self.flags.bits()
    }

    fn __repr__(&self) -> String {
        format!("Modern(bits={:#06b})", self.flags.bits())
    }
}
