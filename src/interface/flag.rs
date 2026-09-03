use pyo3::prelude::*;
#[pyclass(from_py_object)]
#[derive(Default, Clone)]
pub struct Modern {
    pub inner: shared::ModernFlags,
}

#[pymethods]
impl Modern {
    #[new]
    fn new() -> Self {
        Self {
            inner: shared::ModernFlags::NONE,
        }
    }

    fn add(&mut self, value: shared::Index) {
        if let Some(flag) = shared::ModernFlags::from_bits(value) {
            self.inner.set(flag);
        }
    }

    fn has(&self, value: shared::Index) -> bool {
        shared::ModernFlags::from_bits(value).is_some_and(|f| self.inner.contains(f))
    }

    fn bits(&self) -> shared::Index {
        self.inner.bits()
    }

    fn __repr__(&self) -> String {
        format!("Modern(bits={:#06b})", self.inner.bits())
    }
}
