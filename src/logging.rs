//! Connect Rust diagnostics to Python's standard logging hierarchy.

use log::{LevelFilter, Log, Metadata, Record};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_log::{Caching, Logger};

const LOG_TARGET: &str = "drift";

/// Prevent a broken Python handler from corrupting the extension call that
/// emitted the record while preserving an exception that predates logging.
struct PythonLogger(Logger);

impl Log for PythonLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.0.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        Python::attach(|py| {
            let existing_error = PyErr::take(py);
            self.0.log(record);
            // `Log::log` cannot return a Python handler failure, so pyo3-log
            // leaves it set in CPython. Report it through `sys.unraisablehook`
            // and clear it; otherwise PyO3 turns an otherwise successful call
            // into a SystemError. Any older exception is restored below.
            if let Some(logging_error) = PyErr::take(py) {
                logging_error.write_unraisable(py, None);
            }
            if let Some(existing_error) = existing_error {
                existing_error.restore(py);
            }
        });
    }

    fn flush(&self) {
        self.0.flush();
    }
}

/// Install the logging bridge.
///
/// Logger objects are cached, but their levels are not, so normal Python
/// reconfiguration takes effect without a drift-specific cache-reset API.
pub(crate) fn init(py: Python<'_>) -> PyResult<()> {
    let logger = Logger::new(py, Caching::Loggers)?
        .filter(LevelFilter::Off)
        .filter_target(LOG_TARGET.to_owned(), LevelFilter::Trace);
    log::set_boxed_logger(Box::new(PythonLogger(logger)))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    log::set_max_level(LevelFilter::Trace);
    Ok(())
}
