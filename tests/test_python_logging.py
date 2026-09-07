"""Rust backend diagnostics use Python's standard logging hierarchy."""

import logging
from collections.abc import Iterator

import drift as dft
import numpy as np
import pytest

LOGGER_NAME = "drift.dispatch.cpu"
N_PARTICLES = 50
INITIAL_STATE = np.tile(
    np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    (N_PARTICLES, 1),
)


def _run_cpu() -> None:
    _cpu_config().run()


def _cpu_config() -> dft.Config:
    kepler = dft.Potential.kepler(1.0)
    sim = dft.Config(engine=dft.Engine.CPU, ts=(0.0, 2.0 * np.pi, 16))
    sim.add(dft.test_particles(INITIAL_STATE), dft.background(kepler))
    return sim


@pytest.fixture(autouse=True)
def restore_drift_logger() -> Iterator[None]:
    logger = logging.getLogger("drift")
    saved = (logger.level, logger.propagate, list(logger.handlers))
    try:
        yield
    finally:
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
        logger.handlers[:] = saved[2]


def test_cpu_run_emits_info_records_on_the_drift_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="drift"):
        _run_cpu()

    records = [
        record for record in caplog.records if record.name == LOGGER_NAME
    ]
    messages = [record.getMessage() for record in records]
    assert any(
        message.startswith("CPU integration starting") for message in messages
    )
    assert any(
        message.startswith("CPU integration finished") for message in messages
    )
    assert all(record.filename == "cpu.rs" for record in records)


def test_cpu_progress_is_ordered_and_rate_limited(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="drift"):
        _run_cpu()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME
    ]
    progress = [
        index
        for index, message in enumerate(messages)
        if message.startswith("stage 1: integrated ")
    ]
    start = next(
        index
        for index, message in enumerate(messages)
        if message.startswith("CPU integration starting")
    )
    finish = next(
        index
        for index, message in enumerate(messages)
        if message.startswith("CPU integration finished")
    )

    assert 0 < len(progress) <= 10
    assert start < progress[0] < progress[-1] < finish


def test_standard_python_levels_can_be_reconfigured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="drift"):
        _run_cpu()
    assert any(
        record.name == LOGGER_NAME and record.levelno == logging.DEBUG
        for record in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="drift"):
        _run_cpu()
    assert [
        record for record in caplog.records if record.name == LOGGER_NAME
    ] == []


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_broken_handler_does_not_abort_integration() -> None:
    class BrokenHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            raise RuntimeError("handler failed")

    logger = logging.getLogger("drift")
    logger.handlers[:] = [BrokenHandler()]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    trajectory, _ = _cpu_config().run()
    assert trajectory is not None
    assert trajectory.shape == (16, N_PARTICLES, 6)
