"""Backend failures reach Python as a typed, catchable exception hierarchy."""

import numpy as np
import pytest

import drift as dft

# A radial plunge toward the Kepler singularity under a tight tolerance forces
# DOPR54 below its smallest step and returns a nonzero error code.
_PLUNGE = np.array([[1.0, 0.0, 0.0, -3.0, 0.0, 0.0]], dtype=np.float64)


def _plunge_config() -> dft.Config:
    sim = dft.Config(
        engine=dft.Engine.CPU,
        method=dft.Method.DOPR54,
        ts=(0.0, 5.0, 64),
        tolerance=(1.0e-13, 1.0e-13),
    )
    sim.add(
        dft.test_particles(_PLUNGE), dft.background(dft.Potential.kepler(1.0))
    )
    return sim


def test_exception_hierarchy() -> None:
    assert issubclass(dft.IntegrationError, dft.DriftError)
    assert issubclass(dft.DriftError, RuntimeError)
    assert dft.DriftError.__module__ == "drift"
    assert dft.IntegrationError.__module__ == "drift"


def test_failed_integration_raises_drift_integration_error() -> None:
    with pytest.raises(dft.IntegrationError, match="error code") as raised:
        _plunge_config().run()

    assert isinstance(raised.value, dft.DriftError)
    assert isinstance(raised.value, RuntimeError)
    assert isinstance(raised.value, Exception)


def test_unsupported_configuration_uses_not_implemented_error() -> None:
    sim = dft.Config(
        engine=dft.Engine.CPU,
        method=dft.Method.DOPR54,
        variant=dft.Variant.Modern,
    )
    sim.add(
        dft.test_particles(_PLUNGE), dft.background(dft.Potential.kepler(1.0))
    )

    with pytest.raises(NotImplementedError) as raised:
        sim.run()
    assert not isinstance(raised.value, dft.DriftError)


def test_input_validation_still_raises_plain_value_error() -> None:
    with pytest.raises(ValueError) as raised:
        dft.Config(tolerance=-1.0)
    assert not isinstance(raised.value, dft.DriftError)
