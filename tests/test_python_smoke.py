from typing import NamedTuple

import drift as dft
import numpy as np
import numpy.typing as npt
import pytest

IntegrationResult = list[npt.NDArray[np.float64] | None]

# At 100 timesteps this exceeds the legacy 11 * MAX_PARTICLES allocation.
N_PARTICLES = 20
SUPPORTED_METHODS = (dft.Method.DOPR54, dft.Method.DOP853)
GMC_INITIAL_STATE = np.tile(
    np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    (N_PARTICLES, 1),
)
ISO_INITIAL_STATE = np.tile(
    np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float64),
    (N_PARTICLES, 1),
)


class SmokeModel(NamedTuple):
    gal: dft.Container
    gmc: dft.Container
    iso: dft.Container


def _make_model() -> SmokeModel:
    kp = dft.Potential.kepler(1.0)
    gal = dft.background(kp)
    gmc = dft.particles(kp, GMC_INITIAL_STATE)
    iso = dft.test_particles(ISO_INITIAL_STATE)
    return SmokeModel(gal, gmc, iso)


def _run_gpu(
    model: SmokeModel,
    method: dft.Method,
    variant: dft.Variant = dft.Variant.Compatible,
) -> IntegrationResult:
    sim = dft.Config(
        engine=dft.Engine.GPU,
        method=method,
        variant=variant,
    )
    sim.add(model.gmc, model.gal)
    sim.add(model.iso, model.gmc, model.gal)
    return sim.run()


def _run_cpu(model: SmokeModel, method: dft.Method) -> IntegrationResult:
    sim = dft.Config(engine=dft.Engine.CPU, method=method)
    sim.add(model.gmc, model.gal)
    sim.add(model.iso, model.gmc, model.gal)
    return sim.run()


def _particle_trajectories(
    result: IntegrationResult,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    gmc_trajectory, _, iso_trajectory = result
    assert isinstance(iso_trajectory, np.ndarray)
    assert isinstance(gmc_trajectory, np.ndarray)
    return iso_trajectory, gmc_trajectory


@pytest.fixture(scope="module")
def model() -> SmokeModel:
    return _make_model()


@pytest.fixture(scope="module")
def compatible_results(
    model: SmokeModel,
) -> dict[dft.Method, IntegrationResult]:
    return {method: _run_gpu(model, method) for method in SUPPORTED_METHODS}


@pytest.fixture(scope="module")
def cpu_compatible_results(
    model: SmokeModel,
) -> dict[dft.Method, IntegrationResult]:
    return {method: _run_cpu(model, method) for method in SUPPORTED_METHODS}


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_results_follow_registration_order(
    compatible_results: dict[dft.Method, IntegrationResult], method: dft.Method
) -> None:
    gmc_trajectory, background_result, iso_trajectory = compatible_results[
        method
    ]

    assert isinstance(iso_trajectory, np.ndarray)
    assert background_result is None
    assert isinstance(gmc_trajectory, np.ndarray)


def test_empty_config_returns_no_results() -> None:
    assert dft.Config().run() == []


def test_particle_group_requires_a_registered_force_source(
    model: SmokeModel,
) -> None:
    sim = dft.Config()
    sim.add(model.iso, model.gmc)

    with pytest.raises(ValueError, match="registered with add"):
        sim.run()


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_particle_results_are_time_major_trajectories(
    compatible_results: dict[dft.Method, IntegrationResult], method: dft.Method
) -> None:
    for trajectory in _particle_trajectories(compatible_results[method]):
        assert trajectory.shape == (100, N_PARTICLES, 6)
        assert trajectory.dtype == np.float64


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_gpu_trajectories_are_accurate_over_one_orbit(
    compatible_results: dict[dft.Method, IntegrationResult], method: dft.Method
) -> None:
    trajectories = _particle_trajectories(compatible_results[method])
    initial_states = (ISO_INITIAL_STATE, GMC_INITIAL_STATE)

    for trajectory, initial_state in zip(
        trajectories, initial_states, strict=True
    ):
        np.testing.assert_array_equal(trajectory[0], initial_state)
        assert not np.array_equal(trajectory[1], initial_state)
        np.testing.assert_allclose(
            trajectory[-1], initial_state, rtol=0.0, atol=2.0e-10
        )


def test_unimplemented_gpu_variant_is_rejected(model: SmokeModel) -> None:
    with pytest.raises(NotImplementedError, match="Engine.GPU"):
        _run_gpu(model, dft.Method.DOPR54, dft.Variant.Modern)


def test_unimplemented_cpu_variant_is_rejected(model: SmokeModel) -> None:
    sim = dft.Config(engine=dft.Engine.CPU, variant=dft.Variant.Modern)
    sim.add(model.iso, model.gal)

    with pytest.raises(NotImplementedError, match="Engine.CPU or Engine.GPU"):
        sim.run()


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_cpu_compatible_results_are_time_major_trajectories(
    cpu_compatible_results: dict[dft.Method, IntegrationResult],
    method: dft.Method,
) -> None:
    trajectories = _particle_trajectories(cpu_compatible_results[method])

    for trajectory, initial_state in zip(
        trajectories, (ISO_INITIAL_STATE, GMC_INITIAL_STATE), strict=True
    ):
        assert trajectory.shape == (100, N_PARTICLES, 6)
        assert trajectory.dtype == np.float64
        np.testing.assert_array_equal(trajectory[0], initial_state)
        assert not np.array_equal(trajectory[1], initial_state)
        np.testing.assert_allclose(
            trajectory[-1], initial_state, rtol=0.0, atol=2.0e-10
        )


def test_default_config_uses_cpu_compatible_path(
    model: SmokeModel,
    cpu_compatible_results: dict[str, IntegrationResult],
) -> None:
    sim = dft.Config()
    sim.add(model.gmc, model.gal)
    sim.add(model.iso, model.gmc, model.gal)

    default = _particle_trajectories(sim.run())
    explicit = _particle_trajectories(cpu_compatible_results[dft.Method.DOPR54])
    for default_trajectory, explicit_trajectory in zip(
        default, explicit, strict=True
    ):
        np.testing.assert_array_equal(default_trajectory, explicit_trajectory)


@pytest.mark.parametrize(
    "tolerance",
    (0.0, -1.0, float("nan"), float("inf"), (1.0e-12, 0.0)),
)
def test_invalid_tolerances_are_rejected(
    tolerance: float | tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="finite and greater than zero"):
        dft.Config(tolerance=tolerance)


def test_add_rejects_empty_and_cyclic_dependencies(model: SmokeModel) -> None:
    sim = dft.Config()

    with pytest.raises(ValueError, match="at least one dependency"):
        sim.add(model.iso)
    with pytest.raises(ValueError, match="cycle"):
        sim.add(model.iso, model.iso)

    sim.add(model.gmc, model.gal)
    sim.add(model.iso, model.gmc)
    with pytest.raises(ValueError, match="cycle"):
        sim.add(model.gal, model.iso)

    # Rejected edges leave the existing graph usable, and duplicates are harmless.
    sim.add(model.gmc, model.gal, model.gal)
    result = sim.run()
    assert isinstance(result[0], np.ndarray)
    assert result[1] is None
    assert isinstance(result[2], np.ndarray)


def test_dependency_alias_warns_at_the_call_site(model: SmokeModel) -> None:
    sim = dft.Config()

    with pytest.warns(DeprecationWarning, match="use Config.add") as warnings:
        sim.dependency(model.gmc, model.gal)

    assert warnings[0].filename == __file__


def run_gpu_smoke() -> IntegrationResult:
    return _run_gpu(_make_model(), dft.Method.DOPR54)


if __name__ == "__main__":
    print(run_gpu_smoke())
