from collections.abc import Callable

import drift as dft
import numpy as np
import pytest

INITIAL_STATE = np.zeros((1, 6), dtype=np.float64)
MAX_MODEL_COMPONENTS = 11
MAX_PARTICLES = 1000


@pytest.mark.parametrize(
    ("factory", "args"),
    (
        (dft.Potential.kepler, ()),
        (dft.Potential.plummer, ()),
        (dft.Potential.plummer, (1.0,)),
    ),
)
def test_potential_parameters_are_required(
    factory: Callable[..., dft.Potential], args: tuple[float, ...]
) -> None:
    with pytest.raises(TypeError):
        factory(*args)


@pytest.mark.parametrize(
    "potential",
    (dft.Potential.kepler(1.0), dft.Potential.plummer(1.0, 0.5)),
)
def test_supported_potentials_can_be_attached_to_particles(
    potential: dft.Potential,
) -> None:
    container = dft.part_group(potential, INITIAL_STATE)

    assert isinstance(container, dft.Container)


@pytest.mark.parametrize(
    "state",
    (
        np.zeros((2, 6), dtype=np.float64),
        np.zeros(12, dtype=np.float64),
    ),
)
def test_initial_state_accepts_particle_records(state: np.ndarray) -> None:
    assert dft.test_group(state).num_particles == 2


@pytest.mark.parametrize(
    "state",
    (
        np.zeros(7, dtype=np.float64),
        np.zeros((2, 5), dtype=np.float64),
        np.zeros((1, 2, 3), dtype=np.float64),
    ),
)
def test_initial_state_rejects_partial_or_misdimensioned_records(
    state: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        dft.test_group(state)


@pytest.mark.parametrize(
    "state",
    (
        np.empty((0, 6), dtype=np.float64),
        np.empty(0, dtype=np.float64),
    ),
)
def test_initial_state_rejects_empty_particle_groups(state: np.ndarray) -> None:
    with pytest.raises(ValueError, match="at least one particle"):
        dft.test_group(state)


def test_initial_state_enforces_particle_capacity() -> None:
    assert (
        dft.test_group(
            np.zeros((MAX_PARTICLES, 6), dtype=np.float64)
        ).num_particles
        == MAX_PARTICLES
    )

    with pytest.raises(ValueError, match="at most 1000"):
        dft.test_group(np.zeros((MAX_PARTICLES + 1, 6), dtype=np.float64))


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_initial_state_rejects_non_finite_values(value: float) -> None:
    state = INITIAL_STATE.copy()
    state[0, 0] = value

    with pytest.raises(ValueError, match="must be finite"):
        dft.test_group(state)


def test_container_particle_counts_are_read_only() -> None:
    particles = dft.test_group(np.zeros((2, 6), dtype=np.float64))
    background = dft.bg_feature(dft.Potential.kepler(1.0))

    assert particles.num_particles == 2
    assert background.num_particles is None

    for container in (particles, background):
        with pytest.raises(AttributeError):
            setattr(container, "num_particles", 3)


def test_unsupported_potential_cannot_be_attached_to_particles() -> None:
    with pytest.raises(NotImplementedError, match="only Kepler and Plummer"):
        dft.part_group(dft.Potential.bovy(), INITIAL_STATE)


def test_container_creation_is_not_limited_by_process_lifetime() -> None:
    potential = dft.Potential.kepler(1.0)

    for _ in range(2 * MAX_MODEL_COMPONENTS):
        dft.bg_feature(potential)


def test_run_rejects_more_containers_than_one_model_can_hold() -> None:
    potential = dft.Potential.kepler(1.0)
    background = dft.bg_feature(potential)
    state = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    particles = [dft.test_group(state) for _ in range(MAX_MODEL_COMPONENTS)]
    config = dft.Config()

    for group in particles[:-1]:
        config.add(group, background)

    with pytest.raises(ValueError, match="at most 11"):
        config.add(particles[-1], background)

    assert len(config.run()) == MAX_MODEL_COMPONENTS


def test_separate_configs_can_use_distinct_high_identity_containers() -> None:
    potential = dft.Potential.kepler(1.0)

    for _ in range(2):
        background = dft.bg_feature(potential)
        particles = dft.test_group(INITIAL_STATE)
        config = dft.Config()
        config.add(particles, background)
