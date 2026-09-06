import drift as dft
import numpy as np
import pytest

INITIAL_STATE = np.zeros((1, 6), dtype=np.float64)


@pytest.mark.parametrize(
    "potential",
    (dft.Potential.kepler(1.0), dft.Potential.plummer(1.0, 0.5)),
)
def test_supported_potentials_can_be_attached_to_particles(
    potential: dft.Potential,
) -> None:
    container = dft.part_group(potential, INITIAL_STATE)

    assert isinstance(container, dft.Container)


def test_unsupported_potential_cannot_be_attached_to_particles() -> None:
    with pytest.raises(NotImplementedError, match="only Kepler and Plummer"):
        dft.part_group(dft.Potential.bovy(), INITIAL_STATE)
