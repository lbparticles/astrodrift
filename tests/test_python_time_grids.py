import drift as dft
import numpy as np
import numpy.typing as npt
import pytest

type TimeGrid = tuple[float, float, int] | list[float] | npt.NDArray[np.float64]


@pytest.mark.parametrize(
    "ts",
    (
        (0.0, 1.0, 2),
        (1.0, -1.0, 7),
        np.linspace(0.0, 1.0, 7),
        np.linspace(0.0, 1.0e-12, 7),
        np.linspace(1.0e12, 1.0e12 + 1.0e6, 7),
        np.linspace(1.0, -1.0, 7),
    ),
)
def test_valid_time_grids_are_accepted(ts: TimeGrid) -> None:
    dft.Config(ts=ts)


def test_read_only_time_array_is_accepted() -> None:
    times = np.linspace(0.0, 1.0, 7)
    times.flags.writeable = False

    dft.Config(ts=times)


@pytest.mark.parametrize(
    ("ts", "message"),
    (
        ([0.0, 1.0e-12, 3.0e-12], "uniformly spaced"),
        ([0.0, 0.25, 0.500_000_000_001, 0.75, 1.0], "uniformly spaced"),
        (np.logspace(0.0, 1.0, 7), "uniformly spaced"),
        ([0.0, 0.5, 0.5, 1.0], "strictly increasing or strictly decreasing"),
        ([0.0, 0.5, 0.25, 1.0], "strictly increasing or strictly decreasing"),
        ((float("nan"), 1.0, 3), "finite"),
        ([0.0, float("inf")], "finite"),
        ((0.0, 1.0, 1), "between 2 and 1024"),
        ((0.0, 1.0, 1025), "between 2 and 1024"),
        ([0.0], "between 2 and 1024"),
        (np.linspace(0.0, 1.0, 1025), "between 2 and 1024"),
        ((1.0, 1.0, 2), "non-zero"),
        ([1.0, 1.0], "non-zero"),
        (np.linspace(0.0, 1.0, 7)[::2], "contiguous"),
    ),
)
def test_invalid_time_grids_are_rejected(ts: TimeGrid, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dft.Config(ts=ts)
