from enum import StrEnum
from typing import Any, cast

import drift as dft
import pytest


@pytest.mark.parametrize(
    ("selector", "value", "expected"),
    (
        (dft.Engine, "CPU", dft.Engine.CPU),
        (dft.Method, "DOP853", dft.Method.DOP853),
        (dft.Variant, "Compatible", dft.Variant.Compatible),
    ),
)
def test_selectors_support_value_construction(
    selector: type[StrEnum], value: str, expected: StrEnum
) -> None:
    assert selector(value) is expected


@pytest.mark.parametrize(
    "keyword",
    ({"engine": "CPU"}, {"method": "DOPR54"}, {"variant": "Compatible"}),
)
def test_config_rejects_raw_strings(keyword: dict[str, str]) -> None:
    config = cast(Any, dft.Config)
    with pytest.raises(TypeError, match="expected a drift.* member"):
        config(**keyword)
