import numpy as np
import pytest

from pallas.base import Athena


@pytest.mark.parametrize(
    "value,quoted",
    [
        (None, "NULL"),
        ("", "''"),
        ("hello", "'hello'"),
        ("'", "''''"),
        (42, "42"),
        (np.int64(42), "42"),
        (3.14, "3.14"),
        (float("nan"), "nan()"),
        (float("inf"), "infinity()"),
        (float("-inf"), "-infinity()"),
        (np.float64(3.14), "3.14"),
        (np.float64("nan"), "nan()"),
        (np.float64("inf"), "infinity()"),
        (np.float64("-inf"), "-infinity()"),
        (np.nan, "nan()"),
        (np.inf, "infinity()"),
        (-np.inf, "-infinity()"),
    ],
)
def test_quote_value(value, quoted):
    assert Athena.quote(value) == quoted


@pytest.mark.parametrize("value", [object(), (), [], {}])
def test_quote_invalid(value):
    with pytest.raises(TypeError):
        Athena.quote(value)
