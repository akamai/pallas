# Copyright 2020 Akamai Technologies, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime as dt
from decimal import Decimal

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
        (Decimal("0.1"), "DECIMAL '0.1'"),
        (dt.date(2001, 8, 22), "DATE '2001-08-22'"),
        (
            dt.datetime(2001, 8, 22, 3, 4, 5, 321000),
            "TIMESTAMP '2001-08-22 03:04:05.321'",
        ),
        (b"\x00\xff\x00", "from_base64('AP8A')"),
    ],
)
def test_quote_value(value, quoted):
    assert Athena.quote(value) == quoted


@pytest.mark.parametrize("value", [object(), (), [], {}])
def test_quote_invalid(value):
    with pytest.raises(TypeError):
        Athena.quote(value)
