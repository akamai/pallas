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

"""
SQL helpers.

Implements quoting compatible with Athena.
"""

import base64
import datetime as dt
import math
import numbers
from decimal import Decimal
from typing import Union

SQL_SCALAR = Union[None, str, float, numbers.Real, Decimal, bytes, dt.date]


def _quote_str(value: str) -> str:
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _quote_integral(value: Union[int, numbers.Integral]) -> str:
    if not isinstance(value, int):
        value = int(value)
    return str(value)


def _quote_real(value: Union[float, numbers.Real]) -> str:
    if not isinstance(value, float):
        value = float(value)
    if math.isnan(value):
        return "nan()"
    if math.isinf(value):
        return "infinity()" if value > 0 else "-infinity()"
    return str(value)


def _quote_decimal(value: Decimal) -> str:
    return f"DECIMAL {_quote_str(str(value))}"


def _quote_bytes(value: bytes) -> str:
    encoded = base64.b64encode(value).decode("ascii")
    return f"from_base64({_quote_str(encoded)})"


def _quote_datetime(value: dt.datetime) -> str:
    encoded = value.isoformat(sep=" ", timespec="milliseconds")
    return f"TIMESTAMP {_quote_str(encoded)}"


def _quote_date(value: dt.date) -> str:
    encoded = value.isoformat()
    return f"DATE {_quote_str(encoded)}"


def quote(value: SQL_SCALAR) -> str:
    """
    Quote scalar method to an SQL expression.

    :param value: Python value
    :return: SQL expression
    """
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return _quote_str(value)
    if isinstance(value, int) or isinstance(value, numbers.Integral):
        return _quote_integral(value)
    if isinstance(value, float) or isinstance(value, numbers.Real):
        return _quote_real(value)
    if isinstance(value, Decimal):
        return _quote_decimal(value)
    if isinstance(value, dt.datetime):
        return _quote_datetime(value)
    if isinstance(value, dt.date):
        return _quote_date(value)
    if isinstance(value, bytes):
        return _quote_bytes(value)
    raise TypeError(f"Cannot quote {type(value)}.")
