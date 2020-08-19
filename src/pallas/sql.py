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

from __future__ import annotations

import base64
import datetime as dt
import math
import numbers
import re
import textwrap
from decimal import Decimal
from typing import List, Mapping, Tuple, Union

SQL_SCALAR = Union[None, str, float, numbers.Real, Decimal, bytes, dt.date]

# Parameters supported for SQL interpolation.
#
# Intentionally use Tuple and List instead of Sequence because
# we do not want to accept strings.
PARAMETERS = Union[
    None, Tuple[SQL_SCALAR, ...], List[SQL_SCALAR], Mapping[str, SQL_SCALAR]
]


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
    Quote a scalar value for an SQL expression.

    Parametrized queries should be preferred to explicit quoting.

    Following Python types can be quoted to an SQL expressions:

    - :data:`None` – SQL ``NULL``
    - :class:`str`
    - :class:`int`, including subclasses of numbers.Integral
    - :class:`float`, including subclasses or numbers.Real
    - :class:`Decimal` – SQL ``DECIMAL``
    - :class:`datetime.date` – SQL ``DATE``
    - :class:`datetime.datetime` – SQL ``TIMESTAMP``
    - :class:`bytes` – SQL ``VARBINARY``

    :param value: Python value
    :return: an SQL expression
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


def substitute_parameters(operation: str, parameters: PARAMETERS = None) -> str:
    """
    Substitute parameters in SQL query.
    """
    if parameters is None:
        # When no parameters are given, no substitution happens,
        # so no special quoting is necessary.
        # This is consistent with psycopg2 or MySQLdb behavior.
        return operation
    elif isinstance(parameters, Mapping):
        return operation % {name: quote(param) for name, param in parameters.items()}
    elif isinstance(parameters, (list, tuple)):
        return operation % tuple(quote(param) for param in parameters)
    raise TypeError("SQL parameters must be a sequence or a mapping.")


_comment_1 = r"--[^\n]*\n"
_comment_2 = r"/\*([^*]|\*(?!/))*\*/"

SELECT_RE = re.compile(
    rf"(\s+|{_comment_1}|{_comment_2})*(SELECT|WITH)\b", re.IGNORECASE
)


def is_select(sql: str) -> bool:
    """
    Return whether an SQL statement is SELECT.

    Only SELECT statements are considered cacheable.
    """
    return SELECT_RE.match(sql) is not None


def normalize_sql(sql: str) -> str:
    """
    Normalizes an SQL query.

    Query normalization can improve caching.

    Following normalization operations are done:
    - Common indentation is removed.
    - Heading and trailing new lines are removed.
    - Trailing whitespace is removed from end of lines.
    - Line endings are normalized to LF
    """
    lines = sql.splitlines()
    joined = "\n".join(line.rstrip() for line in lines)
    return textwrap.dedent(joined).strip()
