"""
SQL helpers.

Implements quoting compatible with Athena.
"""

import math
import numbers
from typing import Union

SQL_SCALAR = Union[None, str, float, numbers.Real]


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
    raise TypeError(f"Cannot quote {type(value)}.")
