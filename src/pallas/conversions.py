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
Conversions from strings returned by Athena to Python types.
"""

from __future__ import annotations

import datetime as dt
import json
from abc import ABCMeta, abstractmethod
from decimal import Decimal
from typing import Dict, Generic, Iterable, List, Optional, Sequence, TypeVar

from pallas._compat import numpy as np
from pallas._compat import pandas as pd

T_co = TypeVar("T_co", covariant=True)


def _pd_array(values: Sequence[object], *, dtype: object) -> object:
    if dtype == "object":
        # Workaround for ValueError: PandasArray must be 1-dimensional.
        # When all values are lists of same length, Pandas/NumPy think
        # that we are constructing a 2-D array.
        data = np.empty(len(values), dtype="object")
        data[:] = values
    else:
        data = values
    return pd.array(data, dtype=dtype, copy=False)


class Converter(Generic[T_co], metaclass=ABCMeta):
    """
    Convert values returned by Athena to Python types.
    """

    @property
    @abstractmethod
    def dtype(self) -> object:
        """Pandas dtype"""

    def read(self, value: Optional[str]) -> Optional[T_co]:
        """
        Read value returned from Athena.

        Expect a string or ``None`` because optional strings
        are what Athena returns at its API and that is also
        what can be parsed from CSV stored in S3.
        """
        if value is None:
            return None
        return self.read_str(value)

    @abstractmethod
    def read_str(self, value: str) -> T_co:
        """
        Read value from string

        To be implemented in subclasses.
        """

    def read_array(
        self, values: Iterable[Optional[str]], dtype: Optional[object] = None,
    ) -> object:  # Pandas array
        """
        Convert values returned from Athena to Pandas array.

        :param values: Iterable yielding strings and ``None``
        :param dtype: optional Pandas dtype to force
        """
        if dtype is None:
            dtype = self.dtype
        converted = [self.read(value) for value in values]
        return _pd_array(converted, dtype=dtype)


class TextConverter(Converter[str]):
    @property
    def dtype(self) -> object:
        return "string"

    def read_str(self, value: str) -> str:
        return value


class BooleanConverter(Converter[bool]):
    @property
    def dtype(self) -> object:
        return "boolean"

    def read_str(self, value: str) -> bool:
        return {"true": True, "false": False}[value]


class IntConverter(Converter[int]):
    def __init__(self, size: int) -> None:
        self._size = size

    @property
    def dtype(self) -> object:
        return f"Int{self._size}"

    def read_str(self, value: str) -> int:
        return int(value)


class FloatConverter(Converter[float]):
    def __init__(self, size: int) -> None:
        self._size = size

    @property
    def dtype(self) -> object:
        return f"float{self._size}"

    def read_str(self, value: str) -> float:
        return float(value)


class DecimalConverter(Converter[Decimal]):
    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> Decimal:
        return Decimal(value)


class DateConverter(Converter[dt.date]):
    @property
    def dtype(self) -> object:
        return "datetime64[ns]"

    def read_str(self, value: str) -> dt.date:
        return dt.date.fromisoformat(value)


class DateTimeConverter(Converter[dt.datetime]):
    @property
    def dtype(self) -> object:
        return "datetime64[ns]"

    def read_str(self, value: str) -> dt.datetime:
        return dt.datetime.fromisoformat(value)


class BinaryConverter(Converter[bytes]):
    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> bytes:
        return bytes.fromhex(value)


class ArrayConverter(Converter[List[str]]):
    """
    Parse string returned by Athena to a list.

    Array parsing has multiple limitations because of the
    serialization format that Athena uses:

     - Always returns a list of strings because Athena does
       not send more details about item types.
     - It is not possible to distinguish comma in values from
       an item separator. We assume that values do not contain the comma.
     - We are not able to distinguish an empty array
       and an array with one empty string.
       This converter returns an empty array in that case.

    """

    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> List[str]:
        if not value.startswith("[") or not value.endswith("]"):
            raise ValueError(f"Invalid array value: {value!r}")
        content = value[1:-1]
        if not content:
            return []
        return content.split(", ")


class MapConverter(Converter[Dict[str, str]]):
    """
    Convert string value returned from Athena to a dictionary.

    Map parsing has multiple limitations because of the
    serialization format that Athena uses:

    - Always returns a mapping from strings to strings because
      Athena does not send more details about item types.
    - It is not possible to distinguish a comma or an equal sign
      in values from control characters.
      We assume that values do not contain the comma or the equal sign.
    """

    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> Dict[str, str]:
        if not value.startswith("{") or not value.endswith("}"):
            raise ValueError(f"Invalid map value: {value!r}")
        content = value[1:-1]
        if not content:
            return {}
        parts = (part.partition("=") for part in content.split(", "))
        return {k: v for k, _, v in parts}


class JSONConverter(Converter[object]):
    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> object:
        return json.loads(value)


default_converter = TextConverter()

CONVERTERS: Dict[str, Converter[object]] = {
    "boolean": BooleanConverter(),
    "tinyint": IntConverter(8),
    "smallint": IntConverter(16),
    "integer": IntConverter(32),
    "bigint": IntConverter(64),
    "float": FloatConverter(32),
    "double": FloatConverter(64),
    "decimal": DecimalConverter(),
    "date": DateConverter(),
    "timestamp": DateTimeConverter(),
    "varbinary": BinaryConverter(),
    "array": ArrayConverter(),
    "map": MapConverter(),
    "json": JSONConverter(),
}


def get_converter(column_type: str) -> Converter[object]:
    """
    Return a converter for a column type.

    :param column_type: a column type as reported by Athena
    :return: a converter instance.
    """
    return CONVERTERS.get(column_type, default_converter)
