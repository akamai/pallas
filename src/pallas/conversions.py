from __future__ import annotations

import datetime as dt
import json
from abc import ABCMeta, abstractmethod
from decimal import Decimal
from typing import Dict, Generic, List, Optional, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Converter(Generic[T_co], metaclass=ABCMeta):
    """
    Convert values returned by Athena to Python types.
    """

    @property
    @abstractmethod
    def dtype(self) -> object:
        """Pandas dtype"""

    def read(self, value: Optional[str]) -> Optional[T_co]:
        if value is None:
            return None
        return self.read_str(value)

    @abstractmethod
    def read_str(self, value: str) -> T_co:
        """Read value from string"""


class TextConverter(Converter[str]):
    @property
    def dtype(self) -> object:
        return "str"

    def read_str(self, value: str) -> str:
        return value


class BooleanConverter(Converter[bool]):
    @property
    def dtype(self) -> object:
        return "bool"

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
        return f"datetime64[ns]"

    def read_str(self, value: str) -> dt.date:
        return dt.date.fromisoformat(value)


class DateTimeConverter(Converter[dt.datetime]):
    @property
    def dtype(self) -> object:
        return f"datetime64[ns]"

    def read_str(self, value: str) -> dt.datetime:
        return dt.datetime.fromisoformat(value)


class BinaryConverter(Converter[bytes]):
    @property
    def dtype(self) -> object:
        return "bytes"

    def read_str(self, value: str) -> bytes:
        return bytes.fromhex(value)


class ArrayConverter(Converter[List[str]]):
    """
    Parse string returned by Athena to a list.

    Always returns a list of strings because Athena does
    not send more details about item types.
    """

    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> List[str]:
        if not value.startswith("[") or not value.endswith("]"):
            raise ValueError(f"Invalid array value: {value!r}")
        return value[1:-1].split(", ")


class JSONConverter(Converter[object]):
    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> object:
        return json.loads(value)


class MapConverter(Converter[Dict[str, str]]):
    """
    Convert string value returned from Athena to a dictionary.

    Always returns a mapping from strings to strings because
    Athena does not send more details about item types.
    """

    @property
    def dtype(self) -> object:
        return "object"

    def read_str(self, value: str) -> Dict[str, str]:
        if not value.startswith("{") or not value.endswith("}"):
            raise ValueError(f"Invalid map value: {value!r}")
        parts = value[1:-1].split(", ")
        return {k: v for k, _, v in (part.partition("=") for part in parts)}


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
    "map": MapConverter(),
    "array": ArrayConverter(),
    "json": JSONConverter(),
}


def get_converter(column_type: str) -> Converter[object]:
    """
    Return a converter for a column type.

    :param column_type: a column type as reported by Athena
    :return: a converter instance.
    """
    return CONVERTERS.get(column_type, default_converter)
