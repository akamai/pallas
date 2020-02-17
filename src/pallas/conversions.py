from __future__ import annotations

from typing import Any, Optional


def convert_int(v: str) -> int:
    return int(v)


def convert_float(v: str) -> float:
    return float(v)


def convert_bool(v: str) -> bool:
    return {"true": True, "false": False}[v]


CONVERTERS = {
    "boolean": convert_bool,
    "tinyint": convert_int,
    "smallint": convert_int,
    "integer": convert_int,
    "bigint": convert_int,
    "float": convert_float,
    "double": convert_float,
}


def convert_value(column_type: str, value: Optional[str]) -> Any:
    if value is None:
        return None
    converter = CONVERTERS.get(column_type)
    if not converter:
        return value
    return converter(value)


DTYPES = {
    "boolean": "bool",
    "tinyint": "Int8",
    "smallint": "Int16",
    "integer": "Int32",
    "bigint": "Int64",
    "float": "float32",
    "double": "Float64",
}


def get_dtype(column_type: str) -> str:
    return DTYPES.get(column_type, "str")
