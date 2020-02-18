from __future__ import annotations

from typing import Any, Dict, List, Optional


def parse_int(v: str) -> int:
    """
    Parse an integral number returned by Athena.
    """
    return int(v)


def parse_float(v: str) -> float:
    """
    Parse a floating-point number returned by Athena.
    """
    return float(v)


def parse_bool(v: str) -> bool:
    """
    Parse a boolean value returned by Athena.
    """
    return {"true": True, "false": False}[v]


def parse_array(value: str) -> List[str]:
    """
    Parse string returned by Athena to a list.

    Always returns a list of strings because Athena does
    not send more details about item types.
    """
    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError(f"Invalid array value: {value!r}")
    return value[1:-1].split(", ")


def parse_map(value: str) -> Dict[str, str]:
    """
    Convert string value returned from Athena to a dictionary.

    Always returns a mapping from strings to strings because
    Athena does not send more details about item types.
    """
    if not value.startswith("{") or not value.endswith("}"):
        raise ValueError(f"Invalid map value: {value!r}")
    parts = value[1:-1].split(", ")
    return {k: v for k, _, v in (part.partition("=") for part in parts)}


PARSERS = {
    "boolean": parse_bool,
    "tinyint": parse_int,
    "smallint": parse_int,
    "integer": parse_int,
    "bigint": parse_int,
    "float": parse_float,
    "double": parse_float,
    "map": parse_map,
    "array": parse_array,
}


def parse_value(column_type: str, value: Optional[str]) -> Any:
    """
    Parse value returned by Athena to a corresponding Python type.

    :param column_type: column type as reported by Athena
    :param value: string column value as returned by Athena
    :return: Python value
    """
    if value is None:
        return None
    converter = PARSERS.get(column_type)
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
    "map": "object",
    "array": "object",
}


def get_dtype(column_type: str) -> str:
    """
    Return Pandas dtype for an Athena type.
    :param column_type: column type as reported by Athena
    :return: Pandas dtype
    """
    return DTYPES.get(column_type, "str")
