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


def convert_value(value_type: str, value: Optional[str]) -> Any:
    if value is None:
        return None
    converter = CONVERTERS.get(value_type)
    if not converter:
        return value
    return converter(value)
