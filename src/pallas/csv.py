import io
import re
from typing import Iterable, Iterator, Optional, Sequence, TextIO, Tuple

CSVValue = Optional[str]
CSVRow = Sequence[CSVValue]


control_re = re.compile(r'\n|,|"')
quote_re = re.compile(r'"')


def _encode_value(value: CSVValue) -> str:
    if value is None:
        return ""
    quoted = value.replace('"', '""')
    return f'"{quoted}"'


def _decode_value(raw: str) -> CSVValue:
    if raw == "":
        return None
    if not (raw.startswith('"') and raw.endswith('"')):
        raise ValueError(raw)
    parts = raw[1:-1].split('""')
    if any('"' in part for part in parts):
        raise ValueError(f"Invalid value: {raw}")
    return '"'.join(parts)


def _tokenize(stream: TextIO) -> Iterator[Tuple[str, str]]:
    pos = 0
    buffer = ""
    value = ""
    quoted = False
    while True:
        if not pos < len(buffer):
            pos = 0
            buffer = stream.read(io.DEFAULT_BUFFER_SIZE)
            if not buffer:
                break
        pattern = quote_re if quoted else control_re
        match = pattern.search(buffer, pos)
        if match:
            start, end = match.span()
            value += buffer[pos:start]
            control = buffer[start:end]
            pos = end
        else:
            value += buffer[pos:]
            control = ""
            pos = len(buffer)
        if control == '"':
            value += control
            quoted = not quoted
        elif control:
            yield value, control
            value = ""
    if value:
        yield value, ""


def read_csv(stream: TextIO) -> Iterator[CSVRow]:
    """
    Read CSV using format that Athena uses.

    All values are quoted (quotes are doubled).
    Empty unquoted string denotes ``None``.

    :param stream: readable file-like
    :return: sequence of records
    """
    row = []
    for value, control in _tokenize(stream):
        row.append(_decode_value(value))
        if control == "\n":
            yield tuple(row)
            row = []
    if row:
        raise ValueError("Missing trailing newline.")


def write_csv(data: Iterable[CSVRow], stream: TextIO) -> None:
    """
    Write CSV using format that Athena uses.

    All values are quoted (quotes are doubled).
    Empty unquoted string denotes ``None``.

    :param data: sequence of records
    :param stream: writable file-like
    """
    for row in data:
        line = ",".join(_encode_value(v) for v in row)
        stream.write(line + "\n")
