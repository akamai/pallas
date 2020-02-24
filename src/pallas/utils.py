"""
Assorted helpers.
"""

from __future__ import annotations

from typing import Iterator, Optional


class Fibonacci:

    max_value: Optional[int]

    def __init__(self, *, max_value: Optional[int] = None) -> None:
        self.max_value = max_value

    def __iter__(self) -> Iterator[int]:
        a = b = 1
        while self.max_value is None or a < self.max_value:
            yield a
            a, b = b, a + b
        while True:
            yield self.max_value


def truncate_str(v: str, max_length: int = 80) -> str:
    """Trim the given text if too long."""
    if len(v) <= max_length:
        return v
    head = max_length * 2 // 3
    tail = max_length - head - 3
    return v[:head] + "..." + v[-tail:]
