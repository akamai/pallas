from typing import Iterator


class Fibonacci:

    max_value: int

    def __init__(self, *, max_value: int = 600) -> None:
        self.max_value = max_value

    def __iter__(self) -> Iterator[int]:
        a = b = 1
        while a < self.max_value:
            yield a
            a, b = b, a + b
        while True:
            yield self.max_value
