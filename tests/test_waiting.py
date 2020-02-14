import itertools
from pallas.waiting import Fibonacci


def test_fibonacci_wo_max_value():
    generator = Fibonacci()
    items = itertools.islice(generator, 12)
    assert list(items) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def test_fibonacci_w_max_value():
    generator = Fibonacci(max_value=60)
    items = itertools.islice(generator, 12)
    assert list(items) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 60, 60]
