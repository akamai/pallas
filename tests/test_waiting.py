import itertools
from pallas.waiting import Fibonacci


def test_fibonacci_generator():
    generator = Fibonacci()
    items = itertools.islice(generator, 16)
    assert list(items) == [
        1,
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
        55,
        89,
        144,
        233,
        377,
        600,
        600,
    ]
