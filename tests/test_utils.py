import itertools

from pallas.utils import Fibonacci, truncate_str


def test_fibonacci_wo_max_value():
    generator = Fibonacci()
    items = itertools.islice(generator, 12)
    assert list(items) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


def test_fibonacci_w_max_value():
    generator = Fibonacci(max_value=60)
    items = itertools.islice(generator, 12)
    assert list(items) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 60, 60]


def test_truncate_str_short():
    assert truncate_str("Hello world!") == "Hello world!"


def test_truncate_str_long():
    s = "Hello, " + 20 * "hello, " + "world!"
    assert truncate_str(s) == (
        "Hello, hello, hello, hello, hello, hello, hello,"
        " hell...lo, hello, hello, world!"
    )
