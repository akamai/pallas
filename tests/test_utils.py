from pallas.utils import truncate_str


def test_truncate_str_short():
    assert truncate_str("Hello world!") == "Hello world!"


def test_truncate_str_long():
    s = "Hello, " + 20 * "hello, " + "world!"
    assert truncate_str(s) == (
        "Hello, hello, hello, hello, hello, hello, hello,"
        " hell...lo, hello, hello, world!"
    )
