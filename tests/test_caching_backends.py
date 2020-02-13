import pytest

from pallas.caching.backends import CacheMiss, MemoryCache, FileCache


@pytest.fixture(name="memory_cache")
def memory_cache_fixture(tmp_path):
    return MemoryCache()


@pytest.fixture(name="file_cache")
def file_cache_fixture(tmp_path):
    return FileCache(tmp_path)


@pytest.fixture(name="cache", params=["memory", "file"])
def cache_fixture(request):
    yield request.getfixturevalue(f"{request.param}_cache")


class TestCaches:
    def test_get_missing(self, cache):
        with pytest.raises(CacheMiss):
            cache.get("foo")

    def test_set_get(self, cache):
        cache.set("foo", "Hello Foo!")
        assert cache.get("foo") == "Hello Foo!"

    def test_has_missing(self, cache):
        assert not cache.has("foo")

    def test_has_present(self, cache):
        cache.set("foo", "Hello Foo!")
        assert cache.has("foo")

    def test_read_missing(self, cache):
        with pytest.raises(CacheMiss):
            cache.reader("foo")

    def test_read(self, cache):
        cache.set("foo", "Hello Foo,\nHope you are doing well.")
        with cache.reader("foo") as stream:
            assert stream.readline() == "Hello Foo,\n"
            assert stream.readline() == "Hope you are doing well."

    def test_write(self, cache):
        with cache.writer("foo") as stream:
            stream.write("Hello Foo,\n")
            stream.write("Hope you are doing well.")
        assert cache.get("foo") == "Hello Foo,\nHope you are doing well."
