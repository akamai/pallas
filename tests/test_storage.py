import pytest

from pallas.storage import NotFoundError, MemoryStorage, FileStorage, S3Storage


@pytest.fixture(name="memory_cache")
def memory_cache_fixture(tmp_path):
    return MemoryStorage()


@pytest.fixture(name="file_cache")
def file_cache_fixture(tmp_path):
    return FileStorage(tmp_path)


@pytest.fixture(name="s3_cache")
def s3_cache_fixture(s3_tmp_uri):
    return S3Storage.from_uri(s3_tmp_uri)


@pytest.fixture(name="cache", params=["memory", "file", "s3"])
def cache_fixture(request):
    yield request.getfixturevalue(f"{request.param}_cache")


class TestCaches:
    def test_get_missing(self, cache):
        with pytest.raises(NotFoundError):
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
        with pytest.raises(NotFoundError):
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
