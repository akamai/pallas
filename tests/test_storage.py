import pytest

from pallas.storage import NotFoundError, MemoryStorage, FileStorage, S3Storage


@pytest.fixture(name="memory_storage")
def memory_storage_fixture(tmp_path):
    return MemoryStorage()


@pytest.fixture(name="file_storage")
def file_storage_fixture(tmp_path):
    return FileStorage(tmp_path)


@pytest.fixture(name="s3_storage")
def s3_storage_fixture(s3_tmp_uri):
    return S3Storage.from_uri(s3_tmp_uri)


@pytest.fixture(name="storage", params=["memory", "file", "s3"])
def storage_fixture(request):
    """
    Yields all implemented storage implementations.
    """
    yield request.getfixturevalue(f"{request.param}_storage")


class TestStorage:
    def test_get_missing(self, storage):
        with pytest.raises(NotFoundError):
            storage.get("foo")

    def test_set_get(self, storage):
        storage.set("foo", "Hello Foo!")
        assert storage.get("foo") == "Hello Foo!"

    def test_has_missing(self, storage):
        assert not storage.has("foo")

    def test_has_present(self, storage):
        storage.set("foo", "Hello Foo!")
        assert storage.has("foo")

    def test_read_missing(self, storage):
        with pytest.raises(NotFoundError):
            storage.reader("foo")

    def test_read(self, storage):
        storage.set("foo", "Hello Foo,\nHope you are doing well.")
        with storage.reader("foo") as stream:
            assert stream.readline() == "Hello Foo,\n"
            assert stream.readline() == "Hope you are doing well."

    def test_write(self, storage):
        with storage.writer("foo") as stream:
            stream.write("Hello Foo,\n")
            stream.write("Hope you are doing well.")
        assert storage.get("foo") == "Hello Foo,\nHope you are doing well."
