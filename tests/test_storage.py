import os

import pytest

from pallas.storage import (
    FileStorage,
    MemoryStorage,
    NotFoundError,
    S3Storage,
    storage_from_uri,
)


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


class TestFromURI:
    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            storage_from_uri("unknown://")

    @pytest.mark.parametrize("uri", ["memory:", "memory://"])
    def test_memory_from_uri(self, uri):
        storage = storage_from_uri(uri)
        assert isinstance(storage, MemoryStorage)
        assert storage.uri == "memory:"

    @pytest.mark.parametrize(
        "uri",
        ["memory://netloc", "memory:path", "memory:?query=1", "memory:#fragment"],
    )
    def test_invalid_memory_from_uri(self, uri):
        with pytest.raises(ValueError):
            storage_from_uri(uri)

    @pytest.mark.parametrize(
        "uri,base_dir",
        [
            ("/path", "/path"),
            ("path", os.path.abspath("./path")),
            ("./path", os.path.abspath("./path")),
            ("~/path", os.path.expanduser("~/path")),
            ("file:/path", "/path"),
            ("file:path", os.path.abspath("./path")),
            ("file:./path", os.path.abspath("./path")),
            ("file:~/path", os.path.expanduser("~/path")),
            ("file:///path", "/path"),
        ],
    )
    def test_file_from_uri(self, uri, base_dir):
        storage = storage_from_uri(uri)
        assert isinstance(storage, FileStorage)
        assert storage.uri == f"file:{base_dir}"
        assert str(storage.base_dir) == base_dir

    @pytest.mark.parametrize(
        "uri",
        [
            "file:",
            "file://",
            "file://netloc",
            "file:/path?query=1",
            "file:/path#fragment",
        ],
    )
    def test_invalid_file_from_uri(self, uri):
        with pytest.raises(ValueError):
            storage_from_uri(uri)

    @pytest.mark.parametrize(
        "uri,bucket,prefix",
        [
            ("s3://bucket", "bucket", ""),
            ("s3://bucket/", "bucket", ""),
            ("s3://bucket/path", "bucket", "path/"),
            ("s3://bucket/path/", "bucket", "path/"),
        ],
    )
    def test_s3_from_uri(self, uri, bucket, prefix):
        storage = storage_from_uri(uri)
        assert isinstance(storage, S3Storage)
        assert storage.uri == f"s3://{bucket}/{prefix}"
        assert str(storage.bucket) == bucket
        assert str(storage.prefix) == prefix

    @pytest.mark.parametrize(
        "uri",
        [
            "s3:",
            "s3://",
            "s3:path",
            "s3:/path",
            "s3:///path",
            "s3://bucket?query=1",
            "s3://bucket#fragment",
        ],
    )
    def test_invalid_s3_from_uri(self, uri):
        with pytest.raises(ValueError):
            storage_from_uri(uri)
