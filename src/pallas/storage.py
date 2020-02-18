from __future__ import annotations

import io
import pathlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, TextIO, Type, Union
from urllib.parse import urlsplit

import boto3
import botocore


class UnsupportedURIError(ValueError):
    """
    Storage URI not supported.
    """


class NotFoundError(KeyError):
    """
    Storage item not found.

    Raised when trying to read a non-existent item.
    """


class StorageWriter(io.StringIO):
    """
    Writable stream that saves content to :class:`.Storage` on close.

    Used by the default implementation of the :meth:`.Storage.writer` method.
    """

    def __init__(self, storage: Storage, key: str) -> None:
        super().__init__()
        self._storage = storage
        self._key = key

    def close(self) -> None:
        self._storage.set(self._key, self.getvalue())
        super().close()


class Storage(metaclass=ABCMeta):
    """
    Interface for a simple key-value storage.

    Implementations of are used primarily as cache backends,
    but other use cases are possible.
    For example, the :class:`.S3Storage` class can be used
    for downloading query results form S3.
    """

    @classmethod
    @abstractmethod
    def from_uri(cls, uri: str) -> Storage:
        """
        Construct an instance from a string URI.
        """

    @property
    @abstractmethod
    def uri(self) -> str:
        """
        Return a string URI describing this storage instance.
        """

    @abstractmethod
    def get(self, key: str) -> str:
        """
        Get value of the given key.

        Provides a simple method for retrieving short values.
        For longer content, the :meth:`.reader` can be better.

        Raises :exception:`.NotFoundError` if the key is not found.
        """

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """
        Set value of the given key.

        Provides a simple method for storing short values.
        For longer content, use the :meth:`.writer` method.
        """

    @abstractmethod
    def has(self, key: str) -> bool:
        """
        Test presence of the given key.

        In most cases, it is more efficient to try to retrieve a value
        and catch possible exception than to check presence first.
        """

    def reader(self, key: str) -> TextIO:
        """
        Return a file-like object opened for reading.

        The default implementation returns a in-memory stream
        with the whole content read from storage,
        but subclasses can provide a more efficient method.

        Raises :exception:`.NotFoundError` if the key is not found.
        """
        return io.StringIO(self.get(key))

    def writer(self, key: str) -> TextIO:
        """
        Return a file-like object opened for writing.

        The default implementation buffers written content in memory
        and stores it at once on close,
        but subclasses can provide a more efficient method.
        """
        return StorageWriter(self, key)


class MemoryStorage(Storage):
    """
    Storage implementation storing data in memory.

    Useful mainly for testing.
    """

    _data: Dict[str, str]

    def __init__(self) -> None:
        self._data = {}

    @classmethod
    def from_uri(cls, uri: str) -> MemoryStorage:
        scheme, netloc, path, query, fragment = urlsplit(uri, scheme="memory")
        if scheme != "memory":
            raise UnsupportedURIError("Not a memory scheme.")
        if netloc or path or query or fragment:
            raise UnsupportedURIError("Superfluous component.")
        return cls()

    @property
    def uri(self) -> str:
        return "memory:"

    def get(self, key: str) -> str:
        try:
            return self._data[key]
        except KeyError:
            raise NotFoundError(key) from None

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data


class FileStorage(Storage):
    """
    Storage implementation storing data on a local filesystem.
    """

    _initialized: bool = False
    _base_dir: pathlib.Path

    def __init__(self, base_dir: Union[str, pathlib.Path]) -> None:
        self._base_dir = pathlib.Path(base_dir).expanduser().absolute()

    @classmethod
    def from_uri(cls, uri: str) -> FileStorage:
        scheme, netloc, path, query, fragment = urlsplit(uri, scheme="file")
        if scheme != "file":
            raise UnsupportedURIError("Not a file scheme.")
        if netloc:
            raise UnsupportedURIError("The scheme does not support hostname.")
        if query or fragment:
            raise UnsupportedURIError("The scheme does not support query or fragment.")
        if not path:
            raise UnsupportedURIError("Path is empty.")
        return cls(path)

    @property
    def uri(self) -> str:
        return f"file:{self._base_dir}"

    @property
    def base_dir(self) -> pathlib.Path:
        return self._base_dir

    def get(self, key: str) -> str:
        try:
            return self._get_file(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NotFoundError(key) from None

    def set(self, key: str, value: str) -> None:
        self._get_file(key).write_text(value, encoding="utf-8")

    def has(self, key: str) -> bool:
        return self._get_file(key).exists()

    def reader(self, key: str) -> TextIO:
        try:
            return self._get_file(key).open("r", encoding="utf-8", newline="")
        except FileNotFoundError:
            raise NotFoundError(key) from None

    def writer(self, key: str) -> TextIO:
        return self._get_file(key).open("w", encoding="utf-8", newline="")

    def _get_file(self, key: str) -> pathlib.Path:
        if not self._initialized:
            # Create cache directory when first used.
            self._base_dir.mkdir(exist_ok=True, parents=True)
            self._initialized = True
        return self.base_dir / key


class S3Storage(Storage):
    """
    Storage implementation storing data in AWS S3.
    """

    _client: Any  # boto3 S3 client

    def __init__(
        self, bucket: str, prefix: str = "", *, client: Optional[Any] = None
    ) -> None:
        if client is None:
            client = boto3.client("s3")
        self._client = client
        self._bucket = bucket
        self._prefix = prefix

    @classmethod
    def from_uri(cls, uri: str) -> S3Storage:
        scheme, netloc, path, query, fragment = urlsplit(uri, scheme="s3")
        if scheme != "s3":
            raise UnsupportedURIError("Not a s3 scheme.")
        if query or fragment:
            raise UnsupportedURIError("The scheme does not support query or fragment.")
        if not netloc:
            raise UnsupportedURIError("Bucket is empty.")
        prefix = path.strip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return cls(netloc, prefix)

    @property
    def uri(self) -> str:
        return f"s3://{self._bucket}/{self._prefix}"

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    def get(self, key: str) -> str:
        with self.reader(key) as stream:
            return stream.read()

    def set(self, key: str, value: str) -> None:
        params = self._get_params(key)
        params["Body"] = value.encode("utf-8")
        self._client.put_object(**params)

    def has(self, key: str) -> bool:
        params = self._get_params(key)
        try:
            self._client.head_object(**params)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
        return True

    def reader(self, key: str) -> TextIO:
        params = self._get_params(key)
        try:
            response = self._client.get_object(**params)
        except self._client.exceptions.NoSuchKey:
            raise NotFoundError(key) from None
        # Use _raw_stream (urllib3.response.HTTPResponse) because
        # boto3 wrapper does not implement full Binary I/O interface.
        raw = response["Body"]._raw_stream
        # TextIOWrapper does not work with auto_close
        # https://urllib3.readthedocs.io/en/latest/user-guide.html#using-io-wrappers-with-response-content
        raw.auto_close = False
        return io.TextIOWrapper(raw, encoding="utf-8", newline="")

    def _get_params(self, key: str) -> Dict[str, Any]:
        return dict(Bucket=self.bucket, Key=self.prefix + key)


STORAGE_REGISTRY: Dict[str, Type[Storage]] = {
    "memory": MemoryStorage,
    "file": FileStorage,
    "s3": S3Storage,
}


def storage_from_uri(uri: str, *, default_scheme: str = "file") -> Storage:
    """
    Construct a storage instance from a string URI.
    """
    scheme, *rest = urlsplit(uri, scheme=default_scheme)
    try:
        cls = STORAGE_REGISTRY[scheme]
    except KeyError:
        raise UnsupportedURIError(f"Unknown scheme: {scheme}.") from None
    return cls.from_uri(uri)
