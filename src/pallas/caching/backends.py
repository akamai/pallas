from __future__ import annotations

import io
import pathlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, TextIO
from urllib.parse import urlsplit

import boto3
import botocore


class CacheMiss(KeyError):
    """Item not found in cache."""


class _CacheWriter(io.StringIO):
    def __init__(self, cache: Cache, key: str) -> None:
        super().__init__()
        self._cache = cache
        self._key = key

    def close(self) -> None:
        self._cache.set(self._key, self.getvalue())
        super().close()


class Cache(metaclass=ABCMeta):
    """
    Cache interface.
    """

    @abstractmethod
    def get(self, key: str) -> str:
        """
        Get value of the given cache key.

        Provides a simple method for retrieving short values.
        For longer content, the :meth:`.reader` can be better.

        Raises `CacheMiss` if the key is not in the cache.
        """

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """
        Set value of the given cache key.

        Provides a simple method for storing short values.
        For longer content, the :meth:`.writer` can be better.
        """

    @abstractmethod
    def has(self, key: str) -> bool:
        """
        Test presence of the given cache key.

        It is more efficient to try to retrieve a value
        and catch possible exception than to check presence first.
        """

    def reader(self, key: str) -> TextIO:
        """
        Return a file-like object opened for reading.

        Raises `CacheMiss` if the key is not in the cache.
        Content will be read from the cache.
        """
        return io.StringIO(self.get(key))

    def writer(self, key: str) -> TextIO:
        """
        Return a file-like object opened for writing.

        Written content will be stored in the cache.
        """
        return _CacheWriter(self, key)


class MemoryCache(Cache):
    """
    Cache implementation storing data in memory.
    """

    _data: Dict[str, str]

    def __init__(self) -> None:
        self._data = {}

    def get(self, key: str) -> str:
        try:
            return self._data[key]
        except KeyError:
            raise CacheMiss(key) from None

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data


class FileCache(Cache):
    """
    Cache implementation storing data on a local filesystem.
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = pathlib.Path(base_dir)

    def get(self, key: str) -> str:
        try:
            return self._get_cache_file(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CacheMiss(key) from None

    def set(self, key: str, value: str) -> None:
        self._get_cache_file(key).write_text(value, encoding="utf-8")

    def has(self, key: str) -> bool:
        return self._get_cache_file(key).exists()

    def reader(self, key: str) -> TextIO:
        try:
            return self._get_cache_file(key).open("r", encoding="utf-8", newline="")
        except FileNotFoundError:
            raise CacheMiss(key) from None

    def writer(self, key: str) -> TextIO:
        return self._get_cache_file(key).open("w", encoding="utf-8", newline="")

    def _get_cache_file(self, key: str) -> pathlib.Path:
        return self.base_dir / key


class S3Cache(Cache):

    client: Any  # boto3 S3 client

    def __init__(
        self, bucket: str, prefix: str = "", *, client: Optional[Any] = None
    ) -> None:
        if client is None:
            client = boto3.client("s3")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self._client = client
        self._bucket = bucket
        self._prefix = prefix

    @classmethod
    def from_uri(cls, uri: str) -> S3Cache:
        scheme, netloc, path, query, fragment = urlsplit(uri, scheme="s3")
        if scheme != "s3" or query or fragment:
            raise ValueError("Unsupported URI.")
        return cls(netloc, path)

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
            raise CacheMiss(key) from None
        # Use _raw_stream (urllib3.response.HTTPResponse) because
        # boto3 wrapper does not implement full Binary I/O interface.
        raw = response["Body"]._raw_stream
        # TextIOWrapper does not work with auto_close
        # https://urllib3.readthedocs.io/en/latest/user-guide.html#using-io-wrappers-with-response-content
        raw.auto_close = False
        return io.TextIOWrapper(raw, encoding="utf-8", newline="")

    def _get_params(self, key: str) -> Dict[str, Any]:
        return dict(Bucket=self._bucket, Key=self._prefix + key)
