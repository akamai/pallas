from __future__ import annotations

import io
import pathlib
from abc import ABCMeta, abstractmethod
from typing import Dict, TextIO


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

        Raises `CacheMiss` if the key is not in the cache.
        """

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """
        Set value of the given cache key.
        """

    def has(self, key: str) -> bool:
        """
        Test presence of the given cache key.

        It is more efficient to try to retrieve a value and catch
        possible exception than to test its presence first.
        """
        try:
            self.get(key)
        except CacheMiss:
            return False
        return True

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
