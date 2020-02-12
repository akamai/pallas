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
    @abstractmethod
    def get(self, key: str) -> str:
        """Get value of the given cache key or ``None``."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set value of the given cache key."""

    def reader(self, key: str) -> TextIO:
        """Return a file-like object opened for reading or ``None``."""
        return io.StringIO(self.get(key))

    def writer(self, key: str) -> TextIO:
        """Return a file-like object opened for writing."""
        return _CacheWriter(self, key)


class MemoryCache(Cache):

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


class FileCache(Cache):
    def __init__(self, base_dir: str) -> None:
        self.base_dir = pathlib.Path(base_dir)

    def get(self, key: str) -> str:
        try:
            return self._get_cache_file(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CacheMiss(key) from None

    def set(self, key: str, value: str) -> None:
        self._get_cache_file(key).write_text(value, encoding="utf-8")

    def reader(self, key: str) -> TextIO:
        try:
            return self._get_cache_file(key).open("r", encoding="utf-8", newline="")
        except FileNotFoundError:
            raise CacheMiss(key) from None

    def writer(self, key: str) -> TextIO:
        return self._get_cache_file(key).open("w", encoding="utf-8", newline="")

    def _get_cache_file(self, key: str) -> pathlib.Path:
        return self.base_dir / key
