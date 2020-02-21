from __future__ import annotations

from typing import Dict
from urllib.parse import urlsplit

from pallas.storage.base import NotFoundError, Storage, UnsupportedURIError


def memory_parse_uri(uri: str) -> None:
    """
    Check that memory URI is valid.
    """
    scheme, netloc, path, query, fragment = urlsplit(uri, scheme="memory")
    if scheme != "memory":
        raise UnsupportedURIError("Not a memory scheme.")
    if netloc or path or query or fragment:
        raise UnsupportedURIError("Superfluous component.")


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
        memory_parse_uri(uri)
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
