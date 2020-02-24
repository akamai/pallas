"""
Storage implementations for caching results.
"""

from __future__ import annotations

from typing import Mapping, Type
from urllib.parse import urlsplit

from pallas.storage.base import NotFoundError, Storage, UnsupportedURIError
from pallas.storage.filesystem import FileSystemStorage
from pallas.storage.memory import MemoryStorage
from pallas.storage.s3 import S3Storage

__all__ = [
    "Storage",
    "UnsupportedURIError",
    "NotFoundError",
    "storage_from_uri",
]


STORAGE_REGISTRY: Mapping[str, Type[Storage]] = {
    "memory": MemoryStorage,
    "file": FileSystemStorage,
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
