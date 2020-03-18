# Copyright 2020 Akamai Technologies, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
