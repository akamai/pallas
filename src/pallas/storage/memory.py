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

    def size(self) -> int:
        return len(self._data)
