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

import io
from abc import ABCMeta, abstractmethod
from typing import TextIO


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

    Implementations re used  as cache backends.
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.uri!r}>"

    def __str__(self) -> str:
        return self.uri

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
