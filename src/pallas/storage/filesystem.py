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

import pathlib
from typing import TextIO, Union
from urllib.parse import urlsplit

from pallas.storage.base import NotFoundError, Storage, UnsupportedURIError


def file_parse_uri(uri: str) -> str:
    """
    Parse path from file URI.
    """
    scheme, netloc, path, query, fragment = urlsplit(uri, scheme="file")
    if scheme != "file":
        raise UnsupportedURIError("Not a file scheme.")
    if netloc:
        raise UnsupportedURIError("The scheme does not support hostname.")
    if query or fragment:
        raise UnsupportedURIError("The scheme does not support query or fragment.")
    if not path:
        raise UnsupportedURIError("Path is empty.")
    return path


class FileSystemStorage(Storage):
    """
    Storage implementation storing data on a local filesystem.
    """

    _initialized: bool = False
    _base_dir: pathlib.Path

    def __init__(self, base_dir: Union[str, pathlib.Path]) -> None:
        self._base_dir = pathlib.Path(base_dir).expanduser().absolute()

    @classmethod
    def from_uri(cls, uri: str) -> FileSystemStorage:
        path = file_parse_uri(uri)
        return cls(path)

    @property
    def uri(self) -> str:
        return f"file:{self._base_dir}/"

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
