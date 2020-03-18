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
from typing import Any, Dict, Optional, TextIO, Tuple
from urllib.parse import urlsplit

import boto3
import botocore

from pallas.storage.base import NotFoundError, Storage, UnsupportedURIError


def s3_parse_uri(uri: str) -> Tuple[str, str]:
    """
    Get bucket name and path from an S3 uri.
    """
    scheme, netloc, path, query, fragment = urlsplit(uri, scheme="s3")
    if scheme != "s3":
        raise UnsupportedURIError("Not a s3 scheme.")
    if query or fragment:
        raise UnsupportedURIError("The scheme does not support query or fragment.")
    if not netloc:
        raise UnsupportedURIError("Bucket is empty.")
    return netloc, path.lstrip("/")


def s3_wrap_body(body: Any) -> TextIO:
    """
    Wrap body returned by S3 GetObject to a valid TextIO
    """
    # Use _raw_stream (urllib3.response.HTTPResponse) because
    # boto3 wrapper does not implement full Binary I/O interface.
    raw = body._raw_stream
    # TextIOWrapper does not work with auto_close
    # https://urllib3.readthedocs.io/en/latest/user-guide.html#using-io-wrappers-with-response-content
    raw.auto_close = False
    return io.TextIOWrapper(raw, encoding="utf-8", newline="")


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
        bucket, prefix = s3_parse_uri(uri)
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return cls(bucket, prefix)

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
        return s3_wrap_body(response["Body"])

    def _get_params(self, key: str) -> Dict[str, Any]:
        return dict(Bucket=self.bucket, Key=self.prefix + key)
