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
Methods for setting up Athena clients.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, TextIO

from pallas.caching import AthenaCache
from pallas.client import Athena
from pallas.proxies import Boto3Proxy


def configure_logging(  # type: ignore
    *,
    level: int | str = logging.INFO,
    stream: TextIO = sys.stdout,
    **kwargs,
) -> None:
    """
    Do basic configuration for the logging system.

    Calls :func:`logging.basicConfig` internally, but:

    - Sets *level* to the "pallas" logger only
    - Log level default to "INFO:
    - *stream* defaults to :attr:`sys.stdout` instead of :attr:`sys.stderr`

    Can be safely called no matter whether logging was already configured:

    - If logging was already configured, this function just sets
      a level for the "pallas" logger.
    - If logging was not configured yet, it enabled logging to stdout.

    :param level: Set the "pallas" logger to the specified level.
    :param stream: Use the specified stream to initialize the StreamHandler.
    :param kwargs: passed to :func:`logging.basicConfig()`
    """
    logging.basicConfig(stream=stream, **kwargs)
    logging.getLogger("pallas").setLevel(level)


def setup(
    *,
    region: str | None = None,
    database: str | None = None,
    workgroup: str | None = None,
    output_location: str | None = None,
    cache_local: str | None = None,
    cache_remote: str | None = None,
    cache_failed: bool = AthenaCache.failed,
    normalize: bool = Athena.normalize,
    kill_on_interrupt: bool = Athena.kill_on_interrupt,
) -> Athena:
    """
    Setup an :class:`.Athena` client.

    All configuration options can be given to this method,
    but many of them can be overridden after the client is constructed.

    :param region: an AWS region.
        By default, region from AWS config (``~/.aws/config``) is used.
    :param database: a name of Athena database.
        Can be overridden in SQL.
    :param workgroup: a name of Athena workgroup.
        Workgroup can set resource limits or override output location.
        Defaults to the Athena default workgroup.
    :param output_location: an output location at S3 for query results.
        Optional if an output location is specified for the *workgroup*.
    :param cache_local: an URI of a local cache.
        Both results and query execution IDs are stored in the local cache.
    :param cache_remote: an URI of a remote cache.
        Query execution IDs without results are stored in the remote cache.
    :param cache_failed: whether to return failed queries found in cache.
    :param normalize: whether to normalize queries before execution.
    :param kill_on_interrupt: whether to kill queries on KeyboardInterrupt.
    :return: a new instance of Athena client
    """
    athena = Athena(Boto3Proxy(region=region))
    if cache_local is not None:
        athena.cache.local = cache_local
    if cache_remote is not None:
        athena.cache.remote = cache_remote
    athena.cache.failed = cache_failed
    athena.database = database
    athena.workgroup = workgroup
    athena.output_location = output_location
    athena.normalize = normalize
    athena.kill_on_interrupt = kill_on_interrupt
    return athena


def environ_setup(
    environ: Mapping[str, str] | None = None, *, prefix: str = "PALLAS"
) -> Athena:
    """
    Setup an :class:`.Athena` client from environment variables.

    Reads the following environment variables: ::

        export PALLAS_REGION=
        export PALLAS_DATABASE=
        export PALLAS_WORKGROUP=
        export PALLAS_OUTPUT_LOCATION=
        export PALLAS_NORMALIZE=true
        export PALLAS_KILL_ON_INTERRUPT=true
        export PALLAS_CACHE_REMOTE=$PALLAS_OUTPUT_LOCATION
        export PALLAS_CACHE_LOCAL=~/Notebooks/.cache/

    Configuration from the environment variables can be overridden
    after the client is constructed.

    :param environ: A mapping object representing the string environment.
        Defaults to ``os.environ``.
    :param prefix: A prefix of environment variables
    :return: a new instance of Athena client
    """
    config = EnvironConfig(environ, prefix=prefix)
    return setup(
        database=config.get_str("DATABASE"),
        workgroup=config.get_str("WORKGROUP"),
        output_location=config.get_str("OUTPUT_LOCATION"),
        region=config.get_str("REGION"),
        cache_remote=config.get_str("CACHE_REMOTE"),
        cache_local=config.get_str("CACHE_LOCAL"),
        cache_failed=config.get_bool("CACHE_FAILED", AthenaCache.failed),
        normalize=config.get_bool("NORMALIZE", Athena.normalize),
        kill_on_interrupt=config.get_bool(
            "KILL_ON_INTERRUPT", Athena.kill_on_interrupt
        ),
    )


class EnvironConfig:
    def __init__(
        self, environ: Mapping[str, str] | None = None, *, prefix: str
    ) -> None:
        if environ is None:
            environ = os.environ
        self._environ = environ
        self._prefix = prefix

    def get_str(self, key: str, default: str | None = None) -> str | None:
        v = self._get(key)
        if not v:
            return default
        return v

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self._get(key)
        if not v:
            return default
        v = v.lower()
        if v in ("1", "true", "on", "yes"):
            return True
        if v in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"{self._prefix}_{key}: invalid boolean value: {v}")

    def _get(self, key: str) -> str:
        v = self._environ.get(self.key(key), "")
        if not isinstance(v, str):
            # Avoid unexpected behaviour when somebody passes something
            # like {"PALLAS_KILL_ON_INTERRUPT": False} to environ.
            raise TypeError("Environ values must be a string.")
        return v

    def key(self, key: str) -> str:
        return f"{self._prefix}_{key}"
