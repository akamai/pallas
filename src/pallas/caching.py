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
Caching of Athena queries.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Optional
from urllib.parse import urlencode

from pallas.results import QueryResults
from pallas.storage import NotFoundError, Storage, storage_from_uri

logger = logging.getLogger("pallas")


class AthenaCache:
    """
    Caches queries and its results.

    Athena always stores results in S3, so it is possible
    to retrieve past results without manually copying data.

    This class can hold a reference to two instances of cache storage:

    - local, which caches both query execution IDs and query results
    - remote, which cache query execution IDs only.

    It is possible to configure one the backends, both of them,
    or none of them.

    Queries cached in the local storage can be executed without
    an internet connection.
    Queries cached in the remote storage are not executed twice,
    but results have to be downloaded from AWS.

    In theory, it is possible to use remote backend for the local
    cache (or vice versa), but we assume that the local cache
    is actually stored locally

    Instance of this class is returned by the :attr:`.Athena.cache` property.
    It can be updated to reconfigure the caching.
    """

    local_storage: Optional[Storage] = None
    remote_storage: Optional[Storage] = None

    #: Can be set to False to disable caching completely.
    #:
    #: Can be update to enable or disable the caching.
    enabled: bool = True

    #: Can be set to False to disable reading the cache.
    #:
    #: Can be update to reconfigure the caching.
    read: bool = True

    #: Can be set to False to disable writing the cache.
    #:
    #: Can be update to reconfigure the caching.
    write: bool = True

    @property
    def local(self) -> Optional[str]:
        """
        URI of storage for local cache.

        Can be updated to reconfigure the caching.
        """
        if self.local_storage is None:
            return None
        return self.local_storage.uri

    @local.setter
    def local(self, uri: Optional[str]) -> None:
        if uri is None:
            self.local_storage = None
        else:
            self.local_storage = storage_from_uri(uri)

    @property
    def remote(self) -> Optional[str]:
        """
        URI of storage for remote cache.

        Can be updated to reconfigure the caching.
        """
        if self.remote_storage is None:
            return None
        return self.remote_storage.uri

    @remote.setter
    def remote(self, uri: Optional[str]) -> None:
        if uri is None:
            self.remote_storage = None
        else:
            self.remote_storage = storage_from_uri(uri)

    def load_execution_id(self, database: Optional[str], sql: str) -> Optional[str]:
        """
        Retrieve cached query execution ID for the given SQL.

        Looks into both the local and the remote storage.
        """
        if not (self.enabled and self.read):
            return None
        key = self._get_execution_key(database, sql)
        for storage in self._execution_storages:
            try:
                execution_id = storage.get(key)
            except NotFoundError:
                continue
            logger.info(
                f"Query execution loaded from cache {storage}{key}:"
                f" QueryExecutionId={execution_id!r}"
            )
            return execution_id
        return None

    def save_execution_id(
        self, database: Optional[str], sql: str, execution_id: str
    ) -> None:
        """
        Store cached query execution ID for the given SQL.

        Updates both the local and the remote storage.
        """
        if not (self.enabled and self.write):
            return
        key = self._get_execution_key(database, sql)
        for storage in reversed(self._execution_storages):
            storage.set(key, execution_id)
            logger.info(
                f"Query execution saved to cache {storage}{key}:"
                f" QueryExecutionId={execution_id!r}"
            )

    def has_results(self, execution_id: str) -> bool:
        """
        Checks whether results are cached for the given execution ID.

        Looks into the local storage only.
        """
        if not (self.enabled and self.read):
            return False
        key = self._get_results_key(execution_id)
        for storage in self._results_storages:
            if not storage.has(key):
                continue
            logger.info(
                f"Query results are available in cache {storage}{key}:"
                f" QueryExecutionId={execution_id!r}"
            )
            return True
        return False

    def load_results(self, execution_id: str) -> Optional[QueryResults]:
        """
        Retrieve cached results for the given execution ID.

        Looks into the local storage only.
        """
        if not (self.enabled and self.read):
            return None
        key = self._get_results_key(execution_id)
        for storage in self._results_storages:
            try:
                with storage.reader(key) as stream:
                    results = QueryResults.load(stream)
            except NotFoundError:
                continue
            logger.info(
                f"Query results loaded from cache {storage}{key}:"
                f" QueryExecutionId={execution_id!r}: {len(results)} rows"
            )
            return results
        return None

    def save_results(self, execution_id: str, results: QueryResults) -> None:
        """
        Store cached results for the given SQL.

        Updates the local storage only.
        """
        if not (self.enabled and self.write):
            return
        key = self._get_results_key(execution_id)
        for storage in reversed(self._results_storages):
            with storage.writer(key) as stream:
                results.save(stream)
            logger.info(
                f"Query results saved to cache {storage}{key}:"
                f" QueryExecutionId={execution_id!r}: {len(results)} rows"
            )

    @property
    def _execution_storages(self) -> List[Storage]:
        candidates = [self.local_storage, self.remote_storage]
        return [s for s in candidates if s is not None]

    @property
    def _results_storages(self) -> List[Storage]:
        candidates = [self.local_storage]
        return [s for s in candidates if s is not None]

    def _get_execution_key(self, database: Optional[str], sql: str) -> str:
        parts = {}
        if database is not None:
            parts["database"] = database
        parts["sql"] = sql
        encoded = urlencode(parts).encode("utf-8")
        hash = hashlib.sha256(encoded).hexdigest()
        return f"query-{hash}"

    def _get_results_key(self, execution_id: str) -> str:
        return f"results-{execution_id}"
