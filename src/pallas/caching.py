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
Decorators for caching Athena queries.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional
from urllib.parse import urlencode

from pallas.base import AthenaClient, AthenaWrapper
from pallas.results import QueryResults
from pallas.storage import NotFoundError, Storage

logger = logging.getLogger("pallas")


_comment_1 = r"--[^\n]*\n"
_comment_2 = r"/\*([^*]|\*(?!/))*\*/"

SELECT_RE = re.compile(
    rf"(\s+|{_comment_1}|{_comment_2})*(SELECT|WITH)\b", re.IGNORECASE
)


def is_cacheable(sql: str) -> bool:
    """
    Return whether an SQL statement can be cached.

    Only SELECT statements are considered cacheable.
    """
    return SELECT_RE.match(sql) is not None


class AthenaCachingWrapper(AthenaWrapper):
    """
    Athena wrapper that caches query IDs and optionally results.

    Athena always stores results in S3, so it is possible
    to retrieve past results without manually copying data.

    The caching wrapper can work two modes,
    depending on *cache_results* setting:

    1) Remote mode - cache query execution IDs.
    2) Local mode - Cache query execution IDs and query results.

    I the remote mode, cache is used to recover query execution IDs
    that are necessary for downloading results of past queries.

    In the local mode, results are stored to cache too.
    If the cache storage is local then it is possible to retrieve
    cached results without internet connection.
    """

    _storage: Storage

    def __init__(
        self, wrapped: AthenaClient, *, storage: Storage, cache_results: bool = True
    ) -> None:
        super().__init__(wrapped)
        self._storage = storage
        self._cache_results = cache_results

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}: {self.wrapped!r} cached at {self.storage.uri!r}>"
        )

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def cache_results(self) -> bool:
        return self._cache_results

    def start_query_execution(self, sql: str, *, ignore_cache: bool = False) -> str:
        sql_cacheable = is_cacheable(sql)
        if sql_cacheable and not ignore_cache:
            execution_id = self._load_execution_id(sql)
            if execution_id is not None:
                return execution_id
        execution_id = super().start_query_execution(sql, ignore_cache=ignore_cache)
        if sql_cacheable:
            self._save_execution_id(sql, execution_id)
        return execution_id

    def get_query_results(self, execution_id: str) -> QueryResults:
        if self._cache_results:
            results = self._load_results(execution_id)
            if results is not None:
                return results
        results = super().get_query_results(execution_id)
        if self._cache_results:
            # TODO: save only cachable queries
            self._save_results(execution_id, results)
        return results

    def _load_execution_id(self, sql: str) -> Optional[str]:
        key = self._get_execution_id_key(sql)
        try:
            execution_id = self._storage.get(key)
        except NotFoundError:
            return None
        logger.info(
            f"Query execution loaded from cache {self._storage}{key}:"
            f" QueryExecutionId={execution_id!r}"
        )
        return execution_id

    def _save_execution_id(self, sql: str, execution_id: str) -> None:
        key = self._get_execution_id_key(sql)
        self._storage.set(key, execution_id)
        logger.info(
            f"Query execution saved to cache {self._storage}{key}:"
            f" QueryExecutionId={execution_id!r}"
        )

    def _has_results(self, execution_id: str) -> bool:
        return self._storage.has(self._get_results_key(execution_id))

    def _load_results(self, execution_id: str) -> Optional[QueryResults]:
        key = self._get_results_key(execution_id)
        try:
            with self._storage.reader(key) as stream:
                results = QueryResults.load(stream)
        except NotFoundError:
            return None
        logger.info(
            f"Query results loaded from cache {self._storage}{key}:"
            f" QueryExecutionId={execution_id!r}: {len(results)} rows"
        )
        return results

    def _save_results(self, execution_id: str, results: QueryResults) -> None:
        key = self._get_results_key(execution_id)
        with self._storage.writer(key) as stream:
            results.save(stream)
        logger.info(
            f"Query results saved to cache {self._storage}{key}:"
            f" QueryExecutionId={execution_id!r}: {len(results)} rows"
        )

    def _get_execution_id_key(self, sql: str) -> str:
        parts = {}
        if self.database is not None:
            parts["database"] = self.database
        parts["sql"] = sql
        encoded = urlencode(parts).encode("utf-8")
        hash = hashlib.sha256(encoded).hexdigest()
        return f"query-{hash}"

    def _get_results_key(self, execution_id: str) -> str:
        return f"results-{execution_id}"
