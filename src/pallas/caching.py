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

from pallas.base import Athena, AthenaWrapper, Query, QueryWrapper
from pallas.results import QueryResults
from pallas.storage import NotFoundError, Storage

logger = logging.getLogger("pallas")


_comment_1 = r"--[^\n]*\n"
_comment_2 = r"/\*([^*]|\*(?!/))*\*/"

SELECT_RE = re.compile(rf"(\s+|{_comment_1}|{_comment_2})*(SELECT|WITH)\b")


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
        self, wrapped: Athena, *, storage: Storage, cache_results: bool = True
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

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        sql_cacheable = is_cacheable(sql)
        if sql_cacheable and not ignore_cache:
            execution_id = self._load_execution_id(sql)
            if execution_id is not None:
                return self.get_query(execution_id)
        query = super().submit(sql, ignore_cache=ignore_cache)
        if sql_cacheable:
            self._save_execution_id(sql, query.execution_id)
            query = self._wrap_query(query)
        return query

    def get_query(self, execution_id: str) -> Query:
        query = super().get_query(execution_id)
        return self._wrap_query(query)

    def _wrap_query(self, query: Query) -> Query:
        if self._cache_results:
            query = QueryCachingWrapper(query, self._storage)
        return query

    def _load_execution_id(self, sql: str) -> Optional[str]:
        key = self._get_cache_key(sql)
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
        key = self._get_cache_key(sql)
        self._storage.set(key, execution_id)
        logger.info(
            f"Query execution saved to cache {self._storage}{key}:"
            f" QueryExecutionId={execution_id!r}"
        )

    def _get_cache_key(self, sql: str) -> str:
        parts = {}
        if self.database is not None:
            parts["database"] = self.database
        parts["sql"] = sql
        encoded = urlencode(parts).encode("utf-8")
        hash = hashlib.sha256(encoded).hexdigest()
        return f"query-{hash}"


class QueryCachingWrapper(QueryWrapper):
    """
    Query wrapper that caches query results.
    """

    _storage: Storage

    def __init__(self, wrapped: Query, storage: Storage) -> None:
        super().__init__(wrapped)
        self._storage = storage

    def get_results(self) -> QueryResults:
        results = self._load_results()
        if results is not None:
            return results
        results = super().get_results()
        # When an old query is accessed using its execution ID,
        # we have to retrieve its SQL to determine whether it can be cached.
        # Network should not be used until a cache is checked.
        # Getting the SQL here should not produce an additional request.
        if is_cacheable(self.get_info().sql):
            self._save_results(results)
        return results

    def join(self) -> None:
        if self._has_results():
            # If we have results then we can assume that query has finished.
            return
        super().join()

    def _has_results(self) -> bool:
        return self._storage.has(self._get_cache_key())

    def _load_results(self) -> Optional[QueryResults]:
        key = self._get_cache_key()
        try:
            with self._storage.reader(key) as stream:
                results = QueryResults.load(stream)
        except NotFoundError:
            return None
        logger.info(
            f"Query results loaded from cache {self._storage}{key}:"
            f" QueryExecutionId={self.execution_id!r}: {len(results)} rows"
        )
        return results

    def _save_results(self, results: QueryResults) -> None:
        key = self._get_cache_key()
        with self._storage.writer(key) as stream:
            results.save(stream)
        logger.info(
            f"Query results saved to cache {self._storage}{key}:"
            f" QueryExecutionId={self.execution_id!r}: {len(results)} rows"
        )

    def _get_cache_key(self) -> str:
        return f"results-{self.execution_id}"
