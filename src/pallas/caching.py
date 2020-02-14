from __future__ import annotations

import csv
import hashlib
from typing import Optional

from pallas.base import Athena, Query
from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.storage import NotFoundError, Storage


class QueryCachingWrapper(Query):
    """
    Query wrapper that caches query results.
    """

    _inner_query: Query
    _storage: Storage

    def __init__(self, query: Query, storage: Storage) -> None:
        self._inner_query = query
        self._storage = storage

    @property
    def execution_id(self) -> str:
        return self._inner_query.execution_id

    def get_info(self) -> QueryInfo:
        return self._inner_query.get_info()

    def get_results(self) -> QueryResults:
        results = self._load_results()
        if results is not None:
            return results
        results = self._inner_query.get_results()
        self._save_results(results)
        return results

    def kill(self) -> None:
        return self._inner_query.kill()

    def join(self) -> None:
        if self._has_results():
            # If we have results then we can assume that query has finished.
            # Avoiding unnecessary checks allows us to work
            # completely offline when cached results are available.
            return
        super().join()

    def _has_results(self) -> bool:
        return self._storage.has(self._get_results_cache_key())

    def _load_results(self) -> Optional[QueryResults]:
        try:
            stream = self._storage.reader(self._get_results_cache_key())
        except NotFoundError:
            return None
        reader = csv.reader(stream)
        column_names = next(reader)
        column_types = next(reader)
        data = list(reader)
        return QueryResults(column_names, column_types, data)

    def _save_results(self, results: QueryResults) -> None:
        stream = self._storage.writer(self._get_results_cache_key())
        writer = csv.writer(stream)
        writer.writerow(results.column_names)
        writer.writerow(results.column_types)
        writer.writerows(results.data)

    def _get_results_cache_key(self) -> str:
        return f"results-{self.execution_id}"


class AthenaCachingWrapper(Athena):
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

    _inner_athena: Athena
    _storage: Storage

    def __init__(
        self, athena: Athena, *, storage: Storage, cache_results: bool = True
    ) -> None:
        self._inner_athena = athena
        self._storage = storage
        self._cache_results = cache_results

    def submit(self, sql: str) -> Query:
        execution_id = self._load_execution_id(sql)
        if execution_id is not None:
            return self.get_query(execution_id)
        inner_query = self._inner_athena.submit(sql)
        self._save_execution_id(sql, inner_query.execution_id)
        return self._wrap_query(inner_query)

    def get_query(self, execution_id: str) -> Query:
        inner_query = self._inner_athena.get_query(execution_id)
        return self._wrap_query(inner_query)

    def _wrap_query(self, inner_query: Query) -> Query:
        if self._cache_results:
            return QueryCachingWrapper(inner_query, self._storage)
        return inner_query

    def _load_execution_id(self, sql: str) -> Optional[str]:
        try:
            return self._storage.get(self._get_execution_cache_key(sql))
        except NotFoundError:
            return None

    def _save_execution_id(self, sql: str, execution_id: str) -> None:
        self._storage.set(self._get_execution_cache_key(sql), execution_id)

    def _get_execution_cache_key(self, sql: str) -> str:
        plain = sql
        hash = hashlib.sha256(plain.encode("utf-8")).hexdigest()
        return f"query-{hash}"
