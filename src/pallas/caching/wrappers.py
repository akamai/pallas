import csv
import hashlib
from typing import Optional

from pallas.base import Athena, Query, QueryInfo, QueryResults
from pallas.caching.backends import Cache, CacheMiss


class QueryCachingWrapper(Query):

    _inner_query: Query
    _cache: Cache

    def __init__(self, query: Query, cache: Cache):
        self._inner_query = query
        self._cache = cache

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

    def _load_results(self) -> Optional[QueryResults]:
        try:
            stream = self._cache.reader(self._get_results_cache_key())
        except CacheMiss:
            return None
        reader = csv.reader(stream)
        column_names = next(reader)
        column_types = next(reader)
        data = list(reader)
        return QueryResults(column_names, column_types, data)

    def _save_results(self, results: QueryResults) -> None:
        stream = self._cache.writer(self._get_results_cache_key())
        writer = csv.writer(stream)
        writer.writerow(results.column_names)
        writer.writerow(results.column_types)
        writer.writerows(results.data)

    def _get_results_cache_key(self) -> str:
        return f"results-{self.execution_id}"


class AthenaCachingWrapper(Athena):

    _inner_athena: Athena
    _cache: Cache

    def __init__(self, athena: Athena, *, cache: Cache) -> None:
        self._inner_athena = athena
        self._cache = cache

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
        return QueryCachingWrapper(inner_query, self._cache)

    def _load_execution_id(self, sql: str) -> Optional[str]:
        try:
            return self._cache.get(self._get_execution_cache_key(sql))
        except CacheMiss:
            return None

    def _save_execution_id(self, sql: str, execution_id: str) -> None:
        self._cache.set(self._get_execution_cache_key(sql), execution_id)

    def _get_execution_cache_key(self, sql: str) -> str:
        plain = sql
        hash = hashlib.sha256(plain.encode("utf-8")).hexdigest()
        return f"query-{hash}"
