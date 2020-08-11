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

import time
from typing import Iterable, Optional

from pallas.base import AthenaClient
from pallas.caching import AthenaCache
from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.sql import is_select, normalize_sql, quote
from pallas.utils import Fibonacci


class Query:
    """
    Athena query

    Provides access to one query execution.

    Instances of this class are returned by :meth:`Athena.submit`
    and :meth:`Athena.get_query` methods.
    """

    _execution_id: str
    _client: AthenaClient
    _cache: AthenaCache

    #: Whether to kill queries on KeyboardInterrupt
    kill_on_interrupt: bool = False

    backoff: Iterable[int] = Fibonacci(max_value=60)

    _finished_info: Optional[QueryInfo] = None

    def __init__(
        self, execution_id: str, *, client: AthenaClient, cache: AthenaCache,
    ) -> None:
        self._execution_id = execution_id
        self._client = client
        self._cache = cache

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: execution_id={self.execution_id!r}>"

    @property
    def execution_id(self) -> str:
        """
        Athena query execution ID.

        Returns a unique ID of this query execution.
        This ID can be used to retrieve the query using
        the :meth:`.Athena.get_query()` method.
        """
        return self._execution_id

    def get_info(self) -> QueryInfo:
        """
        Retrieve information about this query execution.

        Returns a status of this query with other information.
        """
        # Query info is cached if the query finished and cannot change.
        if self._finished_info is not None:
            return self._finished_info
        info = self._client.get_query_execution(self._execution_id)
        if info.finished:
            self._finished_info = info
        return info

    def get_results(self) -> QueryResults:
        """
        Retrieve results of this query execution.

        Waits until this query execution finishes and downloads results.
        """
        results = self._cache.load_results(self._execution_id)
        if results is not None:
            return results
        self.join()
        results = self._client.get_query_results(self._execution_id)
        use_cache = is_select(self.get_info().sql)
        if use_cache:
            self._cache.save_results(self._execution_id, results)
        return results

    def kill(self) -> None:
        """
        Kill this query execution.
        """
        self._client.stop_query_execution(self._execution_id)

    def join(self) -> None:
        """
        Wait until this query execution finishes.
        """
        if self._cache.has_results(self._execution_id):
            return
        for delay in self.backoff:
            info = self.get_info()
            if info.finished:
                info.check()
                break
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                if not self.kill_on_interrupt:
                    raise
                self.kill()
                # Catch only the first KeyboardInterrupt
                self.kill_on_interrupt = False
                self.join()


class Athena:
    """
    Athena client
    """

    quote = staticmethod(quote)

    #: Name of Athena database to be be queried.
    #:
    #: Individual queries can override this in SQL.
    database: Optional[str] = None

    #: Name of Athena workgroup.
    #:
    #: Workgroup can set resource limits or override output location.
    workgroup: Optional[str] = None

    #: URI of output location on S3.
    #:
    #: Can be empty if default location is configured for a workgroup.
    output_location: Optional[str] = None

    #: Whether to normalize queries before execution.
    normalize: bool = True

    #: Whether to kill queries on KeyboardInterrupt
    kill_on_interrupt: bool = True

    _client: AthenaClient
    _cache: AthenaCache

    def __init__(self, client: AthenaClient) -> None:
        self._client = client
        self._cache = AthenaCache()

    def __repr__(self) -> str:
        parts = []
        if self.database is not None:
            parts.append(f"database={self.database!r}")
        if self.workgroup is not None:
            parts.append(f"workgroup={self.workgroup!r}")
        if self.output_location is not None:
            parts.append(f"output_location={self.output_location!r}")
        return f"<{type(self).__name__}: {', '.join(parts)}>"

    @property
    def client(self) -> AthenaClient:
        return self._client

    @property
    def cache(self) -> AthenaCache:
        return self._cache

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        """
        Submit a query and return.

        This is a non-blocking method that starts a query and returns.
        Returns a :class:`Query` instance for monitoring query execution
        and downloading results later.

        :param sql: an SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: a query instance
        """
        if self.normalize:
            sql = normalize_sql(sql)
        use_cache = is_select(sql)
        if use_cache and not ignore_cache:
            execution_id = self._cache.load_execution_id(self.database, sql)
            if execution_id is not None:
                return self.get_query(execution_id)
        execution_id = self._client.start_query_execution(sql)
        if use_cache:
            self._cache.save_execution_id(self.database, sql, execution_id)
        return self.get_query(execution_id)

    def get_query(self, execution_id: str) -> Query:
        """
        Get a previously submitted query execution.

        Athena stores results in S3 and does not delete them by default.
        This method can get past queries and retrieve their results.

        :param execution_id: an Athena query execution ID.
        :return: a query instance
        """
        query = Query(execution_id, client=self._client, cache=self._cache)
        query.kill_on_interrupt = self.kill_on_interrupt
        return query

    def execute(self, sql: str, *, ignore_cache: bool = False) -> QueryResults:
        """
        Execute a query and wait for results.

        This is a blocking method that waits until query finishes.
        Returns :class:`QueryResults`.

        :param sql: SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: query results
        """
        return self.submit(sql, ignore_cache=ignore_cache).get_results()
