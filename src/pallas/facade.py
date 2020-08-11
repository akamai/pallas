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

from pallas.caching import AthenaCache
from pallas.info import QueryInfo
from pallas.proxies import AthenaProxy
from pallas.results import QueryResults
from pallas.sql import is_select, normalize_sql, quote
from pallas.utils import Fibonacci


class Query:
    """
    Athena query

    Provides access to one query execution.
    It can be used to monitor status of the query results
    or retrieving results when the execution finishes.

    Instances of this class are returned by :meth:`Athena.submit`
    and :meth:`Athena.get_query` methods.
    """

    #: Delays in seconds between for checking query status.
    backoff: Iterable[int] = Fibonacci(max_value=60)

    #: Whether to kill queries on KeyboardInterrupt
    kill_on_interrupt: bool = False

    _execution_id: str
    _proxy: AthenaProxy
    _cache: AthenaCache

    _info: Optional[QueryInfo] = None

    def __init__(
        self, execution_id: str, *, proxy: AthenaProxy, cache: AthenaCache,
    ) -> None:
        self._execution_id = execution_id
        self._proxy = proxy
        self._cache = cache

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: execution_id={self.execution_id!r}>"

    @property
    def execution_id(self) -> str:
        """
        Athena query execution ID.

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
        if self._info is not None:
            return self._info
        info = self._proxy.get_query_execution(self._execution_id)
        if info.finished:
            self._info = info
        return info

    def get_results(self) -> QueryResults:
        """
        Download results of this query execution.

        Waits until this query execution finishes and downloads results.
        """
        # When a user calls athena.get_query(execution_id).get_results(),
        # we have to look into the cache withou knowing what SQL was executed,
        # so whether the query is cacheable.
        results = self._cache.load_results(self._execution_id)
        if results is not None:
            return results
        self.join()
        info = self.get_info()
        results = self._proxy.get_query_results(info)
        should_cache = is_select(info.sql)
        if should_cache:
            self._cache.save_results(self._execution_id, results)
        return results

    def kill(self) -> None:
        """
        Kill this query execution.
        """
        self._proxy.stop_query_execution(self._execution_id)

    def join(self) -> None:
        """
        Wait until this query execution finishes.
        """
        # When we have results locally, exit without calling any AWS API.
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
                # Catch only the first KeyboardInterrupt
                self.kill_on_interrupt = False
                self.kill()
                self.join()


class Athena:
    """
    Athena client.

    Provides methods to execute SQL queries in AWS Athena,
    with an optional caching and other helpers.

    Can be used as a blocking or a non-blocking client.
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

    _proxy: AthenaProxy
    _cache: AthenaCache

    def __init__(self, proxy: AthenaProxy) -> None:
        self._proxy = proxy
        self._cache = AthenaCache()

    def __repr__(self) -> str:
        parts = [
            f"database={self.database!r}",
            f"workgroup={self.workgroup!r}",
            f"output_location={self.output_location!r}",
        ]
        return f"<{type(self).__name__}: {', '.join(parts)}>"

    @property
    def proxy(self) -> AthenaProxy:
        return self._proxy

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
        should_cache = is_select(sql)
        if should_cache and not ignore_cache:
            execution_id = self._cache.load_execution_id(self.database, sql)
            if execution_id is not None:
                return self.get_query(execution_id)
        execution_id = self._proxy.start_query_execution(
            sql,
            database=self.database,
            workgroup=self.workgroup,
            output_location=self.output_location,
        )
        if should_cache:
            self._cache.save_execution_id(self.database, sql, execution_id)
        return self.get_query(execution_id)

    def get_query(self, execution_id: str) -> Query:
        """
        Get a previously submitted query execution.

        Athena stores results in S3 and does not delete them by default.
        This method can get past queries to retrieve their results.

        :param execution_id: an Athena query execution ID.
        :return: a query instance
        """
        query = Query(execution_id, proxy=self._proxy, cache=self._cache)
        query.kill_on_interrupt = self.kill_on_interrupt
        return query

    def execute(self, sql: str, *, ignore_cache: bool = False) -> QueryResults:
        """
        Execute a query and wait for results.

        This is a blocking method that waits until query finishes.

        :param sql: SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: query results
        """
        return self.submit(sql, ignore_cache=ignore_cache).get_results()
