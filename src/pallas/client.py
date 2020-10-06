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

import copy
import time
from typing import Iterable, Optional

from pallas.caching import AthenaCache
from pallas.info import QueryInfo
from pallas.proxies import AthenaProxy
from pallas.results import QueryResults
from pallas.sql import (
    PARAMETERS,
    is_select,
    normalize_sql,
    quote,
    substitute_parameters,
)
from pallas.utils import Fibonacci


class Query:
    """
    Athena query

    Provides access to one query execution.
    It can be used to monitor status of the query results
    or retrieving results when the execution finishes.

    Instances of this class are returned by :meth:`Athena.submit`
    and :meth:`Athena.get_query` methods.
    You should not need to create this class directly.

    :param execution_id: Athena query execution ID.
    :param proxy: an internal proxy to execute queries
    :param cache: a cache instance
    """

    #: Delays in seconds between for checking query status.
    backoff: Iterable[int] = Fibonacci(max_value=60)

    #: Whether to kill this query on KeyboardInterrupt
    #:
    #: Initially set to :attr:`Athena.kill_on_interrupt`.
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

        This ID can be used to retrieve this query later using
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

        Cached results can be returned, if the caching was configured.
        Only SELECT queries are cached.

        Waits until this query execution finishes and downloads results.
        Raises :class:`.AthenaQueryError` if the query failed.
        """
        # When a user calls athena.get_query(execution_id).get_results(),
        # we have to look into the cache without knowing what SQL was executed,
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

        This is a non-blocking operation.
        It does not wait until the query is killed.
        """
        self._proxy.stop_query_execution(self._execution_id)

    def join(self) -> None:
        """
        Wait until this query execution finishes.

        Raises :class:`.AthenaQueryError` if the query failed.
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

    Use :func:`.setup` or :func:`.environ_setup` to construct
    this class without touching Pallas internals.

    :param proxy: an internal proxy to execute queries
    """

    quote = staticmethod(quote)

    #: Name of Athena database to be be queried.
    #:
    #: Can be overridden in SQL.
    database: Optional[str] = None

    #: Name of Athena workgroup.
    #:
    #: Workgroup can set resource limits or override output location.
    #: When None, defaults to the Athena default workgroup.
    workgroup: Optional[str] = None

    #: URI of output location on S3.
    #:
    #: Optional if an output location is specified for :attr:`.workgroup`.
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
        """
        Cache implementation.

        It is possible to update properties of the :attr:`.cache`
        attribute to reconfigure caching in place.

        Alternatively, the :meth:`.using` method can apply
        a new configuration without affecting an existing instance.

        :rtype: :class:`.AthenaCache`
        """
        return self._cache

    def using(
        self,
        *,
        database: Optional[str] = None,
        workgroup: Optional[str] = None,
        output_location: Optional[str] = None,
        normalize: Optional[bool] = None,
        kill_on_interrupt: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
        cache_read: Optional[bool] = None,
        cache_write: Optional[bool] = None,
    ) -> Athena:
        """
        Crate a new instance with an updated configuration.

        This method can be useful if you need to override a configuration
        for one query, but you do not want to affect future queries.

        :param database: name of Athena database to be be queried.
        :param workgroup: name of Athena workgroup.
        :param output_location: URI of output location on S3.
        :param normalize: whether to normalize queries before execution.
        :param kill_on_interrupt: whether to kill queries on KeyboardInterrupt
        :param cache_enabled: whether a cache should be used.
        :param cache_read: whether a cache should be read.
        :param cache_write: whether a cache should be written.
        :return: an updated copy of this client
        """
        other = copy.copy(self)
        other._cache = copy.copy(self._cache)
        if database is not None:
            other.database = database
        if workgroup is not None:
            other.workgroup = workgroup
        if output_location is not None:
            other.output_location = output_location
        if normalize is not None:
            other.normalize = normalize
        if kill_on_interrupt is not None:
            other.kill_on_interrupt = kill_on_interrupt
        if cache_enabled is not None:
            other._cache.enabled = cache_enabled
        if cache_read is not None:
            other._cache.read = cache_read
        if cache_write is not None:
            other._cache.write = cache_write
        return other

    def execute(
        self,
        operation: str,
        parameters: PARAMETERS = None,
    ) -> QueryResults:
        """
        Execute a query and return results.

        This is a blocking method that waits until the query finishes.

        Cached results or results from an existing query can be returned,
        if the caching was configured. Only SELECT queries are cached.

        Raises :class:`.AthenaQueryError` if the query fails.

        :param operation: an SQL query to be executed
            Can contain ``%s`` or ``%(key)s`` placeholders for substitution
            by *parameters*.
        :param parameters: parameters to substitute in *operation*.
            All substitute parameters are quoted appropriately.
            See the :meth:`.quote` method for a supported parameter types.
        :type parameters: Union[None, Tuple[SQL_SCALAR, ...], Mapping[str, SQL_SCALAR]]
        :return: query results
        """
        return self.submit(operation, parameters).get_results()

    def submit(
        self,
        operation: str,
        parameters: PARAMETERS = None,
    ) -> Query:
        """
        Submit a query and return.

        This is a non-blocking method that starts a query and returns.
        Returns a :class:`Query` instance for monitoring query execution
        and downloading results later.

        An existing query can be returned, if the caching was configured.
        Only SELECT queries are cached.

        :param operation: an SQL query to be executed
            Can contain ``%s`` or ``%(key)s`` placeholders for substitution
            by *parameters*.
        :param parameters: parameters to substitute in *operation*.
            All substitute parameters are quoted appropriately.
            See the :meth:`.quote` method for a supported parameter types.
        :type parameters: Union[None, Tuple[SQL_SCALAR, ...], Mapping[str, SQL_SCALAR]]
        :return: a query instance
        """
        sql = self._get_sql(operation, parameters)
        should_cache = is_select(sql)
        if should_cache:
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

        This method can be used to retrieve a query executed in the past.
        Because Athena stores results in S3 and does not delete them by default,
        it is possible to download results until they are manually deleted.

        :param execution_id: an Athena query execution ID.
        :return: a query instance
        """
        query = Query(execution_id, proxy=self._proxy, cache=self._cache)
        query.kill_on_interrupt = self.kill_on_interrupt
        return query

    def _get_sql(self, operation: str, parameters: PARAMETERS) -> str:
        sql = substitute_parameters(operation, parameters)
        if self.normalize:
            sql = normalize_sql(sql)
        return sql
