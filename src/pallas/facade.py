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

from typing import Optional

from pallas.base import AthenaClient
from pallas.caching import AthenaCachingWrapper
from pallas.info import QueryInfo
from pallas.interruptions import AthenaKillOnInterruptWrapper
from pallas.normalization import AthenaNormalizationWrapper
from pallas.results import QueryResults
from pallas.sql import quote
from pallas.storage import Storage


class Query:
    """
    Athena query

    Provides access to one query execution.

    Instances of this class are returned by :meth:`Athena.submit`
    and :meth:`Athena.get_query` methods.
    """

    _client: AthenaClient
    _execution_id: str

    _finished_info: Optional[QueryInfo] = None

    def __init__(self, client: AthenaClient, execution_id: str) -> None:
        self._client = client
        self._execution_id = execution_id

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
        info = self._client.get_query_info(self._execution_id)
        if info.finished:
            self._finished_info = info
        return info

    def get_results(self) -> QueryResults:
        """
        Retrieve results of this query execution.

        Waits until this query execution finishes and downloads results.
        """
        return self._client.get_query_results(self._execution_id)

    def kill(self) -> None:
        """
        Kill this query execution.
        """
        self._client.kill_query(self._execution_id)

    def join(self) -> None:
        """
        Wait until this query execution finishes.
        """
        self._client.join_query(self._execution_id)


class Athena:
    """
    Athena client
    """

    quote = staticmethod(quote)

    _client: AthenaClient

    def __init__(
        self,
        client: AthenaClient,
        storage_remote: Optional[Storage] = None,
        storage_local: Optional[Storage] = None,
        normalize: bool = False,
        kill_on_interrupt: bool = False,
    ) -> None:
        if storage_remote is not None:
            client = AthenaCachingWrapper(
                client, storage=storage_remote, cache_results=False
            )
        if storage_local is not None:
            client = AthenaCachingWrapper(
                client, storage=storage_local, cache_results=True
            )
        if normalize:
            client = AthenaNormalizationWrapper(client)
        if kill_on_interrupt:
            client = AthenaKillOnInterruptWrapper(client)
        self._client = client

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self._client!r}>"

    @property
    def client(self) -> AthenaClient:
        return self._client

    @property
    def database(self) -> Optional[str]:
        """
        Name of Athena database that will be queries.

        Individual queries can override this in SQL.
        """
        return self._client.database

    @property
    def workgroup(self) -> Optional[str]:
        """
        Name of Athena workgroup.

        Workgroup can set resource limits or override output location.
        """
        return self._client.workgroup

    @property
    def output_location(self) -> Optional[str]:
        """
        Query output location on S3.

        Can be empty if default location is configured for a workgroup.
        """
        return self._client.output_location

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
        execution_id = self._client.submit(sql, ignore_cache=ignore_cache)
        return Query(self._client, execution_id)

    def get_query(self, execution_id: str) -> Query:
        """
        Get a previously submitted query execution.

        Athena stores results in S3 and does not delete them by default.
        This method can get past queries and retrieve their results.

        :param execution_id: an Athena query execution ID.
        :return: a query instance
        """
        return Query(self._client, execution_id)

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
