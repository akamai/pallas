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
Interfaces and base classes for querying AWS Athena.

Pallas exposes to main interfaces:
 - :class:`.Athena` for submitting queries.
 - :class:`.Query` for non-blocking monitoring of query status.

Classes :class:`.AthenaProxy` and :class`.QueryProxy`
are the core implementations of the interfaces.
They can be optionally decorated by wrappers that provide
extra functionality (like caching) preserving the same interface.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional

from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.sql import quote


class Athena(metaclass=ABCMeta):
    """
    Athena interface

    Provides methods to execute SQL queries in AWS Athena.

    This is an abstract base class.
    The :class:`.AthenaProxy` subclass submits actual queries to AWS API.
    Other subclasses are implemented as decorators adding extra
    functionally (for example caching).
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    quote = staticmethod(quote)

    @property
    @abstractmethod
    def database(self) -> Optional[str]:
        """
        Name of Athena database that will be queries.

        Individual queries can override this in SQL.
        """

    @property
    @abstractmethod
    def workgroup(self) -> Optional[str]:
        """
        Name of Athena workgroup.

        Workgroup can set resource limits or override output location.
        """

    @property
    def output_location(self) -> Optional[str]:
        """
        Query output location on S3.

        Can be empty if default location is configured for a workgroup.
        """

    def execute(self, sql: str, *, ignore_cache: bool = False) -> QueryResults:
        """
        Execute a query and wait for results.

        This is a blocking method that waits until query finishes.
        Returns :class:`QueryResults`.

        :param sql: SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: query results
        """
        # Do not override this method.
        # Wrappers do not call execute on the wrapped instance,
        # so overrides are likely to have no effect.
        return self.submit(sql, ignore_cache=ignore_cache).get_results()

    @abstractmethod
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

    @abstractmethod
    def get_query(self, execution_id: str) -> Query:
        """
        Get a previously submitted query execution.

        Athena stores results in S3 and does not delete them by default.
        This method can get past queries and retrieve their results.

        :param execution_id: an Athena query execution ID.
        :return: a query instance
        """


class Query(metaclass=ABCMeta):
    """
    Query interface

    Provides access to one query execution.

    Instances of this class are returned by :meth:`Athena.submit`
    and :meth:`Athena.get_query` methods.
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: execution_id={self.execution_id!r}>"

    @property
    @abstractmethod
    def execution_id(self) -> str:
        """
        Athena query execution ID.

        Returns a unique ID of this query execution.
        This ID can be used to retrieve the query using
        the :meth:`.Athena.get_query()` method.
        """

    @abstractmethod
    def get_info(self) -> QueryInfo:
        """
        Retrieve information about this query execution.

        Returns a status of this query with other information.
        """

    @abstractmethod
    def get_results(self) -> QueryResults:
        """
        Retrieve results of this query execution.

        Waits until this query execution finishes and downloads results.
        """

    @abstractmethod
    def kill(self) -> None:
        """
        Kill this query execution.
        """

    @abstractmethod
    def join(self) -> None:
        """
        Wait until this query execution finishes.
        """


class AthenaWrapper(Athena):
    """
    Base class for :class:`.Athena` decorators.
    """

    _wrapped: Athena

    def __init__(self, wrapped: Athena) -> None:
        self._wrapped = wrapped

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.wrapped!r}>"

    @property
    def wrapped(self) -> Athena:
        return self._wrapped

    @property
    def database(self) -> Optional[str]:
        return self._wrapped.database

    @property
    def workgroup(self) -> Optional[str]:
        return self._wrapped.workgroup

    @property
    def output_location(self) -> Optional[str]:
        return self._wrapped.output_location

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        return self._wrapped.submit(sql, ignore_cache=ignore_cache)

    def get_query(self, execution_id: str) -> Query:
        return self._wrapped.get_query(execution_id)


class QueryWrapper(Query):
    """
    Base class for :class:`.Query` decorators.
    """

    _wrapped: Query

    def __init__(self, wrapped: Query) -> None:
        self._wrapped = wrapped

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.wrapped!r}>"

    @property
    def wrapped(self) -> Query:
        return self._wrapped

    @property
    def execution_id(self) -> str:
        return self._wrapped.execution_id

    def get_info(self) -> QueryInfo:
        return self._wrapped.get_info()

    def get_results(self) -> QueryResults:
        return self._wrapped.get_results()

    def kill(self) -> None:
        self._wrapped.kill()

    def join(self) -> None:
        self._wrapped.join()
