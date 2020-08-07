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


class AthenaClient(metaclass=ABCMeta):
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

    @abstractmethod
    def start_query_execution(self, sql: str, *, ignore_cache: bool = False) -> str:
        """
        Submit a query.

        :param sql: an SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: execution_id
        """

    @abstractmethod
    def get_query_execution(self, execution_id: str) -> QueryInfo:
        """
        Retrieve information about a query execution.

        Returns a status of the query with other information.
        """

    @abstractmethod
    def get_query_results(self, execution_id: str) -> QueryResults:
        """
        Retrieve results of a query execution.

        Waits until the query execution finishes and downloads results.
        """

    @abstractmethod
    def stop_query_execution(self, execution_id: str) -> None:
        """
        Kill a query execution.
        """

    @abstractmethod
    def join_query_execution(self, execution_id: str) -> None:
        """
        Wait until a query execution finishes.
        """


class AthenaWrapper(AthenaClient):
    """
    Base class for :class:`.Athena` decorators.
    """

    _wrapped: AthenaClient

    def __init__(self, wrapped: AthenaClient) -> None:
        self._wrapped = wrapped

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.wrapped!r}>"

    @property
    def wrapped(self) -> AthenaClient:
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

    def start_query_execution(self, sql: str, *, ignore_cache: bool = False) -> str:
        return self._wrapped.start_query_execution(sql, ignore_cache=ignore_cache)

    def get_query_execution(self, execution_id: str) -> QueryInfo:
        return self._wrapped.get_query_execution(execution_id)

    def get_query_results(self, execution_id: str) -> QueryResults:
        return self._wrapped.get_query_results(execution_id)

    def stop_query_execution(self, execution_id: str) -> None:
        self._wrapped.stop_query_execution(execution_id)

    def join_query_execution(self, execution_id: str) -> None:
        self._wrapped.join_query_execution(execution_id)
