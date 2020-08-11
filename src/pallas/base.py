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

    @abstractmethod
    def start_query_execution(
        self,
        sql: str,
        *,
        database: Optional[str] = None,
        workgroup: Optional[str] = None,
        output_location: Optional[str] = None,
    ) -> str:
        """
        Submit a query.

        :param sql: an SQL query to be executed
        :param database: a name of Athena database to be queried
        :param workgroup: a name of Athena workgroup
        :param output_location: URI of output location on S3
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
