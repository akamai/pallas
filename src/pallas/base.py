from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from typing import Optional

from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.waiting import Fibonacci


class Athena(metaclass=ABCMeta):
    """
    Athena interface
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    @property
    @abstractmethod
    def database(self) -> Optional[str]:
        """Name of Athena database"""

    def execute(self, sql: str, *, ignore_cache: bool = False) -> QueryResults:
        """
        Submit query execution and wait for results.

        This is a blocking method that waits until query finishes
        and results are downloaded.

        :param sql: SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: query results
        """
        query = self.submit(sql, ignore_cache=ignore_cache)
        query.join()
        return query.get_results()

    @abstractmethod
    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        """
        Submit query execution.

        This is a non-blocking method that start a query execution
        and returns.

        :param sql: an SQL query to be executed
        :param ignore_cache: do not load cached results
        :return: a query instance
        """

    @abstractmethod
    def get_query(self, execution_id: str) -> Query:
        """
        Get a previously submitted query execution.
        """


class Query(metaclass=ABCMeta):
    """
    Query interface
    """

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: execution_id={self.execution_id!r}>"

    @property
    @abstractmethod
    def execution_id(self) -> str:
        """Athena query execution ID."""

    @abstractmethod
    def get_info(self) -> QueryInfo:
        """Retrieve information about this query execution."""

    @abstractmethod
    def get_results(self) -> QueryResults:
        """Retrieve results of this query execution."""

    @abstractmethod
    def kill(self) -> None:
        """Kill this query execution."""

    def join(self) -> None:
        """Wait until this query execution finishes."""
        for delay in Fibonacci(max_value=60):
            info = self.get_info()
            if info.finished:
                info.check()
                break
            time.sleep(delay)
