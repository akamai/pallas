from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from typing import Optional

from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.waiting import Fibonacci


class Athena(metaclass=ABCMeta):
    """Interface for querying Athena."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def execute(self, sql: str) -> QueryResults:
        """Submit query execution and wait for results."""
        query = self.submit(sql)
        query.join()
        return query.get_results()

    @property
    @abstractmethod
    def database(self) -> Optional[str]:
        """Name of Athena database"""

    @abstractmethod
    def submit(self, sql: str) -> Query:
        """Submit query execution."""

    @abstractmethod
    def get_query(self, execution_id: str) -> Query:
        """Get previously submitted query execution"""


class Query(metaclass=ABCMeta):
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
