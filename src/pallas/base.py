from __future__ import annotations

import time
from abc import abstractmethod, ABCMeta
from typing import (
    cast,
    overload,
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Iterable,
)

from pallas.conversions import convert_value
from pallas.waiting import Fibonacci


class AthenaQueryError(Exception):
    """Athena query failed."""

    def __init__(self, state: str, state_reason: Optional[str]):
        self.state = state
        self.state_reason = state_reason

    def __str__(self) -> str:
        if self.state_reason is not None:
            return f"Athena query {self.state.lower()}: {self.state_reason}"
        return f"Athena query {self.state.lower()}"


class Athena(metaclass=ABCMeta):
    def execute(self, sql: str) -> QueryResults:
        """Submit query execution and wait for results."""
        query = self.submit(sql)
        query.join()
        return query.get_results()

    @abstractmethod
    def submit(self, sql: str) -> Query:
        """Submit query execution."""

    @abstractmethod
    def get_query(self, execution_id: str) -> Query:
        """Get previously submitted query execution"""


class Query(metaclass=ABCMeta):

    _backoff: Iterable[int] = Fibonacci()

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
        for delay in self._backoff:
            info = self.get_info()
            if info.finished:
                info.check()
                break
            time.sleep(delay)


class QueryInfo:
    def __init__(self, data: Mapping[str, Any]):
        self._data = data

    @property
    def execution_id(self) -> str:
        return cast(str, self._data["QueryExecutionId"])

    @property
    def sql(self) -> str:
        return cast(str, self._data["Query"])

    @property
    def database(self) -> Optional[str]:
        return cast(Optional[str], self._data["QueryExecutionContext"].get("Database"))

    @property
    def finished(self) -> bool:
        return self.state in ("SUCCEEDED", "FAILED", "CANCELLED")

    @property
    def succeeded(self) -> bool:
        return self.state == "SUCCEEDED"

    @property
    def state(self) -> str:
        return cast(str, self._data["Status"]["State"])

    @property
    def state_reason(self) -> Optional[str]:
        return cast(Optional[str], self._data["Status"].get("StateChangeReason"))

    def check(self) -> None:
        if self.finished and not self.succeeded:
            raise AthenaQueryError(self.state, self.state_reason)


QueryRecord = Dict[str, Any]


class QueryResults(Sequence[QueryRecord]):

    _column_names: Tuple[str, ...]
    _column_types: Tuple[str, ...]
    _data: Sequence[Tuple[str, ...]]

    def __init__(
        self,
        column_names: Sequence[str],
        column_types: Sequence[str],
        data: Sequence[Sequence[str]],
    ) -> None:
        self._column_names = tuple(column_names)
        self._column_types = tuple(column_types)
        self._data = [tuple(row) for row in data]

    @overload
    def __getitem__(self, index: int) -> QueryRecord:
        ...

    @overload  # noqa: F811
    def __getitem__(self, index: slice) -> Sequence[QueryRecord]:
        ...

    def __getitem__(  # noqa: F811
        self, index: Union[int, slice]
    ) -> Union[QueryRecord, Sequence[QueryRecord]]:
        if isinstance(index, slice):
            raise NotImplementedError
        row = self._data[index]
        info = zip(self._column_names, self._column_types, row)
        return {cn: convert_value(ct, v) for cn, ct, v in info}

    def __len__(self) -> int:
        return len(self._data)

    @property
    def column_names(self) -> Tuple[str, ...]:
        return self._column_names

    @property
    def column_types(self) -> Tuple[str, ...]:
        return self._column_types

    @property
    def data(self) -> Sequence[Tuple[str, ...]]:
        return self._data
