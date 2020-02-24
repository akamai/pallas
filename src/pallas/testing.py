from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pallas.base import Athena, Query
from pallas.info import QueryInfo
from pallas.results import QueryResults


class AthenaFake(Athena):
    """Fake Athena implementation that can be used for testing."""

    database: Optional[str] = None
    workgroup: Optional[str] = None
    output_location: Optional[str] = None

    column_names: Optional[Sequence[str]] = None
    column_types: Optional[Sequence[str]] = None
    data: Optional[Sequence[Sequence[str]]] = None

    _queries: Dict[str, Query]
    _request_log: List[str]

    def __init__(self) -> None:
        self._queries = {}
        self._request_log = []

    @property
    def request_log(self) -> List[str]:
        return self._request_log

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        execution_id = f"query-{len(self._queries)+1}"
        self._request_log.append("StartQueryExecution")
        info = self._get_info(execution_id, sql)
        results = self._get_results()
        query = QueryFake(info, results, self._request_log)
        self._queries[execution_id] = query
        return query

    def get_query(self, execution_id: str) -> Query:
        return self._queries[execution_id]

    def _get_info(self, execution_id: str, sql: str) -> QueryInfo:
        data = {
            "QueryExecutionId": execution_id,
            "Query": sql,
            "ResultConfiguration": {"OutputLocation": ...},
            "QueryExecutionContext": {},
            "Status": {
                "State": "SUCCEEDED",
                "SubmissionDateTime": ...,
                "CompletionDateTime": ...,
            },
            "Statistics": {
                "EngineExecutionTimeInMillis": 0,
                "DataScannedInBytes": 0,
                "TotalExecutionTimeInMillis": 0,
                "QueryQueueTimeInMillis": 0,
                "ServiceProcessingTimeInMillis": 0,
            },
            "WorkGroup": "primary",
        }
        return QueryInfo(data)

    def _get_results(self) -> QueryResults:
        column_names = self.column_names
        column_types = self.column_types
        data = self.data
        if column_names is None:
            column_names = []
        if column_types is None:
            column_types = ["unknown" for _ in column_names]
        if data is None:
            data = []
        return QueryResults(column_names, column_types, data)


class QueryFake(Query):
    """Fake Query implementation that can be used for testing."""

    _info: QueryInfo
    _results: QueryResults
    _request_log: List[str]

    def __init__(
        self, info: QueryInfo, results: QueryResults, request_log: List[str]
    ) -> None:
        self._info = info
        self._results = results
        self._request_log = request_log

    @property
    def execution_id(self) -> str:
        return self._info.execution_id

    def get_info(self) -> QueryInfo:
        self._request_log.append("GetQueryExecution")
        return self._info

    def get_results(self) -> QueryResults:
        self.join()
        self._request_log.append("GetQueryResults")
        return self._results

    def kill(self) -> None:
        self._request_log.append("StopQueryExecution")

    def join(self) -> None:
        self.get_info().check()
