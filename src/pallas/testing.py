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
Fake Athena client for testing purposes.
"""

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

    _sql: Dict[str, str]
    _results: Dict[str, QueryResults]
    _request_log: List[str]

    def __init__(self) -> None:
        self._sql = {}
        self._results = {}
        self._request_log = []

    @property
    def request_log(self) -> List[str]:
        return self._request_log

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        execution_id = f"query-{len(self._results) + 1}"
        self._request_log.append("StartQueryExecution")
        results = self._get_results()
        self._sql[execution_id] = sql
        self._results[execution_id] = results
        return self.get_query(execution_id)

    def get_query(self, execution_id: str) -> Query:
        info = self._get_info(execution_id, self._sql[execution_id])
        results = self._results[execution_id]
        return QueryFake(info, results, self._request_log)

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

    _finished_info: Optional[QueryInfo] = None

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
        # Mimic behavior of QueryProxy - remember info of finished queries.
        if self._finished_info is not None:
            return self._finished_info
        self._request_log.append("GetQueryExecution")
        if self._info.finished:
            self._finished_info = self._info
        return self._info

    def get_results(self) -> QueryResults:
        self.join()
        self._request_log.append("GetQueryResults")
        return self._results

    def kill(self) -> None:
        self._request_log.append("StopQueryExecution")

    def join(self) -> None:
        self.get_info().check()
