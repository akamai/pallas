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
Helpers for testing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pallas.info import QueryInfo
from pallas.proxies import AthenaProxy
from pallas.results import QueryResults


class FakeProxy(AthenaProxy):
    """
    Fake of proxy to AWS Athena.

    Can replace :class:`.Boto3Proxy` for testing.
    """

    column_names: Optional[Sequence[str]] = None
    column_types: Optional[Sequence[str]] = None
    state = "SUCCEEDED"
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

    def start_query_execution(
        self,
        sql: str,
        database: Optional[str] = None,
        workgroup: Optional[str] = None,
        output_location: Optional[str] = None,
    ) -> str:
        execution_id = f"query-{len(self._results) + 1}"
        self._request_log.append("StartQueryExecution")
        results = self._fake_query_results()
        self._sql[execution_id] = sql
        self._results[execution_id] = results
        return execution_id

    def get_query_execution(self, execution_id: str) -> QueryInfo:
        self._request_log.append("GetQueryExecution")
        return self._fake_query_info(execution_id, self._sql[execution_id])

    def get_query_results(self, info: QueryInfo) -> QueryResults:
        self._request_log.append("GetQueryResults")
        return self._results[info.execution_id]

    def stop_query_execution(self, execution_id: str) -> None:
        self._request_log.append("StopQueryExecution")
        self.state = "CANCELLED"

    def _fake_query_info(self, execution_id: str, sql: str) -> QueryInfo:
        data = {
            "QueryExecutionId": execution_id,
            "Query": sql,
            "ResultConfiguration": {"OutputLocation": ...},
            "QueryExecutionContext": {},
            "Status": {
                "State": self.state,
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

    def _fake_query_results(self) -> QueryResults:
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
