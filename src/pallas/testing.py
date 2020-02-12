from typing import Dict, Optional, Sequence

from pallas import QueryResults, QueryInfo
from pallas.base import Athena, Query


class QueryFake(Query):

    _info: QueryInfo
    _results: QueryResults

    def __init__(self, info: QueryInfo, results: QueryResults) -> None:
        self._info = info
        self._results = results

    @property
    def execution_id(self) -> str:
        return self._info.execution_id

    def get_info(self) -> QueryInfo:
        return self._info

    def get_results(self) -> QueryResults:
        return self._results

    def kill(self) -> None:
        raise NotImplementedError


class AthenaFake(Athena):

    column_names: Optional[Sequence[str]] = None
    column_types: Optional[Sequence[str]] = None
    data: Optional[Sequence[Sequence[str]]] = None

    _queries: Dict[str, Query]

    def __init__(self) -> None:
        self._queries = {}

    @property
    def queries(self) -> Sequence[Query]:
        return list(self._queries.values())

    def submit(self, sql: str) -> Query:
        execution_id = f"query-{len(self._queries)}"
        info = self._get_info(execution_id, sql)
        results = self._get_results()
        query = QueryFake(info, results)
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

    def _get_results(self):
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
