from __future__ import annotations

import itertools
from typing import Any, List, Optional, Tuple

import boto3

from pallas.base import Athena, Query
from pallas.info import QueryInfo
from pallas.results import QueryResults


class AthenaProxy(Athena):

    _client: Any  # boto3 Athena client
    _output_location: str
    _database: Optional[str]

    def __init__(
        self,
        *,
        output_location: str,
        client: Optional[Any] = None,
        database: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        if client is None:
            client = boto3.client("athena", region_name=region_name)
        self._client = client
        self._output_location = output_location
        self._database = database

    @property
    def database(self) -> Optional[str]:
        return self._database

    def submit(self, sql: str) -> Query:
        params = dict(
            QueryString=sql,
            ResultConfiguration={"OutputLocation": self._output_location},
        )
        if self._database is not None:
            params.update(QueryExecutionContext={"Database": self._database})
        response = self._client.start_query_execution(**params)
        return self.get_query(response["QueryExecutionId"])

    def get_query(self, execution_id: str) -> Query:
        return QueryProxy(self._client, execution_id)


class QueryProxy(Query):

    _client: Any  # boto3 Athena client
    _execution_id: str
    _finished_info: Optional[QueryInfo] = None

    def __init__(self, client: Any, execution_id: str) -> None:
        self._client = client
        self._execution_id = execution_id

    @property
    def execution_id(self) -> str:
        return self._execution_id

    def get_info(self) -> QueryInfo:
        # Query info is cached if the query finished so it cannot change.
        if self._finished_info is not None:
            return self._finished_info
        response = self._client.get_query_execution(QueryExecutionId=self.execution_id)
        info = QueryInfo(response["QueryExecution"])
        if info.finished:
            self._finished_info = info
        return info

    def kill(self) -> None:
        self._client.stop_query_execution(QueryExecutionId=self.execution_id)

    def get_results(self) -> QueryResults:
        params = dict(QueryExecutionId=self.execution_id)
        paginator = self._client.get_paginator("get_query_results")
        pages = iter(paginator.paginate(**params))
        first_page = next(pages)
        column_info = first_page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        column_names = tuple(column["Name"] for column in column_info)
        column_types = tuple(column["Type"] for column in column_info)
        data: List[Tuple[str, ...]] = []
        for page in itertools.chain([first_page], pages):
            rows = page["ResultSet"]["Rows"]
            data += [
                tuple(item.get("VarCharValue") for item in row["Data"]) for row in rows
            ]
        if data and data[0] == column_names:
            # Skip the first row iff it contains column names.
            # Athena often returns column names in the first row but not always.
            # (for example, SHOW PARTITIONS results do not have this header).
            data = data[1:]
        return QueryResults(column_names, column_types, data)
