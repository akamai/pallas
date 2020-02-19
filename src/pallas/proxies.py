from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import boto3

from pallas.base import Athena, Query
from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.waiting import Fibonacci

logger = logging.getLogger("pallas")


class AthenaProxy(Athena):
    """
    Athena client.

    Executes queries using via AWS API using boto3 library.
    This is the core implementation of the :class:`.Athena` interface.

    It can be decorated by wrappers to provide additional functionality.

    :param database: a name of Athena database.
        If omitted, database should be specified in SQL.
    :param workgroup: a name of Athena workgroup.
        If omitted, default workgroup will be used.
    :param output_location: an output location at S3 for query results.
        Optional if a default location is specified for the *workgroup*.
    :param region_name: an AWS region.
        By default, a region from AWS config is used.
    :param client: a boto3 client to use.
        By default, a new client is constructed.
    """

    _client: Any  # boto3 Athena client
    _database: Optional[str]
    _workgroup: Optional[str]
    _output_location: Optional[str]

    def __init__(
        self,
        *,
        database: Optional[str] = None,
        workgroup: Optional[str] = None,
        output_location: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        if client is None:
            client = boto3.client("athena", region_name=region_name)
        self._client = client
        self._database = database
        self._workgroup = workgroup
        self._output_location = output_location

    def __repr__(self) -> str:
        parts = []
        if self.database is not None:
            parts.append(f"database={self.database!r}")
        if self.output_location is not None:
            parts.append(f"output_location={self.output_location!r}")
        return f"<{type(self).__name__}: {', '.join(parts)}>"

    @property
    def database(self) -> Optional[str]:
        return self._database

    @property
    def workgroup(self) -> Optional[str]:
        return self._workgroup

    @property
    def output_location(self) -> Optional[str]:
        return self._output_location

    def submit(self, sql: str, *, ignore_cache: bool = False) -> Query:
        params: Dict[str, Any] = dict(QueryString=sql)
        if self._database is not None:
            params.update(QueryExecutionContext={"Database": self._database})
        if self._workgroup is not None:
            params.update(WorkGroup=self._workgroup)
        if self._output_location is not None:
            params.update(ResultConfiguration={"OutputLocation": self._output_location})
        response = self._client.start_query_execution(**params)
        query = self.get_query(response["QueryExecutionId"])
        logger.info(
            f"Called Athena StartQueryExecution:"
            f" QueryExecutionId={query.execution_id!r}"
        )
        return query

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
        logger.info(
            f"Called Athena GetQueryExecution:"
            f" QueryExecutionId={self.execution_id!r}: {info}"
        )
        if info.finished:
            self._finished_info = info
        return info

    def get_results(self) -> QueryResults:
        params = dict(QueryExecutionId=self.execution_id)
        paginator = self._client.get_paginator("get_query_results")
        pages = iter(paginator.paginate(**params))

        first_page = next(pages)
        column_info = first_page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        column_names = tuple(column["Name"] for column in column_info)
        column_types = tuple(column["Type"] for column in column_info)
        data = self._read_page(first_page)
        if data and data[0] == column_names:
            # Skip the first row iff it contains column names.
            # Athena often returns column names in the first row but not always.
            # (for example, SHOW PARTITIONS results do not have this header).
            data = data[1:]
        logger.info(
            f"Called Athena GetQueryResults:"
            f" QueryExecutionId={self.execution_id!r}: {len(data)} rows"
        )
        for i, page in enumerate(pages, start=2):
            data += self._read_page(page)
            logger.info(
                f"Called Athena GetQueryResults:"
                f" QueryExecutionId={self.execution_id!r}: {len(data)} rows"
            )
        return QueryResults(column_names, column_types, data)

    def _read_page(self, page: Mapping[str, Any]) -> List[Sequence[str]]:
        rows = page["ResultSet"]["Rows"]
        return [tuple(item.get("VarCharValue") for item in row["Data"]) for row in rows]

    def kill(self) -> None:
        self._client.stop_query_execution(QueryExecutionId=self.execution_id)
        logger.info(
            f"Called Athena StopQueryExecution:"
            f" QueryExecutionId={self.execution_id!r}"
        )

    def join(self) -> None:
        for delay in Fibonacci(max_value=60):
            info = self.get_info()
            if info.finished:
                info.check()
                break
            time.sleep(delay)
