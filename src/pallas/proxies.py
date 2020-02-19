from __future__ import annotations

import csv
import logging
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import boto3

from pallas.base import Athena, Query
from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.storage import s3_parse_uri, s3_wrap_body
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
    :param athena_client: a boto3 client to use.
        By default, a new client is constructed.
    """

    _athena_client: Any  # boto3 Athena client
    _s3_client: Any  # boto3 S3 client
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
        athena_client: Optional[Any] = None,
        s3_client: Optional[Any] = None,
    ) -> None:
        if athena_client is None:
            athena_client = boto3.client("athena", region_name=region_name)
        if s3_client is None:
            s3_client = boto3.client("s3")
        self._athena_client = athena_client
        self._s3_client = s3_client
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
        response = self._athena_client.start_query_execution(**params)
        query = self.get_query(response["QueryExecutionId"])
        logger.info(
            f"Called Athena StartQueryExecution:"
            f" QueryExecutionId={query.execution_id!r}"
        )
        return query

    def get_query(self, execution_id: str) -> Query:
        return QueryProxy(
            execution_id, athena_client=self._athena_client, s3_client=self._s3_client
        )


class QueryProxy(Query):

    _athena_client: Any  # boto3 Athena client
    _s3_client: Any  # boto3 S3 client
    _execution_id: str
    _finished_info: Optional[QueryInfo] = None

    def __init__(
        self, execution_id: str, *, athena_client: Any, s3_client: Any
    ) -> None:
        self._athena_client = athena_client
        self._s3_client = s3_client
        self._execution_id = execution_id

    @property
    def execution_id(self) -> str:
        return self._execution_id

    def get_info(self) -> QueryInfo:
        # Query info is cached if the query finished so it cannot change.
        if self._finished_info is not None:
            return self._finished_info
        response = self._athena_client.get_query_execution(
            QueryExecutionId=self.execution_id
        )
        info = QueryInfo(response["QueryExecution"])
        logger.info(
            f"Called Athena GetQueryExecution:"
            f" QueryExecutionId={self.execution_id!r}: {info}"
        )
        if info.finished:
            self._finished_info = info
        return info

    def get_results(self) -> QueryResults:
        self.join()
        params = dict(QueryExecutionId=self.execution_id)
        response = self._athena_client.get_query_results(**params)
        logger.info(
            f"Called Athena GetQueryResults:"
            f" QueryExecutionId={self.execution_id!r}:"
            f" {len(response['ResultSet']['Rows'])} rows"
        )

        column_names = self._read_column_names(response)
        column_types = self._read_column_types(response)
        if response.get("NextToken"):
            data = self._download_data()
        else:
            data = self._read_data(response)
        if data and data[0] == column_names:
            # Skip the first row iff it contains column names.
            # Athena often returns column names in the first row but not always.
            # (for example, SHOW PARTITIONS results do not have this header).
            data = data[1:]

        return QueryResults(column_names, column_types, data)

    def kill(self) -> None:
        self._athena_client.stop_query_execution(QueryExecutionId=self.execution_id)
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

    def _read_column_names(self, response: Mapping[str, Any]) -> Sequence[str]:
        column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        return tuple(column["Name"] for column in column_info)

    def _read_column_types(self, response: Mapping[str, Any]) -> Sequence[str]:
        column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        return tuple(column["Type"] for column in column_info)

    def _read_data(self, response: Mapping[str, Any]) -> Sequence[Sequence[str]]:
        rows = response["ResultSet"]["Rows"]
        return [tuple(item.get("VarCharValue") for item in row["Data"]) for row in rows]

    def _download_data(self) -> Sequence[Sequence[str]]:
        output_location = self.get_info().output_location
        bucket, key = s3_parse_uri(output_location)
        params = dict(Bucket=bucket, Key=key)
        response = self._s3_client.get_object(**params)
        with s3_wrap_body(response["Body"]) as stream:
            reader = csv.reader(stream, delimiter=",", doublequote=True)
            data = [tuple(row) for row in reader]
        logger.info(f"Downloaded results from S3:" f" Bucket={bucket!r} Key={key!r}")
        return data
