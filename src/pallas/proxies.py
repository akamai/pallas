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
Proxies to AWS Athena APIs.

Classes implemented in this module are the core of the Pallas library.
They issue actual requests using boto3.

Other clients are implemented as decorators wrapping the core
and offering same interface.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import boto3

from pallas.base import Athena, Query
from pallas.csv import read_csv
from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.storage.s3 import s3_parse_uri, s3_wrap_body
from pallas.utils import Fibonacci, truncate_str

logger = logging.getLogger("pallas")


ColumnNames = Sequence[str]
ColumnTypes = Sequence[str]
Row = Sequence[Optional[str]]


class AthenaProxy(Athena):
    """
    Proxy to AWS Athena.

    This is the core implementation of the :class:`.Athena` interface.
    It executes queries via AWS APIs using boto3 library.

    It can be decorated by wrappers to provide additional functionality.

    :param database: a name of Athena database.
        If omitted, database should be specified in SQL.
    :param workgroup: a name of Athena workgroup.
        If omitted, default workgroup will be used.
    :param output_location: an output location at S3 for query results.
        Optional if a default location is specified for the *workgroup*.
    :param region: an AWS region.
        By default, a region from AWS config is used.
    :param athena_client: a boto3 client to use.
        By default, a new client is constructed.
    :param s3_client: a boto3 client to use.
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
        region: Optional[str] = None,
        athena_client: Optional[Any] = None,
        s3_client: Optional[Any] = None,
    ) -> None:
        if athena_client is None:
            athena_client = boto3.client("athena", region_name=region)
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
        logger.info(f"Athena StartQueryExecution: QueryString={truncate_str(sql)!r}")
        response = self._athena_client.start_query_execution(**params)
        query = self.get_query(response["QueryExecutionId"])
        logger.info(f"Athena QueryExecutionId={query.execution_id!r} started.")
        return query

    def get_query(self, execution_id: str) -> Query:
        return QueryProxy(
            execution_id, athena_client=self._athena_client, s3_client=self._s3_client
        )


class QueryProxy(Query):
    """
    Proxy to an Athena query execution.

    This is the core implementation of the :class:`.Query` interface.
    It can monitor and control the query execution via AWS APIs
    using boto3 library.

    It can be decorated by wrappers to provide additional functionality.
    """

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
        logger.info(f"Athena GetQueryExecution: QueryExecutionId={self.execution_id!r}")
        response = self._athena_client.get_query_execution(
            QueryExecutionId=self.execution_id
        )
        info = QueryInfo(response["QueryExecution"])
        logger.info(f"Athena QueryExecution: {info}")
        if info.finished:
            self._finished_info = info
        return info

    def get_results(self) -> QueryResults:
        self.join()
        params = dict(QueryExecutionId=self.execution_id)
        logger.info(f"Athena GetQueryResults: QueryExecutionId={self.execution_id!r}")
        response = self._athena_client.get_query_results(**params)
        column_names = self._read_column_names(response)
        column_types = self._read_column_types(response)
        if response.get("NextToken"):
            logger.info("Athena ResultSet paginated. Will download from S3.")
            data = self._download_data()
        else:
            data = self._read_data(response)
            logger.info(
                f"Athena ResultSet complete: {len(data)} rows (including header)"
            )
        fixed_data = _fix_data(column_names, data)
        return QueryResults(column_names, column_types, fixed_data)

    def kill(self) -> None:
        logger.info(
            f"Athena StopQueryExecution: QueryExecutionId={self.execution_id!r}"
        )
        self._athena_client.stop_query_execution(QueryExecutionId=self.execution_id)

    def join(self) -> None:
        for delay in Fibonacci(max_value=60):
            info = self.get_info()
            if info.finished:
                info.check()
                break
            time.sleep(delay)

    def _read_column_names(self, response: Mapping[str, Any]) -> ColumnNames:
        column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        return tuple(column["Name"] for column in column_info)

    def _read_column_types(self, response: Mapping[str, Any]) -> ColumnTypes:
        column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        return tuple(column["Type"] for column in column_info)

    def _read_data(self, response: Mapping[str, Any]) -> Sequence[Row]:
        rows = response["ResultSet"]["Rows"]
        return [tuple(item.get("VarCharValue") for item in row["Data"]) for row in rows]

    def _download_data(self) -> Sequence[Row]:
        output_location = self.get_info().output_location
        bucket, key = s3_parse_uri(output_location)
        params = dict(Bucket=bucket, Key=key)
        logger.info(f"S3 GetObject:" f" Bucket={bucket!r} Key={key!r}")
        response = self._s3_client.get_object(**params)
        with s3_wrap_body(response["Body"]) as stream:
            data = list(read_csv(stream))
        logger.info(f"S3 Body downloaded: {len(data)} rows (including header)")
        return data


def _fix_data(column_names: ColumnNames, data: Sequence[Row]) -> Sequence[Row]:
    """
    Fix malformed data returned from Athena.

    Queries executed by Presto (typically queries with SELECT)
    repeat column names in the first row of data,
    so we have to remove them.

    Queries by Hive (typically queries with DESCRIBE)
    do not repeat column names, but all columns are combined to one.

    Try to fix both of the above problems here.
    """
    if data and data[0] == column_names:
        # DQL, SELECT statements executed by Presto
        data = data[1:]
    elif all(len(row) == 1 for row in data) and len(column_names) > 1:
        # DCL, DESCRIBE statements executed by Hive
        values = (row[0] for row in data if row[0] is not None)
        data = [v.split("\t", maxsplit=len(column_names) - 1) for v in values]
    return data
