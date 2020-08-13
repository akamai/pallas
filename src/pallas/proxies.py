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

The proxies are internal classes by the :class:`.Athena` client
to issue requests to AWS.
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import boto3

from pallas.csv import read_csv
from pallas.info import QueryInfo
from pallas.results import QueryResults
from pallas.storage.s3 import s3_parse_uri, s3_wrap_body
from pallas.utils import truncate_str

logger = logging.getLogger("pallas")


ColumnNames = Sequence[str]
ColumnTypes = Sequence[str]
Row = Sequence[Optional[str]]


class AthenaProxy(metaclass=ABCMeta):
    """
    Proxy to AWS Athena.

    This is an internal interface that is used by the :class:`.Athena` client.
    The :class:`.Boto3Proxy` implementation will be used in most cases,
    but it can be replaced by :class:`.FakeProxy` for testing.
    """

    @abstractmethod
    def start_query_execution(
        self,
        sql: str,
        *,
        database: Optional[str] = None,
        workgroup: Optional[str] = None,
        output_location: Optional[str] = None,
    ) -> str:
        """
        Submit a query.

        :param sql: an SQL query to be executed
        :param database: a name of Athena database to be queried
        :param workgroup: a name of Athena workgroup
        :param output_location: URI of output location on S3
        :return: execution_id
        """

    @abstractmethod
    def get_query_execution(self, execution_id: str) -> QueryInfo:
        """
        Retrieve information about a query execution.

        Returns a status of the query with other information.
        """

    @abstractmethod
    def get_query_results(self, info: QueryInfo) -> QueryResults:
        """
        Retrieve results of a query execution.

        Waits until the query execution finishes and downloads results.
        """

    @abstractmethod
    def stop_query_execution(self, execution_id: str) -> None:
        """
        Kill a query execution.
        """


class Boto3Proxy(AthenaProxy):
    """
    Proxy to AWS Athena using the boto3 library.

    This is an internal class that is used by the :class:`.Athena` client.
    It can be replaced by :class:`.FakeProxy` for testing.
    """

    _athena_client: Any  # boto3 Athena client
    _s3_client: Any  # boto3 S3 client

    def __init__(
        self,
        *,
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

    def start_query_execution(
        self,
        sql: str,
        database: Optional[str] = None,
        workgroup: Optional[str] = None,
        output_location: Optional[str] = None,
    ) -> str:
        params: Dict[str, Any] = dict(QueryString=sql)
        if database is not None:
            params.update(QueryExecutionContext={"Database": database})
        if workgroup is not None:
            params.update(WorkGroup=workgroup)
        if output_location is not None:
            params.update(ResultConfiguration={"OutputLocation": output_location})
        logger.info(f"Athena StartQueryExecution: QueryString={truncate_str(sql)!r}")
        response = self._athena_client.start_query_execution(**params)
        execution_id = cast(str, response["QueryExecutionId"])
        logger.info(f"Athena QueryExecutionId={execution_id!r} started.")
        return execution_id

    def get_query_execution(self, execution_id: str) -> QueryInfo:
        logger.info(f"Athena GetQueryExecution: QueryExecutionId={execution_id!r}")
        response = self._athena_client.get_query_execution(
            QueryExecutionId=execution_id
        )
        info = QueryInfo(response["QueryExecution"])
        logger.info(f"Athena QueryExecution: {info}")
        return info

    def get_query_results(self, info: QueryInfo) -> QueryResults:
        execution_id = info.execution_id
        params = dict(QueryExecutionId=execution_id)
        logger.info(f"Athena GetQueryResults: QueryExecutionId={execution_id!r}")
        response = self._athena_client.get_query_results(**params)
        column_names = _read_column_names(response)
        column_types = _read_column_types(response)
        if response.get("NextToken"):
            logger.info("Athena ResultSet paginated. Will download from S3.")
            data = self._download_data(info)
        else:
            data = _read_data(response)
            logger.info(
                f"Athena ResultSet complete: {len(data)} rows (including header)"
            )
        fixed_data = _fix_data(column_names, data)
        return QueryResults(column_names, column_types, fixed_data)

    def stop_query_execution(self, execution_id: str) -> None:
        logger.info(f"Athena StopQueryExecution: QueryExecutionId={execution_id!r}")
        self._athena_client.stop_query_execution(QueryExecutionId=execution_id)

    def _download_data(self, info: QueryInfo) -> Sequence[Row]:
        output_location = info.output_location
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


def _read_column_names(response: Mapping[str, Any]) -> ColumnNames:
    column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
    return tuple(column["Name"] for column in column_info)


def _read_column_types(response: Mapping[str, Any]) -> ColumnTypes:
    column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
    return tuple(column["Type"] for column in column_info)


def _read_data(response: Mapping[str, Any]) -> Sequence[Row]:
    rows = response["ResultSet"]["Rows"]
    return [tuple(item.get("VarCharValue") for item in row["Data"]) for row in rows]
