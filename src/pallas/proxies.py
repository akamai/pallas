import itertools

import boto3

from pallas.base import Athena, Query, QueryInfo, QueryResults


class AthenaProxy(Athena):
    def __init__(
        self, *, output_location, client=None, database=None, region_name=None
    ):
        if client is None:
            client = boto3.client("athena", region_name=region_name)
        self._client = client
        self._output_location = output_location
        self._database = database

    def submit(self, sql):
        params = dict(
            QueryString=sql,
            ResultConfiguration={"OutputLocation": self._output_location},
        )
        if self._database is not None:
            params.update(QueryExecutionContext={"Database": self._database})
        response = self._client.start_query_execution(**params)
        return self.get_query(response["QueryExecutionId"])

    def get_query(self, execution_id):
        return QueryProxy(self._client, execution_id)


class QueryProxy(Query):
    def __init__(self, client, execution_id):
        self._client = client
        self._execution_id = execution_id

    @property
    def execution_id(self):
        return self._execution_id

    def get_info(self):
        response = self._client.get_query_execution(QueryExecutionId=self.execution_id)
        return QueryInfo(response["QueryExecution"])

    def kill(self):
        self._client.stop_query_execution(QueryExecutionId=self.execution_id)

    def join(self):
        while True:
            info = self.get_info()
            if info.finished:
                return info

    def get_results(self):
        params = dict(QueryExecutionId=self.execution_id)
        paginator = self._client.get_paginator("get_query_results")
        pages = iter(paginator.paginate(**params))
        first_page = next(pages)
        column_info = first_page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        column_names = tuple(column["Name"] for column in column_info)
        column_types = tuple(column["Type"] for column in column_info)
        data = []
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
