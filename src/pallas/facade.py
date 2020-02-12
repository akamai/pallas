from collections.abc import Sequence
import os

import boto3

from pallas.conversions import convert_value


class Athena:
    def __init__(
        self, *, output_location, client=None, database=None, region_name=None
    ):
        if client is None:
            client = boto3.client("athena", region_name=region_name)
        self._client = client
        self._output_location = output_location
        self._database = database

    @classmethod
    def from_environ(cls, environ=None, *, prefix="PALLAS"):
        if environ is None:
            environ = os.environ
        return cls(
            output_location=environ[f"{prefix}_OUTPUT_LOCATION"],
            database=environ[f"{prefix}_DATABASE"],
            region_name=environ.get(f"{prefix}_REGION_NAME"),
        )

    def execute(self, sql):
        query = self.submit(sql)
        query.join()
        return query.get_results()

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
        return AthenaQuery(self._client, execution_id)


class AthenaQuery:
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
            if info.done:
                return info

    def get_results(self):
        params = dict(QueryExecutionId=self.execution_id)
        response = self._client.get_query_results(**params)
        column_info = response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
        column_names = tuple(column["Name"] for column in column_info)
        column_types = tuple(column["Type"] for column in column_info)
        rows = response["ResultSet"]["Rows"]
        data = [tuple(item.get("VarCharValue") for item in row["Data"]) for row in rows]
        if data and data[0] == column_names:
            # Skip the first row iff it contains column names.
            # Athena often returns column names in the first row but not always.
            # (for example, SHOW PARTITIONS results do not have this header).
            data = data[1:]
        return QueryResultSet(column_names, column_types, data)


class QueryInfo:
    def __init__(self, data):
        self._data = data

    @property
    def execution_id(self):
        return self._data["QueryExecutionId"]

    @property
    def sql(self):
        return self._data["Query"]

    @property
    def database(self):
        return self._data["QueryExecutionContext"].get("Database")

    @property
    def done(self):
        return self.state in ("SUCCEEDED", "FAILED", "CANCELLED")

    @property
    def succeeded(self):
        return self.state == "SUCCEEDED"

    @property
    def state(self):
        return self._data["Status"]["State"]


class QueryResultSet(Sequence):
    def __init__(self, column_names, column_types, data):
        self._column_names = column_names
        self._column_types = column_types
        self._data = data

    def __getitem__(self, index):
        row = self._data[index]
        info = zip(self._column_names, self._column_types, row)
        return {cn: self._convert_value(ct, v) for cn, ct, v in info}

    def __len__(self):
        return len(self._data)

    def _convert_value(self, column_type, value):
        if value is None:
            return None
        return convert_value(column_type, value)
