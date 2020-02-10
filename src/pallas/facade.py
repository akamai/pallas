import os

import boto3


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
        return query.join()

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

    def get_status(self):
        response = self._client.get_query_execution(QueryExecutionId=self.execution_id)
        return response["QueryExecution"]

    def kill(self):
        self._client.stop_query_execution(QueryExecutionId=self.execution_id)

    def join(self):
        state = "RUNNING"
        while state == "RUNNING":
            info = self.get_status()
            state = info["Status"]["State"]
        return info
