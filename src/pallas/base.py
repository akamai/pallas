from abc import abstractmethod, ABCMeta
from collections.abc import Sequence

from pallas.conversions import convert_value


class Athena(metaclass=ABCMeta):
    def execute(self, sql):
        """Submit query execution and wait for results."""
        query = self.submit(sql)
        query.join()
        return query.get_results()

    @abstractmethod
    def submit(self, sql):
        """Submit query execution."""

    @abstractmethod
    def get_query(self, execution_id):
        """Get previously submitted query execution"""


class Query(metaclass=ABCMeta):
    @property
    @abstractmethod
    def execution_id(self):
        """Athena query execution ID."""

    @abstractmethod
    def get_info(self):
        """Retrieve information about this query execution."""

    @abstractmethod
    def get_results(self):
        """Retrieve results of this query execution."""

    @abstractmethod
    def kill(self):
        """Kill this query execution."""

    @abstractmethod
    def join(self):
        """Wait until this query execution finishes."""


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
    def finished(self):
        return self.state in ("SUCCEEDED", "FAILED", "CANCELLED")

    @property
    def succeeded(self):
        return self.state == "SUCCEEDED"

    @property
    def state(self):
        return self._data["Status"]["State"]


class QueryResults(Sequence):
    def __init__(self, column_names, column_types, data):
        self._column_names = column_names
        self._column_types = column_types
        self._data = data

    def __getitem__(self, index):
        row = self._data[index]
        info = zip(self._column_names, self._column_types, row)
        return {cn: convert_value(ct, v) for cn, ct, v in info}

    def __len__(self):
        return len(self._data)
