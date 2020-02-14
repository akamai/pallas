import copy

import pytest

from pallas.exceptions import AthenaQueryError
from pallas.info import QueryInfo


EXAMPLE_INFO = {
    "QueryExecutionId": "e836081e-77f3-5700-4cdb-df1ba5799046",
    "Query": "SELECT 1",
    "StatementType": "DML",
    "ResultConfiguration": {
        "OutputLocation": "s3://example-output-location/e836081e-77f3-5700-4cdb-df1ba5799046.csv"  # noqa: E501
    },
    "QueryExecutionContext": {"Database": "example-database"},
    "Status": {
        "State": "SUCCEEDED",
        "SubmissionDateTime": ...,
        "CompletionDateTime": ...,
    },
    "Statistics": {
        "EngineExecutionTimeInMillis": 300,
        "DataScannedInBytes": 0,
        "TotalExecutionTimeInMillis": 400,
        "QueryQueueTimeInMillis": 50,
        "ServiceProcessingTimeInMillis": 20,
    },
    "WorkGroup": "primary",
}


def get_info(*, state=None, state_reason=None):
    data = copy.deepcopy(EXAMPLE_INFO)
    if state is not None:
        data["Status"]["State"] = state
    if state_reason is not None:
        data["Status"]["StateChangeReason"] = state_reason
    return QueryInfo(data)


class TestQueryInfo:

    def test_properties(self):
        info = get_info()
        assert info.execution_id == "e836081e-77f3-5700-4cdb-df1ba5799046"
        assert info.sql == "SELECT 1"
        assert info.database == "example-database"

    @pytest.mark.parametrize("state", ["QUEUED", "RUNNING"])
    def test_running(self, state):
        info = get_info(state=state)
        assert not info.finished
        assert not info.succeeded
        assert info.state == state
        assert info.state_reason is None
        info.check()  # Should not raise

    def test_succeeded(self):
        info = get_info(state="SUCCEEDED")
        assert info.finished
        assert info.succeeded
        assert info.state == "SUCCEEDED"
        assert info.state_reason is None
        info.check()  # Should not raise

    @pytest.mark.parametrize("state", ["FAILED", "CANCELLED"])
    def test_failed(self, state):
        info = get_info(state=state, state_reason="Something happened.")
        assert info.finished
        assert not info.succeeded
        assert info.state == state
        assert info.state_reason == "Something happened."
        with pytest.raises(AthenaQueryError):
            info.check()
