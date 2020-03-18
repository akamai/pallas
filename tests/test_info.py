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

import copy
import datetime as dt

import pytest

from pallas.exceptions import AthenaQueryError
from pallas.info import QueryInfo, format_price, format_size, format_time


def test_format_price():
    assert format_price(0.00011) == "0.01¢"
    assert format_price(0.0011) == "0.11¢"
    assert format_price(0.011) == "1.10¢"
    assert format_price(0.11) == "11.00¢"
    assert format_price(1.1) == "$1.10"


def test_format_size():
    assert format_size(1.1) == "1B"
    assert format_size(1_100) == "1.10kB"
    assert format_size(1_100_000) == "1.10MB"
    assert format_size(1_100_000_000) == "1.10GB"
    assert format_size(1_100_000_000_000) == "1.10TB"
    assert format_size(1_100_000_000_000_000) == "1100.00TB"


def test_format_time():
    assert format_time(dt.timedelta(seconds=13, milliseconds=570)) == "13.6s"
    assert format_time(dt.timedelta(minutes=13, seconds=57)) == "13min 57s"


EXAMPLE_INFO = {
    "QueryExecutionId": "e83608...1e",
    "Query": "SELECT ...",
    "StatementType": "DML",
    "ResultConfiguration": {
        "OutputLocation": "s3://example-output-location/e83608...1e.csv"
    },
    "QueryExecutionContext": {"Database": "example-database"},
    "Status": {
        "State": "SUCCEEDED",
        "SubmissionDateTime": ...,
        "CompletionDateTime": ...,
    },
    "Statistics": {
        "EngineExecutionTimeInMillis": 300,
        "DataScannedInBytes": 10 ** 9,
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
    def test_repr(self):
        info = get_info()
        assert repr(info) == (
            "<QueryInfo: 'SUCCEEDED, scanned 1.00GB in 0.4s, approx. price 0.50¢'>"
        )

    def test_str(self):
        info = get_info()
        assert str(info) == "SUCCEEDED, scanned 1.00GB in 0.4s, approx. price 0.50¢"

    def test_properties(self):
        info = get_info()
        assert info.execution_id == "e83608...1e"
        assert info.sql == "SELECT ..."
        assert info.database == "example-database"
        assert info.output_location == "s3://example-output-location/e83608...1e.csv"
        assert info.scanned_bytes == 1_000_000_000
        assert info.execution_time == dt.timedelta(seconds=0.4)
        assert info.approx_price == 0.005

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
