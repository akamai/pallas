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

import pytest

from pallas.testing import AthenaFake


@pytest.fixture(name="fake_athena")
def fake_athena_fixture():
    return AthenaFake()


class TestQueryFake:
    def test_repr(self, fake_athena):
        assert repr(fake_athena) == "<AthenaFake>"

    def test_query_repr(self, fake_athena):
        query = fake_athena.submit("SELECT 1")
        assert repr(query) == "<QueryFake: execution_id='query-1'>"

    def test_submit(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        assert query.execution_id == "query-1"
        assert fake_athena.request_log == ["StartQueryExecution"]

    def test_submit_multiple(self, fake_athena):
        query1 = fake_athena.submit("SELECT ...")
        query2 = fake_athena.submit("SELECT ...")
        assert query1.execution_id == "query-1"
        assert query2.execution_id == "query-2"
        assert fake_athena.request_log == ["StartQueryExecution", "StartQueryExecution"]

    def test_get_query(self, fake_athena):
        query1 = fake_athena.submit("SELECT ...")
        query2 = fake_athena.get_query("query-1")
        assert query2.execution_id == query1.execution_id

    def test_query_info(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        fake_athena.request_log.clear()
        info = query.get_info()
        assert info.execution_id == "query-1"
        assert info.sql == "SELECT ..."
        assert info.database is None
        assert info.finished
        assert info.succeeded
        assert fake_athena.request_log == ["GetQueryExecution"]

    def test_query_info_remembered(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        fake_athena.request_log.clear()
        query.get_info()
        query.get_info()
        assert fake_athena.request_log == ["GetQueryExecution"]

    def test_query_info_remembered_not_shared(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        fake_athena.request_log.clear()
        query.get_info()
        fake_athena.get_query(query.execution_id).get_info()
        assert fake_athena.request_log == ["GetQueryExecution", "GetQueryExecution"]

    def test_default_query_results(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        results = query.get_results()
        assert list(results) == []

    def test_custom_query_results(self, fake_athena):
        fake_athena.column_names = "id", "name"
        fake_athena.data = [("1", "foo"), ("2", "bar")]
        query = fake_athena.submit("SELECT ...")
        results = query.get_results()
        assert list(results) == [
            {"id": "1", "name": "foo"},
            {"id": "2", "name": "bar"},
        ]

    def test_custom_query_results_with_custom_types(self, fake_athena):
        fake_athena.column_names = "id", "name"
        fake_athena.column_types = "integer", "varchar"
        fake_athena.data = [("1", "foo"), ("2", "bar")]
        query = fake_athena.submit("SELECT ...")
        results = query.get_results()
        assert list(results) == [
            {"id": 1, "name": "foo"},
            {"id": 2, "name": "bar"},
        ]
