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

from pallas import Athena
from pallas.testing import AthenaFake


@pytest.fixture(name="fake")
def fake_fixture():
    return AthenaFake()


@pytest.fixture(name="athena")
def athena_fixture(fake):
    return Athena(fake)


class TestAthena:
    def test_submit(self, athena, fake):
        athena.submit("SELECT 1")
        assert fake.request_log == ["StartQueryExecution"]

    def test_get_query(self, athena, fake):
        orig_query = athena.submit("SELECT 1")
        fake.request_log.clear()
        athena.get_query(orig_query.execution_id)
        assert fake.request_log == []

    def test_execute(self, athena, fake):
        athena.execute("SELECT 1")
        assert fake.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]


class TestQuery:
    def test_get_info(self, athena, fake):
        """Test that query info is retrieved."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_info()
        assert fake.request_log == ["GetQueryExecution"]

    def test_get_info_twice(self, athena, fake):
        """Test that query info is cached"""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_info()
        query.get_info()
        assert fake.request_log == ["GetQueryExecution"]

    def test_get_info_twice_not_finished(self, athena, fake):
        """Test that query info is not cached when query not finished."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        fake.state = "RUNNING"
        query.get_info()
        query.get_info()
        assert fake.request_log == ["GetQueryExecution", "GetQueryExecution"]

    def test_join(self, athena, fake):
        """Test that query info is checked."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.join()
        assert fake.request_log == ["GetQueryExecution"]

    @pytest.mark.xfail
    def test_join_twice(self, athena, fake):
        """Test that query info is cached."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.join()
        query.join()
        assert fake.request_log == ["GetQueryExecution"]

    @pytest.mark.xfail
    def test_get_info_and_join(self, athena, fake):
        """Test that query info is cached."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_info()
        query.join()
        assert fake.request_log == ["GetQueryExecution"]

    @pytest.mark.xfail
    def test_join_and_get_info(self, athena, fake):
        """Test that query info is cached."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.join()
        query.get_info()
        assert fake.request_log == ["GetQueryExecution"]

    def test_get_results(self, athena, fake):
        """Test that get_results check query info."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_results()
        assert fake.request_log == ["GetQueryExecution", "GetQueryResults"]

    @pytest.mark.xfail
    def test_get_results_twice(self, athena, fake):
        """Test that get_results caches query info but not results."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_results()
        query.get_results()
        assert fake.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
            "GetQueryResults",
        ]

    @pytest.mark.xfail
    def test_get_info_and_get_results(self, athena, fake):
        """Test that get_results use cached query info."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_info()
        query.get_results()
        assert fake.request_log == ["GetQueryExecution", "GetQueryResults"]

    @pytest.mark.xfail
    def test_join_and_get_results(self, athena, fake):
        """Test that get_results use cached query info."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.join()
        query.get_results()
        assert fake.request_log == ["GetQueryExecution", "GetQueryResults"]

    @pytest.mark.xfail
    def test_get_results_and_get_info(self, athena, fake):
        """Test that get_results cache query info."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_results()
        query.get_info()
        assert fake.request_log == ["GetQueryExecution", "GetQueryResults"]

    @pytest.mark.xfail
    def test_get_results_and_join(self, athena, fake):
        """Test that get_results cache query info."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.get_results()
        query.join()
        assert fake.request_log == ["GetQueryExecution", "GetQueryResults"]

    def test_kill(self, athena, fake):
        """Test that query is killed."""
        query = athena.submit("SELECT 1")
        fake.request_log.clear()
        query.kill()
        assert fake.request_log == ["StopQueryExecution"]
