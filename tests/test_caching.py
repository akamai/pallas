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

from pallas import Athena, AthenaQueryError
from pallas.storage.memory import MemoryStorage
from pallas.testing import FakeProxy

FAKE_DATA = [("1", "foo", None)]
ANOTHER_FAKE_DATA = [("2", "bar", None)]


@pytest.fixture(name="storage")
def storage_fixture():
    return MemoryStorage()


@pytest.fixture(name="proxy")
def proxy_fixture():
    """
    Athena mock decorated by caching wrapper.
    """
    proxy = FakeProxy()
    proxy.column_names = "id", "name", "value"
    proxy.column_types = "integer", "varchar", "double"
    proxy.data = FAKE_DATA
    return proxy


@pytest.fixture(name="local_athena")
def local_athena_fixture(proxy, storage):
    """
    Athena client caching locally.

    In local mode, both query execution IDs and results are cached.
    """
    athena = Athena(proxy)
    athena.cache.local_storage = storage
    return athena


@pytest.fixture(name="remote_athena")
def remote_athena_fixture(proxy, storage):
    """
    Athena client caching remotely.

    In remote mode, only query execution IDs are cached.
    """
    athena = Athena(proxy)
    athena.cache.remote_storage = storage
    return athena


@pytest.fixture(name="full_athena")
def full_athena_fixture(proxy, storage):
    """
    Athena client caching both locally and remotely.
    """
    athena = Athena(proxy)
    athena.cache.local_storage = storage
    athena.cache.remote_storage = MemoryStorage()
    return athena


@pytest.fixture(name="athena", params=["local", "remote", "full"])
def athena_fixture(request):
    """
    Yields caching wrapper in both remote and local mode.
    """
    return request.getfixturevalue(f"{request.param}_athena")


def assert_query_results(results):
    assert list(results) == [{"id": 1, "name": "foo", "value": None}]


def assert_another_query_results(results):
    assert list(results) == [{"id": 2, "name": "bar", "value": None}]


class TestAthenaCache:

    # Test execute method

    def test_execute_one_query(self, athena, proxy):
        """Test execution of a query not in cache."""
        results = athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_local_execute_one_query_cache_size(self, local_athena, storage):
        """Test that query ID and results are written to cache."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        assert storage.size() == 2

    def test_remote_execute_one_query_cache_size(self, remote_athena, storage):
        """Test that query ID is written to cache."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        assert storage.size() == 1

    def test_full_execute_one_query_cache_size(self, full_athena):
        """Test that query ID and results are written to cache."""
        full_athena.execute("SELECT 1 id, 'foo' name")
        assert full_athena.cache.local_storage.size() == 2
        assert full_athena.cache.remote_storage.size() == 1

    def test_execute_one_query_not_select(self, athena, proxy, storage):
        """Test execution of a query that should not be cached."""
        results = athena.execute("CREATE TABLE ...")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)
        assert storage.size() == 0

    def test_local_execute_second_query(self, local_athena, proxy, storage):
        """Test execution of a query in cache."""
        local_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        proxy.request_log.clear()
        results = local_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == []
        assert_query_results(results)
        assert storage.size() == 2

    def test_remote_execute_second_query(self, remote_athena, proxy, storage):
        """Test execution of a query in cache."""
        remote_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        proxy.request_log.clear()
        results = remote_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)
        assert storage.size() == 1

    def test_full_execute_second_query(self, full_athena, proxy):
        """Test execution of a query in cache."""
        full_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        proxy.request_log.clear()
        results = full_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == []
        assert_query_results(results)
        assert full_athena.cache.local_storage.size() == 2
        assert full_athena.cache.remote_storage.size() == 1

    def test_execute_second_query_not_select(self, athena, proxy, storage):
        """Test that only SELECT queries are cached."""
        athena.execute("CREATE TABLE ...")
        proxy.request_log.clear()
        results = athena.execute("CREATE TABLE ...")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)
        assert storage.size() == 0

    def test_execute_second_query_different_sql(self, athena, proxy):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        proxy.request_log.clear()
        proxy.data = ANOTHER_FAKE_DATA
        results = athena.execute("SELECT 2 id, 'bar' name")  # different SQL
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_another_query_results(results)

    def test_execute_second_query_different_database(self, athena, proxy):
        """Test that cache is unique to a database."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        proxy.request_log.clear()
        athena.database = "other"
        results = athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_execute_failed_not_cached(self, athena, proxy):
        """Test failed queries in cache are ignored."""
        proxy.state = "FAILED"
        with pytest.raises(AthenaQueryError):
            athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        proxy.state = "SUCCEEDED"
        results = athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "GetQueryExecution",
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_execute_failed_cached(self, athena, proxy):
        """Test failed queries can be cached if desired."""
        athena.cache.failed = True
        proxy.state = "FAILED"
        with pytest.raises(AthenaQueryError):
            athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        with pytest.raises(AthenaQueryError):
            athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == ["GetQueryExecution"]

    # Test athena.submit() method

    def test_submit_one_query(self, athena, proxy, storage):
        """Test that the caching wrapper submits a query if not in cache."""
        athena.submit("SELECT 1 id, 'foo' name")
        assert proxy.request_log == ["StartQueryExecution"]
        assert storage.size() == 1

    def test_submit_one_query_not_select(self, athena, proxy, storage):
        """Test that the caching wrapper submits a query if not in cache."""
        athena.submit("CREATE TABLE ...")
        assert proxy.request_log == ["StartQueryExecution"]
        assert storage.size() == 0

    def test_submit_second_query(self, athena, proxy, storage):
        """Test that one query is submitted only once."""
        athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        athena.submit("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "GetQueryExecution",  # Check that the cached query did not fail.
        ]
        assert storage.size() == 1

    def test_submit_second_query_not_select(self, athena, proxy, storage):
        """Test that only SELECT queries are cached."""
        athena.submit("CREATE TABLE ...")
        proxy.request_log.clear()
        athena.submit("CREATE TABLE ...")
        assert proxy.request_log == ["StartQueryExecution"]
        assert storage.size() == 0

    def test_submit_second_query_different_sql(self, athena, proxy, storage):
        """Test that cache is unique to a query."""
        athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        proxy.data = ANOTHER_FAKE_DATA
        athena.submit("SELECT 2 id, 'bar' name")
        assert proxy.request_log == ["StartQueryExecution"]
        assert storage.size() == 2

    def test_submit_second_query_first_failed(self, athena, proxy):
        """Test that failed failed queries in are ignored."""
        proxy.state = "FAILED"
        athena.submit("SELECT 1 id, 'foo' name")
        proxy.state = "SUCCEEDED"
        proxy.request_log.clear()
        athena.submit("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "GetQueryExecution",  # Discover that the cached query failed.
            "StartQueryExecution",  # Start a new one.
        ]

    # Test athena.get_query() method

    def test_get_uncached_query_get_results(self, athena, proxy, storage):
        """Test getting query not in cache."""
        athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        query_results = athena.get_query("query-1").get_results()
        assert proxy.request_log == ["GetQueryExecution", "GetQueryResults"]
        assert_query_results(query_results)

    def test_local_uncached_query_get_results_cache_size(self, local_athena, storage):
        """Test that results are not cached."""
        local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.get_query("query-1").get_results()
        assert storage.size() == 2

    def test_remote_uncached_query_get_results_cache_size(self, remote_athena, storage):
        """Test that results are not cached."""
        remote_athena.submit("SELECT 1 id, 'foo' name")
        remote_athena.get_query("query-1").get_results()
        assert storage.size() == 1

    def test_local_get_cached_query_get_results(self, local_athena, proxy, storage):
        """Test getting query in cache when results are cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        query_results = local_athena.get_query("query-1").get_results()
        assert proxy.request_log == []
        assert_query_results(query_results)
        assert storage.size() == 2

    def test_remote_get_cached_query_get_results(self, remote_athena, proxy, storage):
        """Test getting query in cache when results are not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        query_results = remote_athena.get_query("query-1").get_results()
        assert proxy.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(query_results)
        assert storage.size() == 1

    def test_get_query_get_results_not_select(self, athena, storage):
        """Test that results are cached for SELECT queries onlys."""
        athena.submit("CREATE TABLE ...")
        athena.get_query("query-1").get_results()
        assert storage.size() == 0

    # Test query.get_info() method

    def test_query_info(self, athena, proxy):
        """Test obtaining information about a query."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        info = query.get_info()
        assert info.execution_id == "query-1"
        assert info.sql == "SELECT 1 id, 'foo' name"
        assert proxy.request_log == ["GetQueryExecution"]

    # Test query.get_results() method

    def test_get_results_one_query(self, athena, proxy):
        """Test getting results query not in cache."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        query_results = query.get_results()
        assert proxy.request_log == ["GetQueryExecution", "GetQueryResults"]
        assert_query_results(query_results)

    def test_local_get_second_results_one_query(self, local_athena, proxy):
        """Test getting results twice results cached."""
        query = local_athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        proxy.request_log.clear()
        query_results = query.get_results()
        assert proxy.request_log == []
        assert_query_results(query_results)

    def test_remote_get_second_results_one_query(self, remote_athena, proxy):
        """Test getting results twice results not cached."""
        query = remote_athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        proxy.request_log.clear()
        query_results = query.get_results()
        assert proxy.request_log == [
            "GetQueryResults",
        ]
        assert_query_results(query_results)

    def test_local_get_results_second_query_same_sql(self, local_athena, proxy):
        """Test getting results query in cache results cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        second_query_results = second_query.get_results()
        assert proxy.request_log == []
        assert_query_results(second_query_results)

    def test_remote_get_results_second_query_same_sql(self, remote_athena, proxy):
        """Test getting results query in cache results not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        second_query = remote_athena.submit("SELECT 1 id, 'foo' name")
        second_query_results = second_query.get_results()
        assert proxy.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(second_query_results)

    def test_get_results_second_query_different_sql(self, athena, proxy):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")
        proxy.data = ANOTHER_FAKE_DATA
        second_query = athena.submit("SELECT 2 id, 'bar' name")
        proxy.request_log.clear()
        second_query_results = second_query.get_results()
        assert proxy.request_log == ["GetQueryExecution", "GetQueryResults"]
        assert_another_query_results(second_query_results)

    def test_local_get_uncached_results_second_query_same_sql(
        self, local_athena, proxy
    ):
        """Test that the second query downloads data if the first does not."""
        local_athena.submit("SELECT 1 id, 'foo' name")  # Does not download results.
        proxy.request_log.clear()
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        second_query_results = second_query.get_results()
        assert proxy.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(second_query_results)

    def test_local_get_cached_results_first_query_same_sql(self, local_athena, proxy):
        """Test that first query can use cache from the second query."""
        first_query = local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        first_query_results = first_query.get_results()
        assert proxy.request_log == []
        assert_query_results(first_query_results)

    # Test query.kill() method

    def test_kill(self, athena, proxy):
        query = athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        query.kill()
        assert proxy.request_log == ["StopQueryExecution"]

    # Test query.join() method

    def test_join_one_query(self, athena, proxy):
        """Test waiting for a query not cached."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        query.join()
        assert proxy.request_log == ["GetQueryExecution"]

    def test_local_join_second_query_query_same_sql(self, local_athena, proxy):
        """Test waiting for a query cached results not cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        second_query.join()
        assert proxy.request_log == []

    def test_remote_join_second_query_query_same_sql(self, remote_athena, proxy):
        """Test waiting for a query cached results not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        second_query = remote_athena.submit("SELECT 1 id, 'foo' name")
        second_query.join()
        assert proxy.request_log == ["GetQueryExecution"]

    def test_join_second_query_query_different_sql(self, athena, proxy):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")
        proxy.data = ANOTHER_FAKE_DATA
        second_query = athena.submit("SELECT 2 id, 'bar' name")
        proxy.request_log.clear()
        second_query.join()
        assert proxy.request_log == ["GetQueryExecution"]

    # Test configuration

    def test_cache_disabled_first_query(self, athena, proxy, storage):
        athena.cache.enabled = False
        athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert storage.size() == 0

    def test_cache_disabled_second_query(self, athena, proxy):
        athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        athena.cache.enabled = False
        athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]

    def test_local_cache_read_disabled_first_query(self, local_athena, proxy, storage):
        local_athena.cache.read = False
        local_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert storage.size() == 2

    def test_remote_cache_read_disabled_first_query(
        self, remote_athena, proxy, storage
    ):
        remote_athena.cache.read = False
        remote_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert storage.size() == 1

    def test_cache_read_disabled_second_query(self, athena, proxy):
        athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        athena.cache.read = False
        athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]

    def test_cache_write_disabled_first_query(self, athena, proxy, storage):
        athena.cache.write = False
        athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert storage.size() == 0

    def test_local_cache_write_disabled_second_query(self, local_athena, proxy):
        local_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        local_athena.cache.write = False
        local_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == []

    def test_remote_cache_write_disabled_second_query(self, remote_athena, proxy):
        remote_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        remote_athena.cache.write = False
        remote_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == ["GetQueryExecution", "GetQueryResults"]

    def test_cached_remotely_only(self, full_athena, proxy):
        """
        Test that remote cache is used to populate local cache.
        """
        # Another client populates remote cache.
        another_athena = Athena(proxy)
        another_athena.cache.remote_storage = full_athena.cache.remote_storage
        another_athena.execute("SELECT 1 id, 'foo' name")
        proxy.request_log.clear()
        # First execution uses remote cache
        full_athena.execute("SELECT 1 id, 'foo' name")
        assert proxy.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert full_athena.cache.local_storage.size() == 2
