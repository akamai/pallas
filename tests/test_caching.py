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

from pallas.caching import AthenaCachingWrapper, is_cacheable
from pallas.storage import MemoryStorage
from pallas.testing import AthenaFake

FAKE_DATA = [("1", "foo", None)]
ANOTHER_FAKE_DATA = [("2", "bar", None)]


@pytest.fixture(name="fake")
def fake_athena_fixture():
    """
    Athena mock decorated by caching wrapper.
    """
    fake = AthenaFake()
    fake.column_names = "id", "name", "value"
    fake.column_types = "integer", "varchar", "double"
    fake.data = FAKE_DATA
    return fake


@pytest.fixture(name="remote_athena")
def caching_athena_fixture(fake):
    """
    Caching wrapper in remote mode.

    In remote mode, only query execution IDs are cached.
    """
    storage = MemoryStorage()
    return AthenaCachingWrapper(fake, storage=storage, cache_results=False)


@pytest.fixture(name="local_athena")
def local_athena_fixture(fake):
    """
    Caching wrapper in local mode.

    In local mode, both query execution IDs and results are cached.
    """
    storage = MemoryStorage()
    return AthenaCachingWrapper(fake, storage=storage)


@pytest.fixture(name="athena", params=["remote", "local"])
def athena_fixture(request):
    """
    Yields caching wrapper in both remote and local mode.
    """
    return request.getfixturevalue(f"{request.param}_athena")


def assert_query_results(results):
    assert list(results) == [{"id": 1, "name": "foo", "value": None}]


def assert_another_query_results(results):
    assert list(results) == [{"id": 2, "name": "bar", "value": None}]


class TestIsCacheable:
    def test_select(self):
        assert is_cacheable("SELECT 1")

    def test_with_select(self):
        assert is_cacheable("WITH (...) AS t SELECT ...")

    def test_insert(self):
        assert not is_cacheable("INSERT ... AS SELECT")

    def test_create(self):
        assert not is_cacheable("CREATE TABLE AS ... SELECT")

    def test_select_without_whitesplace(self):
        assert is_cacheable("SELECT*FROM ...")

    def test_single_line_comments(self):
        assert is_cacheable(
            """
            -- Comment 1
            -- Comment 2
            SELECT
        """
        )

    def test_multi_line_comments(self):
        assert is_cacheable(
            """
            /*
            Comment 1
            Comment 2
            */
            SELECT
        """
        )

    def test_multi_line_comment_escaped(self):
        assert is_cacheable(
            r"""
            /* *\/ */
            SELECT
        """
        )


class TestAthenaCachingWrapper:

    # Test execute method

    def test_repr(self, athena):
        assert (
            repr(athena) == "<AthenaCachingWrapper: <AthenaFake> cached at 'memory:'>"
        )

    def test_remote_query_repr(self, remote_athena):
        query = remote_athena.submit("SELECT 1")
        assert repr(query) == "<QueryFake: execution_id='query-1'>"

    def test_local_query_repr(self, local_athena):
        query = local_athena.submit("SELECT 1")
        assert repr(query) == (
            "<QueryCachingWrapper: <QueryFake: execution_id='query-1'>>"
        )

    def test_execute_one_query(self, athena):
        """Test execution of a query not in cache."""
        results = athena.execute("SELECT 1 id, 'foo' name")
        assert athena.wrapped.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_remote_execute_one_query_cache_size(self, remote_athena):
        """Test that query ID is written to cache."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        assert remote_athena.storage.size() == 1

    def test_local_execute_one_query_cache_size(self, local_athena):
        """Test that query ID and results are written to cache."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        assert local_athena.storage.size() == 2

    def test_execute_one_query_not_select(self, athena):
        """Test execution of a query that should not be cached."""
        results = athena.execute("CREATE TABLE ...")
        assert athena.wrapped.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)
        assert athena.storage.size() == 0

    def test_remote_execute_second_query_same_sql(self, remote_athena):
        """Test execution of a query in cache."""
        remote_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        remote_athena.wrapped.request_log.clear()
        results = remote_athena.execute("SELECT 1 id, 'foo' name")
        assert remote_athena.wrapped.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)
        assert remote_athena.storage.size() == 1

    def test_local_execute_second_query(self, local_athena):
        """Test execution of a query in cache."""
        local_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        local_athena.wrapped.request_log.clear()
        results = local_athena.execute("SELECT 1 id, 'foo' name")
        assert local_athena.wrapped.request_log == []
        assert_query_results(results)
        assert local_athena.storage.size() == 2

    def test_execute_second_query_not_select(self, athena):
        """Test that only SELECT queries are cached."""
        athena.execute("CREATE TABLE ...")
        athena.wrapped.request_log.clear()
        results = athena.execute("CREATE TABLE ...")
        assert athena.wrapped.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)
        assert athena.storage.size() == 0

    def test_execute_second_query_ignore_cache(self, athena):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        athena.wrapped.request_log.clear()
        results = athena.execute("SELECT 1 id, 'foo' name", ignore_cache=True)
        assert athena.wrapped.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_execute_second_query_different_sql(self, athena):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        athena.wrapped.request_log.clear()
        athena.wrapped.data = ANOTHER_FAKE_DATA
        results = athena.execute("SELECT 2 id, 'bar' name")  # different SQL
        assert athena.wrapped.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_another_query_results(results)

    def test_execute_second_query_different_database(self, athena):
        """Test that cache is unique to a database."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        athena.wrapped.request_log.clear()
        athena.wrapped.database = "other"
        results = athena.execute("SELECT 1 id, 'foo' name")
        assert athena.wrapped.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    # Test athena.submit() method

    def test_submit_one_query(self, athena):
        """Test that the caching wrapper submits a query if not in cache."""
        athena.submit("SELECT 1 id, 'foo' name")
        assert athena.wrapped.request_log == ["StartQueryExecution"]
        assert athena.storage.size() == 1

    def test_submit_one_query_not_select(self, athena):
        """Test that the caching wrapper submits a query if not in cache."""
        athena.submit("CREATE TABLE ...")
        assert athena.wrapped.request_log == ["StartQueryExecution"]
        assert athena.storage.size() == 0

    def test_submit_second_query(self, athena):
        """Test that one query is submitted only once."""
        athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        athena.submit("SELECT 1 id, 'foo' name")
        assert athena.wrapped.request_log == []
        assert athena.storage.size() == 1

    def test_submit_second_query_not_select(self, athena):
        """Test that only SELECT queries are cached."""
        athena.submit("CREATE TABLE ...")
        athena.wrapped.request_log.clear()
        athena.submit("CREATE TABLE ...")
        assert athena.wrapped.request_log == ["StartQueryExecution"]
        assert athena.storage.size() == 0

    def test_submit_second_query_ignore_cache(self, athena):
        """Test that cache read can be skipped."""
        athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        athena.submit("SELECT 1 id, 'foo' name", ignore_cache=True)
        assert athena.wrapped.request_log == ["StartQueryExecution"]
        assert athena.storage.size() == 1

    def test_submit_second_query_different_sql(self, athena):
        """Test that cache is unique to a query."""
        athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        athena.wrapped.data = ANOTHER_FAKE_DATA
        athena.submit("SELECT 2 id, 'bar' name")
        assert athena.wrapped.request_log == ["StartQueryExecution"]
        assert athena.storage.size() == 2

    # Test athena.get_query() method

    def test_get_uncached_query_get_results(self, athena):
        """Test getting query not in cache."""
        athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        query_results = athena.get_query("query-1").get_results()
        assert athena.wrapped.request_log == ["GetQueryExecution", "GetQueryResults"]
        assert_query_results(query_results)

    def test_remote_uncached_query_get_results_cache_size(self, remote_athena):
        """Test that results are not cached."""
        remote_athena.submit("SELECT 1 id, 'foo' name")
        remote_athena.get_query("query-1").get_results()
        assert remote_athena.storage.size() == 1

    def test_local_uncached_query_get_results_cache_size(self, local_athena):
        """Test that results are not cached."""
        local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.get_query("query-1").get_results()
        assert local_athena.storage.size() == 2

    def test_remote_get_cached_query_get_results(self, remote_athena):
        """Test getting query in cache when results are not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        remote_athena.wrapped.request_log.clear()
        query_results = remote_athena.get_query("query-1").get_results()
        assert remote_athena.wrapped.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(query_results)
        assert remote_athena.storage.size() == 1

    def test_local_get_cached_query_get_results(self, local_athena):
        """Test getting query in cache when results are cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        local_athena.wrapped.request_log.clear()
        query_results = local_athena.get_query("query-1").get_results()
        assert local_athena.wrapped.request_log == []
        assert_query_results(query_results)
        assert local_athena.storage.size() == 2

    def test_get_query_get_results_not_select(self, athena):
        """Test that results are cached for SELECT queries onlys."""
        athena.submit("CREATE TABLE ...")
        athena.get_query("query-1").get_results()
        assert athena.storage.size() == 0

    # Test query.get_info() method

    def test_query_info(self, athena):
        """Test obtaining information about a query."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        info = query.get_info()
        assert info.execution_id == "query-1"
        assert info.sql == "SELECT 1 id, 'foo' name"
        assert athena.wrapped.request_log == ["GetQueryExecution"]

    # Test query.get_results() method

    def test_get_results_one_query(self, athena):
        """Test getting results query not in cache."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        query_results = query.get_results()
        assert athena.wrapped.request_log == ["GetQueryExecution", "GetQueryResults"]
        assert_query_results(query_results)

    def test_remote_get_second_results_one_query(self, remote_athena):
        """Test getting results twice results not cached."""
        query = remote_athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        remote_athena.wrapped.request_log.clear()
        query_results = query.get_results()
        assert remote_athena.wrapped.request_log == [
            "GetQueryResults",
        ]
        assert_query_results(query_results)

    def test_local_get_second_results_one_query(self, local_athena):
        """Test getting results twice results cached."""
        query = local_athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        local_athena.wrapped.request_log.clear()
        query_results = query.get_results()
        assert local_athena.wrapped.request_log == []
        assert_query_results(query_results)

    def test_remote_get_results_second_query_same_sql(self, remote_athena):
        """Test getting results query in cache results not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        second_query = remote_athena.submit("SELECT 1 id, 'foo' name")
        remote_athena.wrapped.request_log.clear()
        second_query_results = second_query.get_results()
        assert remote_athena.wrapped.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(second_query_results)

    def test_local_get_results_second_query_same_sql(self, local_athena):
        """Test getting results query in cache results cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.wrapped.request_log.clear()
        second_query_results = second_query.get_results()
        assert local_athena.wrapped.request_log == []
        assert_query_results(second_query_results)

    def test_get_results_second_query_different_sql(self, athena):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")
        athena.wrapped.data = ANOTHER_FAKE_DATA
        second_query = athena.submit("SELECT 2 id, 'bar' name")
        athena.wrapped.request_log.clear()
        second_query_results = second_query.get_results()
        assert athena.wrapped.request_log == ["GetQueryExecution", "GetQueryResults"]
        assert_another_query_results(second_query_results)

    def test_local_get_uncached_results_second_query_same_sql(self, local_athena):
        """Test that the second query downloads data if the first does not."""
        local_athena.submit("SELECT 1 id, 'foo' name")  # Does not download results.
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.wrapped.request_log.clear()
        second_query_results = second_query.get_results()
        assert local_athena.wrapped.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(second_query_results)

    def test_local_get_cached_results_first_query_same_sql(self, local_athena):
        """Test that first query can use cache from the second query."""
        first_query = local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.execute("SELECT 1 id, 'foo' name")
        local_athena.wrapped.request_log.clear()
        first_query_results = first_query.get_results()
        assert local_athena.wrapped.request_log == []
        assert_query_results(first_query_results)

    # Test query.kill() method

    def test_kill(self, athena):
        query = athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        query.kill()
        assert athena.wrapped.request_log == ["StopQueryExecution"]

    # Test query.join() method

    def test_join_one_query(self, athena):
        """Test waiting for a query not cached."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        athena.wrapped.request_log.clear()
        query.join()
        assert athena.wrapped.request_log == ["GetQueryExecution"]

    def test_remote_join_second_query_query_same_sql(self, remote_athena):
        """Test waiting for a query cached results not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        second_query = remote_athena.submit("SELECT 1 id, 'foo' name")
        remote_athena.wrapped.request_log.clear()
        second_query.join()
        assert remote_athena.wrapped.request_log == ["GetQueryExecution"]

    def test_local_join_second_query_query_same_sql(self, local_athena):
        """Test waiting for a query cached results not cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.wrapped.request_log.clear()
        second_query.join()
        assert local_athena.wrapped.request_log == []

    def test_join_second_query_query_different_sql(self, athena):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")
        athena.wrapped.data = ANOTHER_FAKE_DATA
        second_query = athena.submit("SELECT 2 id, 'bar' name")
        athena.wrapped.request_log.clear()
        second_query.join()
        assert athena.wrapped.request_log == ["GetQueryExecution"]
