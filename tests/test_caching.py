import pytest

from pallas.caching import AthenaCachingWrapper
from pallas.storage import MemoryStorage
from pallas.testing import AthenaFake


FAKE_DATA = [("1", "foo")]
ANOTHER_FAKE_DATA = [("2", "bar")]


@pytest.fixture(name="fake")
def fake_athena_fixture():
    """
    Athena mock decorated by caching wrapper.
    """
    fake = AthenaFake()
    fake.column_names = "id", "name"
    fake.column_types = "integer", "varchar"
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
    assert list(results) == [{"id": 1, "name": "foo"}]


def assert_another_query_results(results):
    assert list(results) == [{"id": 2, "name": "bar"}]


class TestAthenaCachingWrapper:

    # Test execute method

    def test_execute_one_query(self, athena, fake):
        """Test execution of a query not in cache."""
        results = athena.execute("SELECT 1 id, 'foo' name")
        assert fake.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_remote_execute_second_query_same_sql(self, remote_athena, fake):
        """Test execution of a query in cache."""
        remote_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        fake.request_log.clear()
        results = remote_athena.execute("SELECT 1 id, 'foo' name")
        assert fake.request_log == [
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_query_results(results)

    def test_local_execute_second_query_same_sql(self, local_athena, fake):
        """Test execution of a query in cache."""
        local_athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        fake.request_log.clear()
        results = local_athena.execute("SELECT 1 id, 'foo' name")
        assert fake.request_log == []
        assert_query_results(results)

    def test_execute_second_query_different_sql(self, athena, fake):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        fake.request_log.clear()
        fake.data = ANOTHER_FAKE_DATA
        results = athena.execute("SELECT 2 id, 'bar' name")  # different SQL
        assert fake.request_log == [
            "StartQueryExecution",
            "GetQueryExecution",
            "GetQueryResults",
        ]
        assert_another_query_results(results)

    # Test athena.submit() method

    def test_submit_one_query(self, athena, fake):
        """Test that the caching wrapper submits a query if not in cache."""
        athena.submit("SELECT 1 id, 'foo' name")
        assert fake.request_log == ["StartQueryExecution"]

    def test_local_submit_second_query_same_sql(self, athena, fake):
        """Test that one query is submitted only once."""
        athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        athena.submit("SELECT 1 id, 'foo' name")
        assert fake.request_log == []

    def test_submit_second_query_different_sql(self, athena, fake):
        """Test that cache is unique to a query."""
        athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        fake.data = ANOTHER_FAKE_DATA
        athena.submit("SELECT 2 id, 'bar' name")
        assert fake.request_log == ["StartQueryExecution"]

    # Test athena.get_query() method

    def test_get_uncached_query_get_results(self, athena, fake):
        """Test getting query not in cache."""
        athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = athena.get_query("query-1").get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(query_results)

    def test_remote_get_cached_query_get_results(self, remote_athena, fake):
        """Test getting query in cache when results are not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = remote_athena.get_query("query-1").get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(query_results)

    def test_local_get_cached_query_get_results(self, local_athena, fake):
        """Test getting query in cache when results are cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = local_athena.get_query("query-1").get_results()
        assert fake.request_log == []
        assert_query_results(query_results)

    # Test query.get_info() method

    def test_query_info(self, athena, fake):
        """Test obtaining information about a query."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        info = query.get_info()
        assert info.execution_id == "query-1"
        assert info.sql == "SELECT 1 id, 'foo' name"
        assert fake.request_log == ["GetQueryExecution"]

    # Test query.get_results() method

    def test_get_results_one_query(self, athena, fake):
        """Test getting results query not in cache."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(query_results)

    def test_remote_get_second_results_one_query(self, remote_athena, fake):
        """Test getting results twice results not cached."""
        query = remote_athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        fake.request_log.clear()
        query_results = query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(query_results)

    def test_local_get_second_results_one_query(self, local_athena, fake):
        """Test getting results twice results cached."""
        query = local_athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        fake.request_log.clear()
        query_results = query.get_results()
        assert fake.request_log == []
        assert_query_results(query_results)

    def test_remote_get_results_second_query_same_sql(self, remote_athena, fake):
        """Test getting results query in cache results not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        second_query = remote_athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        second_query_results = second_query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(second_query_results)

    def test_local_get_results_second_query_same_sql(self, local_athena, fake):
        """Test getting results query in cache results cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        second_query_results = second_query.get_results()
        assert fake.request_log == []
        assert_query_results(second_query_results)

    def test_get_results_second_query_different_sql(self, athena, fake):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")
        fake.data = ANOTHER_FAKE_DATA
        second_query = athena.submit("SELECT 2 id, 'bar' name")
        fake.request_log.clear()
        second_query_results = second_query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_another_query_results(second_query_results)

    def test_local_get_uncached_results_second_query_same_sql(self, local_athena, fake):
        """Test that the second query downloads data if the first does not."""
        local_athena.submit("SELECT 1 id, 'foo' name")  # Does not download results.
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        second_query_results = second_query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(second_query_results)

    def test_local_get_cached_results_first_query_same_sql(self, local_athena, fake):
        """Test that first query can use cache from the second query."""
        first_query = local_athena.submit("SELECT 1 id, 'foo' name")
        local_athena.execute("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        first_query_results = first_query.get_results()
        assert fake.request_log == []
        assert_query_results(first_query_results)

    # Test query.kill() method

    def test_kill(self, athena, fake):
        query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query.kill()
        assert fake.request_log == ["StopQueryExecution"]

    # Test query.join() method

    def test_join_one_query(self, athena, fake):
        """Test waiting for a query not cached."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query.join()
        assert fake.request_log == ["GetQueryExecution"]

    def test_remote_join_second_query_query_same_sql(self, remote_athena, fake):
        """Test waiting for a query cached results not cached."""
        remote_athena.execute("SELECT 1 id, 'foo' name")
        second_query = remote_athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        second_query.join()
        assert fake.request_log == ["GetQueryExecution"]

    def test_local_join_second_query_query_same_sql(self, local_athena, fake):
        """Test waiting for a query cached results not cached."""
        local_athena.execute("SELECT 1 id, 'foo' name")
        second_query = local_athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        second_query.join()
        assert fake.request_log == []

    def test_join_second_query_query_different_sql(self, athena, fake):
        """Test that cache is unique to a query."""
        athena.execute("SELECT 1 id, 'foo' name")
        fake.data = ANOTHER_FAKE_DATA
        second_query = athena.submit("SELECT 2 id, 'bar' name")
        fake.request_log.clear()
        second_query.join()
        assert fake.request_log == ["GetQueryExecution"]
