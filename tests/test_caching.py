import pytest

from pallas.caching import AthenaCachingWrapper
from pallas.storage import MemoryStorage
from pallas.testing import AthenaFake


FAKE_DATA = [("1", "foo")]
ANOTHER_FAKE_DATA = [("2", "bar")]


@pytest.fixture(name="fake")
def fake_athena_fixture():
    fake = AthenaFake()
    fake.column_names = "id", "name"
    fake.column_types = "integer", "varchar"
    fake.data = FAKE_DATA
    return fake


@pytest.fixture(name="athena")
def caching_athena_fixture(fake):
    cache = MemoryStorage()
    return AthenaCachingWrapper(fake, cache=cache)


def assert_query_results(results):
    assert list(results) == [{"id": 1, "name": "foo"}]


def assert_another_query_results(results):
    assert list(results) == [{"id": 2, "name": "bar"}]


class TestAthenaCachingDecorator:

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

    def test_execute_second_query_same_sql(self, athena, fake):
        """Test execution of a query in cache."""
        athena.execute("SELECT 1 id, 'foo' name")  # fill cache
        fake.request_log.clear()
        results = athena.execute("SELECT 1 id, 'foo' name")
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

    def test_submit_second_query_same_sql(self, athena, fake):
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

    def test_get_uncached_results_by_execution_id(self, athena, fake):
        """Test that getting results not in cache by execution_id."""
        athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = athena.get_query("query-1").get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(query_results)

    def test_get_cached_results_by_execution_id(self, athena, fake):
        """Test that getting results in cache by execution_id."""
        athena.execute("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = athena.get_query("query-1").get_results()
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
        """Test getting results not in cache."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query_results = query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(query_results)

    def test_get_second_results_one_query(self, athena, fake):
        """Test getting results in cache."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        query.get_results()
        fake.request_log.clear()
        query_results = query.get_results()
        assert fake.request_log == []
        assert_query_results(query_results)

    def test_get_results_second_query_same_sql(self, athena, fake):
        """Test that the results for the second query are read from cache."""
        athena.execute("SELECT 1 id, 'foo' name")
        second_query = athena.submit("SELECT 1 id, 'foo' name")
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

    def test_get_uncached_results_second_query_same_sql(self, athena, fake):
        """Test that the second query downloads data if the first does not."""
        athena.submit("SELECT 1 id, 'foo' name")  # Does not download results.
        second_query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        second_query_results = second_query.get_results()
        assert fake.request_log == ["GetQueryResults"]
        assert_query_results(second_query_results)

    def test_get_cached_results_first_query_same_sql(self, athena, fake):
        """Test that first query can use cache from the second query."""
        first_query = athena.submit("SELECT 1 id, 'foo' name")
        athena.execute("SELECT 1 id, 'foo' name")
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
        """Test waiting for a query not in cache."""
        query = athena.submit("SELECT 1 id, 'foo' name")
        fake.request_log.clear()
        query.join()
        assert fake.request_log == ["GetQueryExecution"]

    def test_join_second_query_query_same_sql(self, athena, fake):
        """Test waiting for a query in cache."""
        athena.execute("SELECT 1 id, 'foo' name")
        second_query = athena.submit("SELECT 1 id, 'foo' name")
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
