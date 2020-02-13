import pytest

from pallas.caching.backends import MemoryCache
from pallas.caching.wrappers import AthenaCachingWrapper
from pallas.testing import AthenaFake


@pytest.fixture(name="fake_athena")
def fake_athena_fixture():
    fake_athena = AthenaFake()
    fake_athena.column_names = "id", "name"
    fake_athena.column_types = "integer", "varchar"
    fake_athena.data = [("1", "foo")]
    return fake_athena


@pytest.fixture(name="caching_athena")
def caching_athena_fixture(fake_athena):
    cache = MemoryCache()
    return AthenaCachingWrapper(fake_athena, cache=cache)


class TestAthenaCachingDecorator:
    def test_submit_same_sql(self, caching_athena, fake_athena):
        query1 = caching_athena.submit("SELECT 1")
        query2 = caching_athena.submit("SELECT 1")
        assert query1.execution_id == "query-1"
        assert query2.execution_id == "query-1"
        assert fake_athena.get_call_count("StartQueryExecution") == 1

    def test_submit_different_sql(self, caching_athena, fake_athena):
        query1 = caching_athena.submit("SELECT 1")
        query2 = caching_athena.submit("SELECT 2")
        assert query1.execution_id == "query-1"
        assert query2.execution_id == "query-2"
        assert fake_athena.get_call_count("StartQueryExecution") == 2

    def test_execute_same_sql(self, caching_athena, fake_athena):
        results1 = caching_athena.execute("SELECT 1")
        results2 = caching_athena.execute("SELECT 1")
        assert fake_athena.get_call_count("StartQueryExecution") == 1
        assert list(results1) == list(results2) == [{"id": 1, "name": "foo"}]
        assert fake_athena.get_call_count("GetQueryResults") == 1

    def test_execute_different_sql(self, caching_athena, fake_athena):
        query1 = caching_athena.submit("SELECT 1")
        query2 = caching_athena.submit("SELECT 2")
        assert query1.execution_id == "query-1"
        assert query2.execution_id == "query-2"
        assert fake_athena.get_call_count("StartQueryExecution") == 2

    def test_query_info(self, caching_athena, fake_athena):
        query = caching_athena.submit("SELECT ...")
        info = query.get_info()
        assert info.execution_id == "query-1"
        assert info.sql == "SELECT ..."
        assert fake_athena.get_call_count("GetQueryExecution") == 1

    def test_get_results_one_query(self, caching_athena, fake_athena):
        query = caching_athena.submit("SELECT 1 id, 'foo' name")
        results1 = query.get_results()
        results2 = query.get_results()
        assert list(results1) == list(results2) == [{"id": 1, "name": "foo"}]
        assert fake_athena.get_call_count("GetQueryResults") == 1

    def test_get_results_same_sql(self, caching_athena, fake_athena):
        results1 = caching_athena.submit("SELECT 1 id, 'foo' name").get_results()
        results2 = caching_athena.submit("SELECT 1 id, 'foo' name").get_results()
        assert list(results1) == list(results2) == [{"id": 1, "name": "foo"}]
        assert fake_athena.get_call_count("GetQueryResults") == 1

    def test_get_results_different_sql(self, caching_athena, fake_athena):
        results1 = caching_athena.submit("SELECT 1 id, 'foo' name").get_results()
        fake_athena.data = [("2", "bar")]
        results2 = caching_athena.submit("SELECT 2 id, 'bar' name").get_results()
        assert list(results1) == [{"id": 1, "name": "foo"}]
        assert list(results2) == [{"id": 2, "name": "bar"}]
        assert fake_athena.get_call_count("GetQueryResults") == 2

    def test_get_results_same_sql_out_of_order(self, caching_athena, fake_athena):
        query1 = caching_athena.submit("SELECT 1 id, 'foo' name")
        results2 = caching_athena.submit("SELECT 1 id, 'foo' name").get_results()
        results1 = query1.get_results()
        assert list(results1) == list(results2) == [{"id": 1, "name": "foo"}]
        assert fake_athena.get_call_count("GetQueryResults") == 1

    def test_get_query_and_get_results(self, caching_athena, fake_athena):
        results1 = caching_athena.submit("SELECT 1 id, 'foo' name").get_results()
        results2 = caching_athena.get_query("query-1").get_results()
        assert list(results1) == list(results2) == [{"id": 1, "name": "foo"}]
        assert fake_athena.get_call_count("GetQueryResults") == 1

    def test_kill(self, caching_athena, fake_athena):
        query = caching_athena.submit("SELECT ...")
        query.kill()
        assert fake_athena.get_call_count("StartQueryExecution") == 1
        assert fake_athena.get_call_count("get_query_execution") == 0
        assert fake_athena.get_call_count("GetQueryResults") == 0
        assert fake_athena.get_call_count("StopQueryExecution") == 1
