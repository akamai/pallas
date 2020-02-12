class TestQueryFake:
    def test_submit(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        assert query.execution_id == "query-0"
        assert fake_athena.queries == [query]

    def test_submit_multiple(self, fake_athena):
        query1 = fake_athena.submit("SELECT ...")
        query2 = fake_athena.submit("SELECT ...")
        assert query1.execution_id == "query-0"
        assert query2.execution_id == "query-1"
        assert fake_athena.queries == [query1, query2]

    def test_get_query(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        assert fake_athena.get_query("query-0") is query

    def test_query_info(self, fake_athena):
        query = fake_athena.submit("SELECT ...")
        info = query.get_info()
        assert info.execution_id == "query-0"
        assert info.sql == "SELECT ..."
        assert info.database is None
        assert info.finished
        assert info.succeeded

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
