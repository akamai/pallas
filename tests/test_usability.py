import pytest

from pallas.usability import AthenaNormalizationWrapper, normalize_sql
from pallas.testing import AthenaFake

NORMALIZED_SQL = """\
SELECT
    c1, c2
FROM
    t\
"""


class TestNormalizeSQL:
    def test_dedent(self):
        sql = normalize_sql(
            """
            SELECT
                c1, c2
            FROM
                t
        """
        )
        assert sql == NORMALIZED_SQL

    def test_trailing_whitespace(self):
        sql = normalize_sql(
            """\N{space}
            SELECT\N{space}\N{space}\N{space}
                c1, c2\N{space}
            FROM\N{space}
                t\N{space}
        """
        )
        assert sql == NORMALIZED_SQL

    def test_empty_lines(self):
        sql = normalize_sql(
            """\n
            SELECT\n\n\n
                c1, c2\n
            FROM\n
                t\n
        """
        )
        assert sql == NORMALIZED_SQL


@pytest.fixture(name="athena")
def athena_fixture():
    fake = AthenaFake()
    fake.database = "test_database"
    return AthenaNormalizationWrapper(fake)


class TestAthenaNormalizationWrapper:
    def test_repr(self, athena):
        assert repr(athena) == "<AthenaNormalizationWrapper: <AthenaFake>>"

    def test_query_repr(self, athena):
        query = athena.submit("SELECT 1")
        assert repr(query) == "<QueryFake: execution_id='query-1'>"

    def test_database(self, athena):
        assert athena.database == "test_database"

    def test_submit(self, athena):
        query = athena.submit(
            """
            SELECT
                c1, c2
            FROM
                t
        """
        )
        assert query.get_info().sql == NORMALIZED_SQL

    def test_get_query(self, athena):
        athena.submit(
            """
            SELECT
                c1, c2
            FROM
                t
        """
        )
        query = athena.get_query("query-1")
        assert query.get_info().sql == NORMALIZED_SQL
