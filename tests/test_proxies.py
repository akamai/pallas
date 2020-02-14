import itertools
import textwrap

import pytest

from pallas.exceptions import AthenaQueryError
from pallas.proxies import AthenaProxy


@pytest.fixture
def athena(region_name, athena_database, s3_tmp_uri):
    return AthenaProxy(
        region_name=region_name,
        database=athena_database,
        output_location=f"{s3_tmp_uri}/output",
    )


class TestAthenaProxy:
    def test_success(self, athena, athena_database):
        query = athena.submit("SELECT 1")
        # Running
        info = query.get_info()
        assert not info.finished
        assert not info.succeeded
        assert info.database == athena_database
        assert info.sql == "SELECT 1"
        assert info.state in ("QUEUED", "RUNNING")
        # Finished
        query.join()
        info = query.get_info()
        assert info.finished
        assert info.succeeded
        assert info.database == athena_database
        assert info.sql == "SELECT 1"
        assert info.state == "SUCCEEDED"

    def test_fail(self, athena, athena_database):
        query = athena.submit("SELECT x")
        with pytest.raises(AthenaQueryError) as excinfo:
            query.join()
        assert str(excinfo.value).startswith("Athena query failed: SYNTAX_ERROR: ")
        info = query.get_info()
        assert info.finished
        assert not info.succeeded
        assert info.state == "FAILED"
        assert info.database == athena_database
        assert info.sql == "SELECT x"

    def test_kill(self, athena, athena_database):
        query = athena.submit("SELECT 1")
        query.kill()
        with pytest.raises(AthenaQueryError) as excinfo:
            query.join()
        assert str(excinfo.value) == "Athena query cancelled: Query cancelled by user"
        info = query.get_info()
        assert info.finished
        assert not info.succeeded
        assert info.state == "CANCELLED"
        assert info.database == athena_database
        assert info.sql == "SELECT 1"

    def test_variaous_results(self, athena):
        sql = """\
            SELECT
                'anonymous',
                null unknown_null,
                true boolean_true,
                false boolean_false,
                cast(null AS BOOLEAN) boolean_null,
                CAST(1 as TINYINT) tinyint_value,
                CAST(2 as SMALLINT) smallint_value,
                CAST(3 as INTEGER) integer_value,
                CAST(4 as BIGINT) bigint_value,
                CAST(null as INTEGER) integer_null,
                CAST(0.1 as REAL) real_value,
                CAST(0.2 as DOUBLE) double_value,
                CAST(null as DOUBLE) double_null,
                nan() double_nan,
                infinity() double_plus_infinity,
                -infinity() double_minus_infinity,
                CAST('a' as CHAR) char_value,
                CAST(NULL as CHAR) char_null,
                CAST('b' as VARCHAR) varchar_value,
                CAST(NULL as VARCHAR) varchar_null
        """
        results = athena.execute(textwrap.dedent(sql))
        assert list(results) == [
            {
                "_col0": "anonymous",
                "unknown_null": None,
                "boolean_true": True,
                "boolean_false": False,
                "boolean_null": None,
                "tinyint_value": 1,
                "smallint_value": 2,
                "integer_value": 3,
                "bigint_value": 4,
                "integer_null": None,
                "real_value": 0.1,
                "double_value": 0.2,
                "double_null": None,
                "double_nan": pytest.approx(float("nan"), nan_ok=True),
                "double_plus_infinity": float("inf"),
                "double_minus_infinity": float("-inf"),
                "char_value": "a",
                "char_null": None,
                "varchar_value": "b",
                "varchar_null": None,
            }
        ]

    def test_empty_results(self, athena):
        sql = "SELECT * FROM (VALUES (1, 'a')) AS t (id, name) WHERE id < 0"
        results = athena.execute(sql)
        assert list(results) == []

    def test_long_results(self, athena):
        r1, r2 = range(20), range(100)
        sql = f"""\
            SELECT * FROM
                (VALUES {', '.join(map(str, r1))}) AS t1 (v1),
                (VALUES {', '.join(map(str, r2))}) AS t2 (v2)
            ORDER BY
                v1, v2
        """
        results = athena.execute(textwrap.dedent(sql))
        assert list(results) == [
            {"v1": v1, "v2": v2} for v1, v2 in itertools.product(r1, r2)
        ]
