import itertools
import os
import textwrap

import pytest


class TestAthena:
    def test_submit(self, athena):
        query = athena.submit("SELECT 1")
        info = query.get_info()
        assert info.state == "RUNNING"
        assert not info.done
        assert not info.succeeded
        assert info.sql == "SELECT 1"
        assert info.database == os.environ["TEST_PALLAS_DATABASE"]

    def test_execute(self, athena):
        query = athena.submit("SELECT 1")
        info = query.join()
        assert info.done
        assert info.succeeded
        assert info.state == "SUCCEEDED"
        assert info.sql == "SELECT 1"
        assert info.database == os.environ["TEST_PALLAS_DATABASE"]

    def test_kill(self, athena):
        query = athena.submit("SELECT 1")
        query.kill()
        info = query.get_info()
        assert info.done
        assert not info.succeeded
        assert info.state == "CANCELLED"
        assert info.sql == "SELECT 1"
        assert info.database == os.environ["TEST_PALLAS_DATABASE"]

    def test_select_wo_column_name(self, athena):
        sql = "SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c'))"
        results = athena.execute(sql)
        assert list(results) == [
            {"_col0": 1, "_col1": "a"},
            {"_col0": 2, "_col1": "b"},
            {"_col0": 3, "_col1": "c"},
        ]

    def test_select_w_column_name(self, athena):
        sql = "SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t (id, name)"
        results = athena.execute(sql)
        assert list(results) == [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
            {"id": 3, "name": "c"},
        ]

    def test_conversions(self, athena):
        sql = """\
            SELECT
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
