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

import datetime as dt
from decimal import Decimal

import numpy as np
import pytest

from pallas import Athena
from pallas.sql import is_select, normalize_sql, substitute_parameters


@pytest.mark.parametrize(
    "value,quoted",
    [
        (None, "NULL"),
        ("", "''"),
        ("hello", "'hello'"),
        ("'", "''''"),
        (42, "42"),
        (np.int64(42), "42"),
        (3.14, "3.14"),
        (float("nan"), "nan()"),
        (float("inf"), "infinity()"),
        (float("-inf"), "-infinity()"),
        (np.float64(3.14), "3.14"),
        (np.float64("nan"), "nan()"),
        (np.float64("inf"), "infinity()"),
        (np.float64("-inf"), "-infinity()"),
        (np.nan, "nan()"),
        (np.inf, "infinity()"),
        (-np.inf, "-infinity()"),
        (Decimal("0.1"), "DECIMAL '0.1'"),
        (dt.date(2001, 8, 22), "DATE '2001-08-22'"),
        (
            dt.datetime(2001, 8, 22, 3, 4, 5, 321000),
            "TIMESTAMP '2001-08-22 03:04:05.321'",
        ),
        (b"\x00\xff\x00", "from_base64('AP8A')"),
    ],
)
def test_quote_value(value, quoted):
    assert Athena.quote(value) == quoted


@pytest.mark.parametrize("value", [object(), (), [], {}])
def test_quote_invalid(value):
    with pytest.raises(TypeError):
        Athena.quote(value)


class TestSubstituteParameters:
    def test_substitute_none(self):
        sql = substitute_parameters("SELECT * FROM t WHERE x LIKE 'A%'")
        assert sql == "SELECT * FROM t WHERE x LIKE 'A%'"

    def test_substitute_sequence(self):
        sql = substitute_parameters(
            "SELECT %s, %s FROM t WHERE x LIKE 'A%%'", [1, "foo"]
        )
        assert sql == "SELECT 1, 'foo' FROM t WHERE x LIKE 'A%'"

    def test_substitute_mapping(self):
        sql = substitute_parameters(
            "SELECT %(id)s, %(name)s FROM t WHERE x LIKE 'A%%'",
            {"id": 1, "name": "foo"},
        )
        assert sql == "SELECT 1, 'foo' FROM t WHERE x LIKE 'A%'"

    @pytest.mark.parametrize("value", [1, "foo", object()])
    def test_invalid(self, value):
        with pytest.raises(TypeError):
            substitute_parameters("SELECT %s", value)


class TestIsSelect:
    def test_select(self):
        assert is_select("SELECT 1")

    def test_select_lowercase(self):
        assert is_select("select 1")

    def test_with_select(self):
        assert is_select("WITH (...) AS t SELECT ...")

    def test_with_select_lowercase(self):
        assert is_select("with (...) AS t select ...")

    def test_insert(self):
        assert not is_select("INSERT ... AS SELECT")

    def test_create(self):
        assert not is_select("CREATE TABLE AS ... SELECT")

    def test_select_without_whitesplace(self):
        assert is_select("SELECT*FROM ...")

    def test_single_line_comments(self):
        assert is_select(
            """
            -- Comment 1
            -- Comment 2
            SELECT
        """
        )

    def test_multi_line_comments(self):
        assert is_select(
            """
            /*
            Comment 1
            Comment 2
            */
            SELECT
        """
        )

    def test_multi_line_comment_escaped(self):
        assert is_select(
            r"""
            /* *\/ */
            SELECT
        """
        )


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
        # Leading and trailing new lines are removed.
        # Other new lines are normalized to LF
        sql = normalize_sql(
            """\n\n\n
            SELECT\n\N{space}\N{space}\N{space}\n\r\n
                c1, c2
            FROM
                t\n\n\n
        """
        )
        assert sql == NORMALIZED_SQL.replace("SELECT", "SELECT\n\n\n")
