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

from pallas import Athena
from pallas.normalization import normalize_sql
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


@pytest.fixture(name="fake")
def fake_fixture():
    fake = AthenaFake()
    fake.database = "test_database"
    return fake


@pytest.fixture(name="athena")
def athena_fixture(fake):
    return Athena(fake, normalize=True)


class TestAthenaNormalization:

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
