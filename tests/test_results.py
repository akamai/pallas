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
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pallas.results import QueryResults

sample_column_names = "id", "name"
sample_column_types = "integer", "varchar"

sample_data = [
    ("1", "foo"),
    ("2", "bar"),
]


sample_results = QueryResults(sample_column_names, sample_column_types, sample_data)


class TestQueryResults:
    def test_repr(self):
        assert repr(sample_results) == (
            "<QueryResults:"
            " 2 results,"
            " column_names=['id', 'name'],"
            " column_types=['integer', 'varchar']"
            ">"
        )

    def test_list(self):
        assert list(sample_results) == [
            {"id": 1, "name": "foo"},
            {"id": 2, "name": "bar"},
        ]

    def test_len(self):
        assert len(sample_results) == 2

    def test_get_item(self):
        assert sample_results[0] == {"id": 1, "name": "foo"}

    def test_get_slice(self):
        assert list(sample_results[:1]) == [{"id": 1, "name": "foo"}]

    def test_get_slice_from_end(self):
        assert list(sample_results[-1:]) == [{"id": 2, "name": "bar"}]

    def test_to_df(self):
        actual = sample_results.to_df()
        expected = pd.DataFrame(
            {
                "id": pd.Series([1, 2], dtype="Int32"),
                "name": pd.Series(["foo", "bar"], dtype="string"),
            }
        )
        assert_frame_equal(actual, expected)

    def test_to_df_custom_dtypes(self):
        name_dtype = pd.CategoricalDtype(["foo", "bar"])
        actual = sample_results.to_df(dtypes={"id": "int64", "name": name_dtype})
        expected = pd.DataFrame(
            {
                "id": pd.Series([1, 2], dtype="int64"),
                "name": pd.Series(["foo", "bar"], dtype=name_dtype),
            }
        )
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        "column_type,values,series",
        [
            ("unknown", [None], pd.Series([None], dtype="string")),
            (
                "varchar",
                [None, "", "value"],
                pd.Series([None, "", "value"], dtype="string"),
            ),
            (
                "char",
                [None, "", "value"],
                pd.Series([None, "", "value"], dtype="string"),
            ),
            (
                "boolean",
                [None, "false", "true"],
                pd.Series([None, False, True], dtype="boolean"),
            ),
            (
                "tinyint",
                [None, "-128", "0", "127"],
                pd.Series([None, -(2 ** 7), 0, 2 ** 7 - 1], dtype="Int8"),
            ),
            (
                "smallint",
                [None, "-32768", "0", "32767"],
                pd.Series([None, -(2 ** 15), 0, 2 ** 15 - 1], dtype="Int16"),
            ),
            (
                "integer",
                [None, "-2147483648", "0", "2147483647"],
                pd.Series([None, -(2 ** 31), 0, 2 ** 31 - 1], dtype="Int32"),
            ),
            (
                "bigint",
                [None, "-9223372036854775808", "0", "9223372036854775807"],
                pd.Series([None, -(2 ** 63), 0, 2 ** 63 - 1], dtype="Int64"),
            ),
            (
                "float",
                [None, "NaN", "1.2E34", "-Infinity"],
                pd.Series([np.nan, np.nan, 1.2e34, -np.inf], dtype="float32"),
            ),
            (
                "double",
                [None, "NaN", "1.2E34", "-Infinity"],
                pd.Series([np.nan, np.nan, 1.2e34, -np.inf], dtype="float64"),
            ),
            (
                "decimal",
                [None, "0", "0.1"],
                pd.Series([None, Decimal("0"), Decimal("0.1")], dtype="object"),
            ),
            (
                "date",
                [None, "2001-08-22"],
                pd.Series([pd.NaT, dt.datetime(2001, 8, 22)], dtype="datetime64[ns]"),
            ),
            (
                "timestamp",
                [None, "2001-08-22 03:04:05.321"],
                pd.Series(
                    [pd.NaT, dt.datetime(2001, 8, 22, 3, 4, 5, 321000)],
                    dtype="datetime64[ns]",
                ),
            ),
            (
                "varbinary",
                [None, "", "00 ff 00"],
                pd.Series([None, b"", b"\x00\xff\x00"], dtype="object"),
            ),
            (
                "array",
                [None, "[]", "[a, b]"],
                pd.Series([None, [], ["a", "b"]], dtype="object"),
            ),
            (
                "map",
                [None, "{}", "{k1=v1, k2=v2}"],
                pd.Series([None, {}, {"k1": "v1", "k2": "v2"}], dtype="object"),
            ),
            (
                "json",
                [None, "null", "1", '{"a": []}'],
                pd.Series([None, None, 1, {"a": []}], dtype="object"),
            ),
        ],
    )
    def test_to_df_types(self, column_type, values, series):
        data = [(v,) for v in values]
        results = QueryResults(["col"], [column_type], data)
        assert_frame_equal(
            results.to_df(), pd.DataFrame({"col": series}), check_column_type="exact"
        )
