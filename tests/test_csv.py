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

import io

import pytest

from pallas.csv import read_csv, write_csv


def read_csv_str(s):
    stream = io.StringIO(s)
    return list(read_csv(stream))


def write_csv_str(data):
    stream = io.StringIO()
    write_csv(data, stream)
    return stream.getvalue()


class TestReadCSV:
    def test_empty(self):
        assert read_csv_str("") == []

    @pytest.mark.parametrize(
        "raw,value",
        [
            ("", None),
            ('""', ""),
            ('"1"', "1"),
            ('"abc"', "abc"),
            ('""""', '"'),
            ('"foo=""bar"""', 'foo="bar"'),
        ],
    )
    def test_value(self, raw, value):
        assert read_csv_str(f"{raw}\n") == [(value,)]

    @pytest.mark.parametrize("raw", ["1", '"', '"x', '""x'])
    def test_invalid_value(self, raw):
        with pytest.raises(ValueError):
            read_csv_str(f"{raw}\n")

    def test_multiple_columns(self):
        assert read_csv_str('"foo","bar"\n') == [("foo", "bar")]

    def test_multiple_columns_with_none(self):
        assert read_csv_str(",\n") == [(None, None)]

    def test_multiple_rows(self):
        assert read_csv_str('"foo"\n"bar"\n') == [("foo",), ("bar",)]

    def test_multiple_rows_with_none(self):
        assert read_csv_str("\n\n") == [(None,), (None,)]

    def test_long(self):
        value = 10_000 * "abc"
        assert read_csv_str(f'"{value}"\n') == [(value,)]

    def test_long_none_value(self):
        value = 10_000 * "abc"
        assert read_csv_str(f'"{value}",\n') == [(value, None)]

    def test_missing_trailing_newline(self):
        with pytest.raises(ValueError):
            read_csv_str('"foo","bar"')

    def test_missing_trailing_newline_after_none(self):
        with pytest.raises(ValueError):
            read_csv_str(",")


class TestWriteCSV:
    def test_empty(self):
        assert write_csv_str([]) == ""

    @pytest.mark.parametrize(
        "raw,value",
        [
            ("", None),
            ('""', ""),
            ('"1"', "1"),
            ('"abc"', "abc"),
            ('""""', '"'),
            ('"foo=""bar"""', 'foo="bar"'),
        ],
    )
    def test_value(self, raw, value):
        assert write_csv_str([(value,)]) == f"{raw}\n"

    def test_multiple_columns(self):
        assert write_csv_str([("foo", "bar")]) == '"foo","bar"\n'

    def test_multiple_columns_with_none(self):
        assert write_csv_str([(None, None)]) == ",\n"

    def test_multiple_rows(self):
        assert write_csv_str([("foo",), ("bar",)]) == '"foo"\n"bar"\n'

    def test_multiple_rows_with_none(self):
        assert write_csv_str([(None,), (None,)]) == "\n\n"

    def test_long(self):
        value = 10_000 * "abc"
        assert write_csv_str([(value,)]) == f'"{value}"\n'

    def test_long_none_value(self):
        value = 10_000 * "abc"
        assert write_csv_str([(value, None)]) == f'"{value}",\n'
