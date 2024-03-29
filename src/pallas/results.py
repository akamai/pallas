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

"""
Encapsulation of results returned from Athena.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, TextIO, cast, overload

from pallas._compat import pandas as pd
from pallas.conversions import Converter, get_converter
from pallas.csv import read_csv, write_csv

QueryRecord = Dict[str, object]


class QueryResults(Sequence[QueryRecord]):
    """
    Collection of Athena query results.

    Implements a list-like interface for accessing individual records.
    Alternatively, can be converted to :class:`pandas.DataFrame`
    using the :meth:`.to_df` method.
    """

    _column_names: Sequence[str]
    _column_types: Sequence[str]
    _data: Sequence[Sequence[str | None]]

    def __init__(
        self,
        column_names: Sequence[str],
        column_types: Sequence[str],
        data: Sequence[Sequence[str | None]],
    ) -> None:
        self._column_names = column_names
        self._column_types = column_types
        self._data = data

    def __repr__(self) -> str:
        parts = [
            f"{len(self)} results",
            f"column_names={self.column_names!r}",
            f"column_types={self.column_types!r}",
        ]
        return f"<{type(self).__name__}: {', '.join(parts)}>"

    @overload
    def __getitem__(self, index: int) -> QueryRecord:
        ...

    @overload  # noqa: F811
    def __getitem__(self, index: slice) -> Sequence[QueryRecord]:
        ...

    def __getitem__(  # noqa: F811
        self, index: int | slice
    ) -> QueryRecord | Sequence[QueryRecord]:
        """
        Return one result or slice of results.

        Records are returned as mappings from column names to values.
        """
        if isinstance(index, slice):
            return QueryResults(self.column_names, self.column_types, self._data[index])
        row = self._data[index]
        return {
            cn: converter.read(v)
            for cn, converter, v in zip(self.column_names, self._converters, row)
        }

    def __len__(self) -> int:
        """
        Return count of this results.
        """
        return len(self._data)

    @classmethod
    def load(cls, stream: TextIO) -> QueryResults:
        """
        Deserialize results from a text stream.
        """
        reader = read_csv(stream)
        column_names = next(reader)
        column_types = next(reader)
        data = list(reader)
        if any(v is None for v in column_names):
            raise ValueError("Missing column name")
        if any(v is None for v in column_types):
            raise ValueError("Missing column type")
        column_names = cast(Sequence[str], column_names)
        column_types = cast(Sequence[str], column_types)
        return cls(column_names, column_types, data)

    def save(self, stream: TextIO) -> None:
        """
        Serialize results to a text stream.
        """
        write_csv([self._column_names, self._column_types], stream)
        write_csv(self._data, stream)

    @property
    def column_names(self) -> Sequence[str]:
        """List of column names."""
        return list(self._column_names)

    @property
    def column_types(self) -> Sequence[str]:
        """List of column types."""
        return list(self._column_types)

    @property
    def _converters(self) -> Sequence[Converter[object]]:
        return list(map(get_converter, self.column_types))

    def to_df(self, dtypes: Mapping[str, object] | None = None) -> pd.DataFrame:
        """
        Convert this results to :class:`pandas.DataFrame`.
        """
        if pd is None:
            raise RuntimeError("Pandas cannot be imported.")
        if dtypes is None:
            dtypes = {}
        frame_data = {}
        for i, (name, converter) in enumerate(zip(self.column_names, self._converters)):
            values = (row[i] for row in self._data)
            frame_data[name] = converter.read_array(values, dtype=dtypes.get(name))
        return pd.DataFrame(frame_data, copy=False)
