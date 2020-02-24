"""
Encapsulation of results returned from Athena.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, TextIO, Union, cast, overload

from pallas.conversions import Converter, get_converter
from pallas.csv import read_csv, write_csv

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None


QueryRecord = Dict[str, object]


class QueryResults(Sequence[QueryRecord]):
    """
    Collection of Athena query results.

    Implements list-like interface.
    """

    _column_names: Sequence[str]
    _column_types: Sequence[str]
    _data: Sequence[Sequence[Optional[str]]]

    def __init__(
        self,
        column_names: Sequence[str],
        column_types: Sequence[str],
        data: Sequence[Sequence[Optional[str]]],
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
        self, index: Union[int, slice]
    ) -> Union[QueryRecord, Sequence[QueryRecord]]:
        if isinstance(index, slice):
            return QueryResults(self.column_names, self.column_types, self._data[index])
        row = self._data[index]
        return {
            cn: converter.read(v)
            for cn, converter, v in zip(self.column_names, self.converters, row)
        }

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def load(cls, stream: TextIO) -> QueryResults:
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
        write_csv([self._column_names, self._column_types], stream)
        write_csv(self._data, stream)

    @property
    def column_names(self) -> Sequence[str]:
        return list(self._column_names)

    @property
    def column_types(self) -> Sequence[str]:
        return list(self._column_types)

    @property
    def converters(self) -> Sequence[Converter[object]]:
        return list(map(get_converter, self.column_types))

    def to_df(self, dtypes: Optional[Mapping[str, object]] = None) -> pd.DataFrame:
        if pd is None:
            raise RuntimeError("Pandas cannot be imported.")
        frame_data = {}
        for i, (column_name, converter) in enumerate(
            zip(self.column_names, self.converters)
        ):
            dtype = converter.dtype
            if dtypes is not None:
                dtype = dtypes.get(column_name, dtype)
            values = [converter.read(row[i]) for row in self._data]
            frame_data[column_name] = _pd_array(values, dtype=dtype)
        return pd.DataFrame(frame_data, copy=False)


def _pd_array(values: Sequence[object], *, dtype: object) -> object:
    if dtype == "object":
        # Workaround for ValueError: PandasArray must be 1-dimensional.
        # When all values are lists of same length, Pandas/NumPy think
        # that we are constructing a 2-D array.
        data = np.empty(len(values), dtype="object")
        data[:] = values
    else:
        data = values
    return pd.array(data, dtype=dtype, copy=False)
