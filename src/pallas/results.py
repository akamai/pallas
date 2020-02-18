from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union, overload

from pallas.conversions import get_dtype, parse_value

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None
    np = None


QueryRecord = Dict[str, Any]


class QueryResults(Sequence[QueryRecord]):
    """
    Collection of Athena query results.

    Implements list-like interface.
    """

    _column_names: Sequence[str]
    _column_types: Sequence[str]
    _data: Sequence[Sequence[str]]

    def __init__(
        self,
        column_names: Sequence[str],
        column_types: Sequence[str],
        data: Sequence[Sequence[str]],
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
            return QueryResults(self.column_names, self.column_types, self.data[index])
        row = self._data[index]
        info = zip(self._column_names, self._column_types, row)
        return {cn: parse_value(ct, v) for cn, ct, v in info}

    def __len__(self) -> int:
        return len(self._data)

    @property
    def column_names(self) -> Sequence[str]:
        return list(self._column_names)

    @property
    def column_types(self) -> Sequence[str]:
        return list(self._column_types)

    @property
    def data(self) -> Sequence[Sequence[str]]:
        return self._data

    def to_df(self, dtypes: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
        if pd is None:
            raise RuntimeError("Pandas cannot be imported.")
        frame_data = {}
        column_info = zip(self.column_names, self.column_types)
        for i, (column_name, column_type) in enumerate(column_info):
            dtype = get_dtype(column_type)
            if dtypes is not None:
                dtype = dtypes.get(column_name, dtype)
            values = [parse_value(column_type, row[i]) for row in self.data]
            frame_data[column_name] = _pd_array(values, dtype=dtype)
        return pd.DataFrame(frame_data, copy=False)


def _pd_array(values: Sequence[Any], *, dtype: str) -> Any:
    if dtype == "object":
        # Workaround for ValueError: PandasArray must be 1-dimensional.
        # When all values are lists of same length, Pandas/NumPy think
        # that we are constructing a 2-D array.
        data = np.empty(len(values), dtype="object")
        data[:] = values
    else:
        data = values
    return pd.array(data, dtype=dtype, copy=False)
